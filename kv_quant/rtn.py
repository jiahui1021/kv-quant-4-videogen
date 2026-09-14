from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .base import KVQuantizer
from .bitpack import pack_bits, unpack_bits
from .incremental import (
    append_segment,
    ensure_state,
    evict_prefix,
    evict_range as evict_state_range,
    is_incremental_state,
    materialize,
    new_state,
    prepare_write,
    recompute_counts,
    state_device,
    state_memory_bytes,
)
from .packing import packed_bytes
from .utils import dequantize_sym, quantize_sym, reshape_channel_groups, timed


class RTNQuantizer(KVQuantizer):
    """Blockwise symmetric RTN with an append-only cache representation."""

    #: Implements init_state/append_kv/materialize_kv for an append-only cache.
    supports_incremental_cache = True

    def __init__(
        self,
        bits: int = 4,
        block_size: int = 16,
        key_bits: int | None = None,
        value_bits: int | None = None,
        name: str | None = None,
        channel_group_size: int | None = None,
    ) -> None:
        key_bits = bits if key_bits is None else key_bits
        value_bits = bits if value_bits is None else value_bits
        channel_group_size = block_size if channel_group_size is None else int(channel_group_size)
        if channel_group_size <= 0:
            raise ValueError("channel_group_size must be > 0")
        resolved_name = name or (
            f"RTN_INT{bits}" if key_bits == value_bits == bits else f"RTN_K{key_bits}_V{value_bits}"
        )
        super().__init__(
            bits=bits,
            # ``block_size`` remains a backwards-compatible CLI spelling for
            # RTN's channel group size. It has no sequence-block meaning.
            block_size=channel_group_size,
            name=resolved_name,
            key_bits=key_bits,
            value_bits=value_bits,
        )
        self.channel_group_size = channel_group_size

    def _quantize_tensor(self, x: torch.Tensor, bits: int) -> Dict[str, Any]:
        xg = reshape_channel_groups(x, self.channel_group_size)
        q, scale = quantize_sym(xg, bits=bits, reduce_dims=(-1,))
        return {
            "q": pack_bits(q, bits, signed=True),
            "q_shape": tuple(q.shape),
            "q_numel": int(q.numel()),
            "packed": True,
            "signed": True,
            "scale": scale,
            "orig_shape": tuple(x.shape),
            "bits": bits,
            "channel_group_size": self.channel_group_size,
            "axis": "token_head_channel_group",
            "tensor_dtype": x.dtype,
        }

    def _dequantize_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        q = state["q"]
        if state.get("packed", False):
            q = unpack_bits(
                q,
                int(state["bits"]),
                state["q_shape"],
                int(state["q_numel"]),
                signed=bool(state.get("signed", True)),
            )
        dtype = state.get("tensor_dtype", torch.bfloat16)
        xg = dequantize_sym(q, state["scale"], dtype=dtype)
        return xg.reshape(tuple(int(dim) for dim in state["orig_shape"]))

    def _segment_memory_bytes(self, state: Dict[str, Any]) -> int:
        def bytes_for_tensor(tensor_state: Dict[str, Any]) -> int:
            q = tensor_state["q"]
            scale = tensor_state["scale"]
            q_bytes = int(q.numel() * q.element_size()) if tensor_state.get("packed", False) else packed_bytes(
                int(q.numel()), int(tensor_state.get("bits", self.bits))
            )
            return q_bytes + int(scale.numel() * scale.element_size())

        return bytes_for_tensor(state["k"]) + bytes_for_tensor(state["v"])

    def _commit_write(
        self,
        state: Dict[str, Any],
        write_k: torch.Tensor,
        write_v: torch.Tensor,
        meta: Dict[str, Any],
    ) -> None:
        if write_k.shape[1] == 0:
            return
        tensor_dtype = meta.get("tensor_dtype", write_k.dtype)
        k_state = self._quantize_tensor(write_k, self.key_bits)
        v_state = self._quantize_tensor(write_v, self.value_bits)
        k_state["tensor_dtype"] = tensor_dtype
        v_state["tensor_dtype"] = tensor_dtype
        append_segment(state, k_state, v_state, int(write_k.shape[1]))

    def init_state(self, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return new_state(meta)

    def append_kv(
        self,
        state: Dict[str, Any],
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if new_k.ndim != 4 or new_v.shape != new_k.shape:
            raise ValueError(
                f"Expected new K/V with matching [B, L, H, D] shapes, got "
                f"{tuple(new_k.shape)} and {tuple(new_v.shape)}"
            )
        state = ensure_state(state, meta)
        meta = dict(meta or {})
        meta.setdefault("tensor_dtype", new_k.dtype)
        before_segments = len(state["segments"])
        had_write = isinstance(state.get("write_k"), torch.Tensor)
        with timed(new_k.device, enabled=self.stats.timing_enabled) as timer:
            result = prepare_write(state, new_k, new_v, meta, self._commit_write)
        if result != "replace" and (had_write or len(state["segments"]) > before_segments):
            self.stats.record_quantize(timer)
            self.stats.quantize_calls += 1
        self.stats.bf16_kv_bytes = int(new_k.numel() * new_k.element_size() + new_v.numel() * new_v.element_size())
        self.stats.compressed_kv_bytes = self.memory_bytes(state)
        recompute_counts(state)
        return state

    def finalize_state(
        self,
        state: Dict[str, Any],
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        state = ensure_state(state, meta)
        write_k = state.get("write_k")
        write_v = state.get("write_v")
        if not isinstance(write_k, torch.Tensor) or write_k.shape[1] == 0:
            return state
        meta = dict(meta or {})
        meta.setdefault("tensor_dtype", write_k.dtype)
        with timed(write_k.device, enabled=self.stats.timing_enabled) as timer:
            self._commit_write(state, write_k, write_v, meta)
        if state.get("write_end") is not None:
            state["committed_end"] = int(state["write_end"])
        state["write_k"] = None
        state["write_v"] = None
        recompute_counts(state)
        self.stats.record_quantize(timer)
        self.stats.quantize_calls += 1
        self.stats.compressed_kv_bytes = self.memory_bytes(state)
        return state

    def _materialize_incremental(
        self,
        state: Dict[str, Any],
        meta: Dict[str, Any] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with timed(state_device(state), enabled=self.stats.timing_enabled) as timer:
            k, v = materialize(
                state,
                lambda tensor_state: self._dequantize_tensor(tensor_state),
                meta=meta,
            )
        self.stats.record_dequantize(timer)
        self.stats.dequantize_calls += 1
        return k, v

    def materialize_kv(
        self,
        state: Dict[str, Any],
        meta: Dict[str, Any] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = ensure_state(state, meta)
        return self._materialize_incremental(state, meta=meta)

    def quantize_kv(self, k, v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if k.ndim != 4 or v.shape != k.shape:
            raise ValueError(
                f"Expected K/V with matching [B, L, H, D] shapes, got {tuple(k.shape)} and {tuple(v.shape)}"
            )
        bf16_bytes = int(k.numel() * k.element_size() + v.numel() * v.element_size())
        with timed(k.device, enabled=self.stats.timing_enabled) as timer:
            k_state = self._quantize_tensor(k, self.key_bits)
            v_state = self._quantize_tensor(v, self.value_bits)
            tensor_dtype = (meta or {}).get("tensor_dtype", k.dtype)
            k_state["tensor_dtype"] = tensor_dtype
            v_state["tensor_dtype"] = tensor_dtype
            state = {"k": k_state, "v": v_state}
        self.stats.record_quantize(timer)
        self.stats.quantize_calls += 1
        self.stats.bf16_kv_bytes = bf16_bytes
        self.stats.compressed_kv_bytes = self.memory_bytes(state)
        return state

    def dequantize_kv(
        self,
        state: Dict[str, Any],
        meta: Dict[str, Any] | None = None,
    ) -> Tuple[Any, Any]:
        if is_incremental_state(state):
            return self._materialize_incremental(state, meta=meta)
        with timed(state["k"]["q"].device, enabled=self.stats.timing_enabled) as timer:
            k = self._dequantize_tensor(state["k"])
            v = self._dequantize_tensor(state["v"])
        self.stats.record_dequantize(timer)
        self.stats.dequantize_calls += 1
        return k, v

    def memory_bytes(self, state: Dict[str, Any]) -> int:
        if is_incremental_state(state):
            return state_memory_bytes(state, self._segment_memory_bytes)
        return self._segment_memory_bytes(state)

    def evict_prefix(self, state: Dict[str, Any], requested_tokens: int) -> int:
        state = ensure_state(state)
        return evict_prefix(state, requested_tokens)

    def evict_range(self, state: Dict[str, Any], start_tokens: int, requested_tokens: int) -> int:
        state = ensure_state(state)
        return evict_state_range(state, start_tokens, requested_tokens)

    def estimate_active_kv_bytes(
        self,
        active_tokens: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> int:
        active_tokens = max(int(active_tokens), 0)
        if head_dim % self.channel_group_size:
            raise ValueError(
                f"head_dim={head_dim} must be divisible by "
                f"channel_group_size={self.channel_group_size}"
            )
        q_values = batch_size * active_tokens * num_heads * head_dim
        scale_values = (
            batch_size
            * active_tokens
            * num_heads
            * (head_dim // self.channel_group_size)
        )
        key_bytes = packed_bytes(q_values, self.key_bits) + scale_values * 2
        value_bytes = packed_bytes(q_values, self.value_bits) + scale_values * 2
        return int(key_bytes + value_bytes)
