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
    new_state,
    prepare_write,
    recompute_counts,
    state_device,
    state_memory_bytes,
)
from .packing import packed_bytes
from .utils import _reshape_blocks, _unshape_blocks, dequantize_sym, fwht_last_dim, quantize_sym, timed


class QuaRotKVQuantizer(KVQuantizer):
    """KV-cache QuaRot phase-A baseline.

    The cache key is stored in post-RoPE Hadamard space and the query uses the
    same orthogonal transform.  This preserves the QK dot product in BF16
    while allowing the rotated key to be quantized per token/head/channel
    group.  V remains ordinary RTN in this phase because V/O compensation is
    a separate model-weight transformation.
    """

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
        if channel_group_size is not None and channel_group_size <= 0:
            raise ValueError("channel_group_size must be > 0")
        resolved_name = name or (
            f"QUAROT_KV_INT{bits}"
            if key_bits == value_bits == bits
            else f"QUAROT_KV_K{key_bits}_V{value_bits}"
        )
        super().__init__(
            bits=bits,
            block_size=block_size,
            name=resolved_name,
            key_bits=key_bits,
            value_bits=value_bits,
        )
        self.channel_group_size = channel_group_size

    def _resolve_channel_group_size(self, head_dim: int) -> int:
        group_size = self.channel_group_size or head_dim
        if group_size > head_dim:
            group_size = head_dim
        if head_dim % group_size != 0:
            raise ValueError(
                f"head_dim={head_dim} must be divisible by channel_group_size={group_size}"
            )
        return int(group_size)

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        return fwht_last_dim(x)

    def _inv_rotate(self, x: torch.Tensor) -> torch.Tensor:
        return fwht_last_dim(x)

    def prepare_attention_qk(
        self,
        roped_query: torch.Tensor,
        roped_key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if roped_query.shape != roped_key.shape:
            raise ValueError("QuaRot requires query and key to have the same [B, L, H, D] shape")
        return self._rotate(roped_query), self._rotate(roped_key)

    def _quantize_rotated_key(self, x: torch.Tensor, bits: int) -> Dict[str, Any]:
        b, l, h, d = x.shape
        group_size = self._resolve_channel_group_size(d)
        groups = d // group_size
        xg = x.reshape(b, l, h, groups, group_size)
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
            "channel_group_size": group_size,
            "axis": "post_rope_token_head_channel_group",
            "rotated": True,
            "tensor_dtype": x.dtype,
        }

    def _quantize_values(self, x: torch.Tensor, bits: int) -> Dict[str, Any]:
        # Phase A keeps V in the original attention space and uses ordinary
        # RTN sequence blocks.
        xb, pad_len = _reshape_blocks(x, self.block_size)
        q, scale = quantize_sym(xb, bits=bits, reduce_dims=(2,))
        return {
            "q": pack_bits(q, bits, signed=True),
            "q_shape": tuple(q.shape),
            "q_numel": int(q.numel()),
            "packed": True,
            "signed": True,
            "scale": scale,
            "pad_len": pad_len,
            "orig_shape": tuple(x.shape),
            "bits": bits,
            "block_size": self.block_size,
            "axis": "sequence_block_rtn",
            "tensor_dtype": x.dtype,
        }

    def _dequantize_rotated_key(self, state: Dict[str, Any]) -> torch.Tensor:
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
        group_size = int(state["channel_group_size"])
        shape = tuple(int(dim) for dim in state["orig_shape"])
        groups = shape[3] // group_size
        xg = dequantize_sym(q, state["scale"], dtype=dtype)
        return xg.reshape(shape[0], shape[1], shape[2], groups * group_size)

    def _dequantize_values(self, state: Dict[str, Any]) -> torch.Tensor:
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
        x = dequantize_sym(q, state["scale"], dtype=dtype)
        return _unshape_blocks(x, int(state["pad_len"]), int(state["orig_shape"][1]))

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
        k_state = self._quantize_rotated_key(write_k, self.key_bits)
        v_state = self._quantize_values(write_v, self.value_bits)
        tensor_dtype = meta.get("tensor_dtype", write_k.dtype)
        k_state["tensor_dtype"] = tensor_dtype
        v_state["tensor_dtype"] = tensor_dtype
        append_segment(state, k_state, v_state, int(write_k.shape[1]))

    def init_state(self, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        state = new_state(meta)
        state["attention_space"] = bool((meta or {}).get("attention_space", False))
        return state

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
        already_rotated = bool(meta.get("already_rotated", False))
        if not already_rotated:
            new_k = self._rotate(new_k)
        # Generic append callers receive the original K back from
        # ``materialize_kv``.  The model runtime opts into attention space
        # explicitly after rotating post-RoPE Q/K together.
        meta["attention_space"] = bool(meta.get("attention_space", already_rotated))
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
            k_parts = []
            v_parts = []
            for segment in state["segments"]:
                k_parts.append(self._dequantize_rotated_key(segment["k"]))
                v_parts.append(self._dequantize_values(segment["v"]))
            residual_k = state.get("residual_k")
            residual_v = state.get("residual_v")
            write_k = state.get("write_k")
            write_v = state.get("write_v")
            if isinstance(residual_k, torch.Tensor) and residual_k.shape[1] > 0:
                k_parts.append(residual_k)
                v_parts.append(residual_v)
            if isinstance(write_k, torch.Tensor) and write_k.shape[1] > 0:
                k_parts.append(write_k)
                v_parts.append(write_v)
            if k_parts:
                k, v = torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)
            else:
                shape = tuple(int(dim) for dim in (meta or {}).get("shape", state.get("shape", (0, 0, 0, 0))))
                dtype = (meta or {}).get("tensor_dtype") or state.get("tensor_dtype") or torch.bfloat16
                device = (meta or {}).get("device", "cpu")
                k = torch.empty((shape[0], 0, shape[2], shape[3]), dtype=dtype, device=device)
                v = k.clone()
            if not state.get("attention_space", False):
                k = self._inv_rotate(k)
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
        meta = dict(meta or {})
        already_rotated = bool(meta.get("already_rotated", False))
        rotated_k = k if already_rotated else self._rotate(k)
        bf16_bytes = int(k.numel() * k.element_size() + v.numel() * v.element_size())
        with timed(k.device, enabled=self.stats.timing_enabled) as timer:
            k_state = self._quantize_rotated_key(rotated_k, self.key_bits)
            v_state = self._quantize_values(v, self.value_bits)
            tensor_dtype = meta.get("tensor_dtype", k.dtype)
            k_state["tensor_dtype"] = tensor_dtype
            v_state["tensor_dtype"] = tensor_dtype
            state = {
                "k": k_state,
                "v": v_state,
                "attention_space": already_rotated,
            }
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
            k = self._dequantize_rotated_key(state["k"])
            v = self._dequantize_values(state["v"])
            if not state.get("attention_space", False):
                k = self._inv_rotate(k)
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
        group_size = self._resolve_channel_group_size(head_dim)
        groups = head_dim // group_size
        key_values = batch_size * active_tokens * num_heads * head_dim
        key_scales = batch_size * active_tokens * num_heads * groups
        num_blocks = (active_tokens + self.block_size - 1) // self.block_size
        value_scales = batch_size * num_blocks * num_heads * head_dim
        value_values = batch_size * num_blocks * self.block_size * num_heads * head_dim
        key_bytes = packed_bytes(key_values, self.key_bits) + key_scales * 2
        value_bytes = packed_bytes(value_values, self.value_bits) + value_scales * 2
        return int(key_bytes + value_bytes)
