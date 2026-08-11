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
from .utils import _reshape_blocks, _unshape_blocks, dequantize_asym, quantize_asym, timed


class KIVIQuantizer(KVQuantizer):
    """KIVI-style asymmetric quantization with faithful K/V axes.

    K is quantized per channel over sequence groups.  V is quantized per
    token, per head, and per channel group; heads therefore never share a
    value scale.  The incremental path keeps a configurable recent BF16
    residual and migrates only complete sequence blocks.
    """

    def __init__(
        self,
        bits: int = 4,
        block_size: int = 16,
        key_bits: int | None = None,
        value_bits: int | None = None,
        name: str | None = None,
        residual_length: int | None = None,
        value_group_size: int | None = None,
    ) -> None:
        key_bits = bits if key_bits is None else key_bits
        value_bits = bits if value_bits is None else value_bits
        residual_length = block_size if residual_length is None else int(residual_length)
        if residual_length < 0:
            raise ValueError("residual_length must be >= 0")
        if residual_length % block_size != 0:
            raise ValueError("residual_length must be a multiple of block_size")
        if value_group_size is not None and value_group_size <= 0:
            raise ValueError("value_group_size must be > 0")
        resolved_name = name or (
            f"KIVI_INT{bits}" if key_bits == value_bits == bits else f"KIVI_K{key_bits}_V{value_bits}"
        )
        super().__init__(
            bits=bits,
            block_size=block_size,
            name=resolved_name,
            key_bits=key_bits,
            value_bits=value_bits,
        )
        self.residual_length = residual_length
        self.value_group_size = value_group_size

    def _resolve_value_group_size(self, head_dim: int) -> int:
        requested = self.value_group_size or self.block_size
        if requested > head_dim:
            requested = head_dim
        if head_dim % requested != 0:
            if self.value_group_size is not None:
                raise ValueError(
                    f"head_dim={head_dim} must be divisible by value_group_size={requested}"
                )
            # Keep the default useful for non-Wan head dimensions while
            # retaining an explicit group size as a strict user contract.
            requested = torch.gcd(
                torch.tensor(int(head_dim)), torch.tensor(int(requested))
            ).item()
            if requested <= 0:
                raise ValueError(f"Could not derive a value group size for head_dim={head_dim}")
        return int(requested)

    def _quantize_keys(self, x: torch.Tensor) -> Dict[str, Any]:
        xb, pad_len = _reshape_blocks(x, self.block_size)
        # KIVI K: per channel across the sequence tokens in each block.
        q, scale, zp = quantize_asym(xb, bits=self.key_bits, reduce_dims=(2,))
        return {
            "q": pack_bits(q, self.key_bits, signed=False),
            "q_shape": tuple(q.shape),
            "q_numel": int(q.numel()),
            "packed": True,
            "signed": False,
            "scale": scale,
            "zp": zp,
            "pad_len": pad_len,
            "orig_shape": tuple(x.shape),
            "bits": self.key_bits,
            "block_size": self.block_size,
            "axis": "sequence_group_per_channel",
            "tensor_dtype": x.dtype,
        }

    def _quantize_values(self, x: torch.Tensor) -> Dict[str, Any]:
        xb, pad_len = _reshape_blocks(x, self.block_size)
        b, nb, block, h, d = xb.shape
        group_size = self._resolve_value_group_size(d)
        groups = d // group_size
        xg = xb.reshape(b, nb, block, h, groups, group_size)
        # KIVI V: each token/head/channel-group has its own scale and zp.
        q, scale, zp = quantize_asym(xg, bits=self.value_bits, reduce_dims=(-1,))
        return {
            "q": pack_bits(q, self.value_bits, signed=False),
            "q_shape": tuple(q.shape),
            "q_numel": int(q.numel()),
            "packed": True,
            "signed": False,
            "scale": scale,
            "zp": zp,
            "pad_len": pad_len,
            "orig_shape": tuple(x.shape),
            "bits": self.value_bits,
            "block_size": self.block_size,
            "value_group_size": group_size,
            "axis": "token_head_channel_group",
            "tensor_dtype": x.dtype,
        }

    def _dequantize_keys(self, state: Dict[str, Any]) -> torch.Tensor:
        q = state["q"]
        if state.get("packed", False):
            q = unpack_bits(
                q,
                int(state["bits"]),
                state["q_shape"],
                int(state["q_numel"]),
                signed=bool(state.get("signed", False)),
            )
        dtype = state.get("tensor_dtype", torch.bfloat16)
        x = dequantize_asym(q, state["scale"], state["zp"], dtype=dtype)
        return _unshape_blocks(x, int(state["pad_len"]), int(state["orig_shape"][1]))

    def _dequantize_values(self, state: Dict[str, Any]) -> torch.Tensor:
        q = state["q"]
        if state.get("packed", False):
            q = unpack_bits(
                q,
                int(state["bits"]),
                state["q_shape"],
                int(state["q_numel"]),
                signed=bool(state.get("signed", False)),
            )
        shape = tuple(int(dim) for dim in state["orig_shape"])
        b, nb, block, h, d = shape[0], (shape[1] + self.block_size - 1) // self.block_size, self.block_size, shape[2], shape[3]
        group_size = int(state["value_group_size"])
        groups = d // group_size
        xg = dequantize_asym(q, state["scale"], state["zp"], dtype=state.get("tensor_dtype", torch.bfloat16))
        xb = xg.reshape(b, nb, block, h, d)
        return _unshape_blocks(xb, int(state["pad_len"]), int(shape[1]))

    def _dequantize_segment(self, state: Dict[str, Any]) -> torch.Tensor:
        # The caller selects K or V through the state object wrapper.
        raise RuntimeError("Use _dequantize_kv_segment for a K/V segment")

    def _segment_memory_bytes(self, state: Dict[str, Any]) -> int:
        def bytes_for_tensor(tensor_state: Dict[str, Any]) -> int:
            q = tensor_state["q"]
            scale = tensor_state["scale"]
            zp = tensor_state["zp"]
            q_bytes = int(q.numel() * q.element_size()) if tensor_state.get("packed", False) else packed_bytes(
                int(q.numel()), int(tensor_state.get("bits", self.bits))
            )
            return q_bytes + int(scale.numel() * scale.element_size()) + int(zp.numel() * zp.element_size())

        return bytes_for_tensor(state["k"]) + bytes_for_tensor(state["v"])

    def _commit_write(
        self,
        state: Dict[str, Any],
        write_k: torch.Tensor,
        write_v: torch.Tensor,
        meta: Dict[str, Any],
    ) -> None:
        residual_k = state.get("residual_k")
        residual_v = state.get("residual_v")
        if isinstance(residual_k, torch.Tensor) and residual_k.shape[1] > 0:
            buffered_k = torch.cat((residual_k, write_k), dim=1)
            buffered_v = torch.cat((residual_v, write_v), dim=1)
        else:
            buffered_k, buffered_v = write_k, write_v

        eligible = max(int(buffered_k.shape[1]) - self.residual_length, 0)
        full_length = (eligible // self.block_size) * self.block_size
        tensor_dtype = meta.get("tensor_dtype", write_k.dtype)
        for start in range(0, full_length, self.block_size):
            end = start + self.block_size
            k_state = self._quantize_keys(buffered_k[:, start:end])
            v_state = self._quantize_values(buffered_v[:, start:end])
            k_state["tensor_dtype"] = tensor_dtype
            v_state["tensor_dtype"] = tensor_dtype
            append_segment(state, k_state, v_state, self.block_size)

        state["residual_k"] = buffered_k[:, full_length:].contiguous()
        state["residual_v"] = buffered_v[:, full_length:].contiguous()

    def init_state(self, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        state = new_state(meta)
        state["residual_length"] = self.residual_length
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
        state.setdefault("residual_length", self.residual_length)
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
                k_parts.append(self._dequantize_keys(segment["k"]))
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
            k_state = self._quantize_keys(k)
            v_state = self._quantize_values(v)
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
            k = self._dequantize_keys(state["k"])
            v = self._dequantize_values(state["v"])
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
        eligible = max(active_tokens - self.residual_length, 0)
        quantized_tokens = (eligible // self.block_size) * self.block_size
        num_blocks = quantized_tokens // self.block_size
        q_values = batch_size * quantized_tokens * num_heads * head_dim
        key_scale_values = batch_size * num_blocks * num_heads * head_dim
        value_group_size = self._resolve_value_group_size(head_dim)
        value_groups = head_dim // value_group_size
        value_scale_values = batch_size * num_blocks * self.block_size * num_heads * value_groups
        key_bytes = packed_bytes(q_values, self.key_bits) + key_scale_values * 2 + key_scale_values * 2
        value_bytes = packed_bytes(q_values, self.value_bits) + value_scale_values * 2 + value_scale_values * 2
        bf16_residual = (active_tokens - quantized_tokens) * batch_size * num_heads * head_dim * 2 * 2
        return int(key_bytes + value_bytes + bf16_residual)
