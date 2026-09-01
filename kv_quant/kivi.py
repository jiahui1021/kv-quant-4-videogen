from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch

from .base import KVQuantizer
from .bitpack import pack_bits_rows, unpack_bits, unpack_bits_rows
from .incremental import (
    is_incremental_state,
    new_state,
    prepare_write,
    state_device,
    tensor_bytes,
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
        key_group_size: int | None = None,
        value_group_size: int | None = None,
    ) -> None:
        key_bits = bits if key_bits is None else key_bits
        value_bits = bits if value_bits is None else value_bits
        key_group_size = block_size if key_group_size is None else int(key_group_size)
        if key_group_size <= 0:
            raise ValueError("key_group_size must be > 0")
        residual_length = key_group_size if residual_length is None else int(residual_length)
        if residual_length < 0:
            raise ValueError("residual_length must be >= 0")
        if residual_length % key_group_size != 0:
            raise ValueError("residual_length must be a multiple of key_group_size")
        if value_group_size is not None and value_group_size <= 0:
            raise ValueError("value_group_size must be > 0")
        resolved_name = name or (
            f"KIVI_INT{bits}" if key_bits == value_bits == bits else f"KIVI_K{key_bits}_V{value_bits}"
        )
        super().__init__(
            bits=bits,
            # Preserve the legacy attribute for integrations, but use the
            # explicit name below within the quantizer implementation.
            block_size=key_group_size,
            name=resolved_name,
            key_bits=key_bits,
            value_bits=value_bits,
        )
        self.residual_length = residual_length
        self.key_group_size = key_group_size
        self.value_group_size = value_group_size

    def _resolve_value_group_size(self, head_dim: int) -> int:
        requested = self.value_group_size or self.key_group_size
        if head_dim % requested != 0:
            raise ValueError(
                f"head_dim={head_dim} must be divisible by value_group_size={requested}"
            )
        return int(requested)

    def _quantize_keys(self, x: torch.Tensor) -> Dict[str, Any]:
        xb, pad_len = _reshape_blocks(x, self.key_group_size)
        # KIVI K: per channel across the sequence tokens in each block.
        q, scale, zero = quantize_asym(
            xb,
            bits=self.key_bits,
            reduce_dims=(2,),
            parameter_dtype=x.dtype,
        )
        if pad_len:
            valid_tokens = self.key_group_size - pad_len
            tail_q, tail_scale, tail_zero = quantize_asym(
                xb[:, -1:, :valid_tokens],
                bits=self.key_bits,
                reduce_dims=(2,),
                parameter_dtype=x.dtype,
            )
            q[:, -1:, :valid_tokens] = tail_q
            q[:, -1:, valid_tokens:] = 0
            scale[:, -1:] = tail_scale
            zero[:, -1:] = tail_zero
        return {
            "q": pack_bits_rows(q, self.key_bits, signed=False, row_dims=2),
            "q_shape": tuple(q.shape),
            "q_numel": int(q.numel()),
            "packed": True,
            "packed_rows": 2,
            "signed": False,
            "scale": scale,
            "zero": zero,
            "pad_len": pad_len,
            "orig_shape": tuple(x.shape),
            "bits": self.key_bits,
            "key_group_size": self.key_group_size,
            "axis": "sequence_group_per_channel",
            "tensor_dtype": x.dtype,
        }

    def _quantize_values(self, x: torch.Tensor) -> Dict[str, Any]:
        if x.ndim != 4:
            raise ValueError(
                f"Expected V tensor [B, L, H, D], got shape={tuple(x.shape)}"
            )
        b, length, h, d = x.shape
        group_size = self._resolve_value_group_size(d)
        groups = d // group_size
        xg = x.reshape(b, length, h, groups, group_size)
        # KIVI V: each token/head/channel-group has its own scale/min-offset.
        q, scale, zero = quantize_asym(
            xg,
            bits=self.value_bits,
            reduce_dims=(-1,),
            parameter_dtype=x.dtype,
        )
        return {
            "q": pack_bits_rows(q, self.value_bits, signed=False, row_dims=2),
            "q_shape": tuple(q.shape),
            "q_numel": int(q.numel()),
            "packed": True,
            "packed_rows": 2,
            "signed": False,
            "scale": scale,
            "zero": zero,
            "pad_len": 0,
            "orig_shape": tuple(x.shape),
            "bits": self.value_bits,
            "key_group_size": self.key_group_size,
            "value_group_size": group_size,
            "axis": "token_head_channel_group",
            "tensor_dtype": x.dtype,
        }

    def _dequantize_keys(self, state: Dict[str, Any]) -> torch.Tensor:
        q = state["q"]
        if state.get("packed", False):
            if state.get("packed_rows") is not None:
                q = unpack_bits_rows(
                    q,
                    int(state["bits"]),
                    state["q_shape"],
                    signed=bool(state.get("signed", False)),
                    row_dims=int(state["packed_rows"]),
                )
            else:
                q = unpack_bits(
                    q,
                    int(state["bits"]),
                    state["q_shape"],
                    int(state["q_numel"]),
                    signed=bool(state.get("signed", False)),
                )
        dtype = state.get("tensor_dtype", torch.bfloat16)
        x = dequantize_asym(q, state["scale"], state["zero"], dtype=dtype)
        return _unshape_blocks(x, int(state["pad_len"]), int(state["orig_shape"][1]))

    def _dequantize_values(self, state: Dict[str, Any]) -> torch.Tensor:
        q = state["q"]
        if state.get("packed", False):
            if state.get("packed_rows") is not None:
                q = unpack_bits_rows(
                    q,
                    int(state["bits"]),
                    state["q_shape"],
                    signed=bool(state.get("signed", False)),
                    row_dims=int(state["packed_rows"]),
                )
            else:
                q = unpack_bits(
                    q,
                    int(state["bits"]),
                    state["q_shape"],
                    int(state["q_numel"]),
                    signed=bool(state.get("signed", False)),
                )
        shape = tuple(int(dim) for dim in state["orig_shape"])
        if len(state["q_shape"]) == 6:
            key_group_size = int(
                state.get("key_group_size", self.key_group_size)
            )
            b, length, h, d = shape
            blocks = (length + key_group_size - 1) // key_group_size
            group_size = int(state["value_group_size"])
            groups = d // group_size
            xg = dequantize_asym(
                q,
                state["scale"],
                state["zero"],
                dtype=state.get("tensor_dtype", torch.bfloat16),
            )
            xb = xg.reshape(b, blocks, key_group_size, h, d)
            return _unshape_blocks(
                xb,
                int(state.get("pad_len", 0)),
                length,
            )
        b, length, h, d = shape
        group_size = int(state["value_group_size"])
        groups = d // group_size
        xg = dequantize_asym(q, state["scale"], state["zero"], dtype=state.get("tensor_dtype", torch.bfloat16))
        return xg.reshape(b, length, h, groups * group_size)

    @staticmethod
    def _tensor_state_memory_bytes(tensor_state: Dict[str, Any]) -> int:
        return int(
            tensor_bytes(tensor_state.get("q"))
            + tensor_bytes(tensor_state.get("scale"))
            + tensor_bytes(tensor_state.get("zero"))
        )

    def _sync_state(self, state: Dict[str, Any]) -> None:
        k_segments = state.setdefault("k_segments", [])
        v_segments = state.setdefault("v_segments", [])
        state["segments"] = k_segments
        state["k"] = k_segments[0]["state"] if k_segments else None
        state["v"] = v_segments[0]["state"] if v_segments else None

        quantized_k = sum(int(item["length"]) for item in k_segments)
        quantized_v = sum(int(item["length"]) for item in v_segments)
        residual_k = state.get("residual_k")
        residual_v = state.get("residual_v")
        write_k = state.get("write_k")
        write_v = state.get("write_v")
        residual_k_tokens = (
            int(residual_k.shape[1]) if isinstance(residual_k, torch.Tensor) else 0
        )
        residual_v_tokens = (
            int(residual_v.shape[1]) if isinstance(residual_v, torch.Tensor) else 0
        )
        write_tokens = (
            int(write_k.shape[1]) if isinstance(write_k, torch.Tensor) else 0
        )
        if isinstance(write_v, torch.Tensor) and int(write_v.shape[1]) != write_tokens:
            raise ValueError("KIVI mutable K/V writes must have equal lengths")
        if quantized_k + residual_k_tokens != quantized_v + residual_v_tokens:
            raise ValueError(
                "KIVI K/V committed histories have different logical lengths: "
                f"{quantized_k + residual_k_tokens} != "
                f"{quantized_v + residual_v_tokens}"
            )

        state["quantized_k_tokens"] = quantized_k
        state["quantized_v_tokens"] = quantized_v
        state["quantized_tokens"] = quantized_k
        state["num_tokens"] = quantized_k + residual_k_tokens + write_tokens

        shape_source = next(
            (
                value
                for value in (residual_k, write_k)
                if isinstance(value, torch.Tensor)
            ),
            None,
        )
        if shape_source is None and k_segments:
            shape = tuple(int(dim) for dim in k_segments[0]["state"]["orig_shape"])
            state["shape"] = (
                shape[0],
                int(state["num_tokens"]),
                shape[2],
                shape[3],
            )
        elif isinstance(shape_source, torch.Tensor):
            state["shape"] = (
                int(shape_source.shape[0]),
                int(state["num_tokens"]),
                int(shape_source.shape[2]),
                int(shape_source.shape[3]),
            )

    def _ensure_kivi_state(
        self,
        state: Dict[str, Any],
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if is_incremental_state(state):
            defaults = new_state(meta)
            for key, value in defaults.items():
                state.setdefault(key, value)
            if "k_segments" not in state:
                paired = list(state.get("segments", []))
                state["k_segments"] = [
                    {
                        "state": item["k"],
                        "k": item["k"],
                        "length": int(item["length"]),
                    }
                    for item in paired
                ]
                state["v_segments"] = [
                    {
                        "state": item["v"],
                        "v": item["v"],
                        "length": int(item["length"]),
                    }
                    for item in paired
                ]
            state.setdefault("v_segments", [])
            state.setdefault("residual_length", self.residual_length)
            self._sync_state(state)
            return state

        old_k = state.get("k")
        old_v = state.get("v")
        converted = new_state(meta)
        converted["residual_length"] = self.residual_length
        converted["k_segments"] = []
        converted["v_segments"] = []
        if old_k is not None and old_v is not None:
            k_length = int(old_k["orig_shape"][1])
            v_length = int(old_v["orig_shape"][1])
            converted["k_segments"].append(
                {"state": old_k, "k": old_k, "length": k_length}
            )
            converted["v_segments"].append(
                {"state": old_v, "v": old_v, "length": v_length}
            )
        state.clear()
        state.update(converted)
        self._sync_state(state)
        return state

    def _append_k_state(
        self,
        state: Dict[str, Any],
        tensor: torch.Tensor,
        tensor_dtype: torch.dtype,
    ) -> None:
        if tensor.shape[1] == 0:
            return
        if tensor.shape[1] % self.key_group_size:
            raise ValueError("KIVI K quantized runs must contain complete key groups")
        packed = self._quantize_keys(tensor)
        packed["tensor_dtype"] = tensor_dtype
        state["k_segments"].append(
            {"state": packed, "k": packed, "length": int(tensor.shape[1])}
        )

    def _append_v_state(
        self,
        state: Dict[str, Any],
        tensor: torch.Tensor,
        tensor_dtype: torch.dtype,
    ) -> None:
        if tensor.shape[1] == 0:
            return
        packed = self._quantize_values(tensor)
        packed["tensor_dtype"] = tensor_dtype
        state["v_segments"].append(
            {"state": packed, "v": packed, "length": int(tensor.shape[1])}
        )

    def _commit_write(
        self,
        state: Dict[str, Any],
        write_k: torch.Tensor,
        write_v: torch.Tensor,
        meta: Dict[str, Any],
    ) -> None:
        residual_k = state.get("residual_k")
        residual_v = state.get("residual_v")
        buffered_k = (
            torch.cat((residual_k, write_k), dim=1)
            if isinstance(residual_k, torch.Tensor) and residual_k.shape[1] > 0
            else write_k
        )
        buffered_v = (
            torch.cat((residual_v, write_v), dim=1)
            if isinstance(residual_v, torch.Tensor) and residual_v.shape[1] > 0
            else write_v
        )

        if self.residual_length > 0:
            k_flush = (
                int(buffered_k.shape[1]) // self.residual_length
            ) * self.residual_length
        else:
            k_flush = (
                int(buffered_k.shape[1]) // self.key_group_size
            ) * self.key_group_size
        v_flush = max(int(buffered_v.shape[1]) - self.residual_length, 0)
        tensor_dtype = meta.get("tensor_dtype", write_k.dtype)
        self._append_k_state(state, buffered_k[:, :k_flush], tensor_dtype)
        self._append_v_state(state, buffered_v[:, :v_flush], tensor_dtype)
        state["residual_k"] = buffered_k[:, k_flush:].contiguous()
        state["residual_v"] = buffered_v[:, v_flush:].contiguous()
        self._sync_state(state)

    def init_state(self, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        state = new_state(meta)
        state["residual_length"] = self.residual_length
        state["k_segments"] = []
        state["v_segments"] = []
        self._sync_state(state)
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
        state = self._ensure_kivi_state(state, meta)
        state.setdefault("residual_length", self.residual_length)
        meta = dict(meta or {})
        meta.setdefault("tensor_dtype", new_k.dtype)
        before_segments = (
            len(state["k_segments"]),
            len(state["v_segments"]),
        )
        had_write = isinstance(state.get("write_k"), torch.Tensor)
        with timed(new_k.device, enabled=self.stats.timing_enabled) as timer:
            result = prepare_write(state, new_k, new_v, meta, self._commit_write)
        self._sync_state(state)
        after_segments = (
            len(state["k_segments"]),
            len(state["v_segments"]),
        )
        if result != "replace" and (had_write or after_segments != before_segments):
            self.stats.record_quantize(timer)
            self.stats.quantize_calls += 1
        self.stats.bf16_kv_bytes = int(new_k.numel() * new_k.element_size() + new_v.numel() * new_v.element_size())
        self.stats.compressed_kv_bytes = self.memory_bytes(state)
        return state

    def finalize_state(
        self,
        state: Dict[str, Any],
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        state = self._ensure_kivi_state(state, meta)
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
        self._sync_state(state)
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
            for segment in state["k_segments"]:
                k_parts.append(self._dequantize_keys(segment["state"]))
            for segment in state["v_segments"]:
                v_parts.append(self._dequantize_values(segment["state"]))
            residual_k = state.get("residual_k")
            residual_v = state.get("residual_v")
            write_k = state.get("write_k")
            write_v = state.get("write_v")
            if isinstance(residual_k, torch.Tensor) and residual_k.shape[1] > 0:
                k_parts.append(residual_k)
            if isinstance(residual_v, torch.Tensor) and residual_v.shape[1] > 0:
                v_parts.append(residual_v)
            if isinstance(write_k, torch.Tensor) and write_k.shape[1] > 0:
                k_parts.append(write_k)
            if isinstance(write_v, torch.Tensor) and write_v.shape[1] > 0:
                v_parts.append(write_v)
            if k_parts or v_parts:
                if not k_parts or not v_parts:
                    raise ValueError("KIVI K/V materialization streams are incomplete")
                k = torch.cat(k_parts, dim=1)
                v = torch.cat(v_parts, dim=1)
                if k.shape[1] != v.shape[1]:
                    raise ValueError(
                        f"KIVI K/V materialized lengths differ: {k.shape[1]} != {v.shape[1]}"
                    )
            else:
                shape = tuple(int(dim) for dim in (meta or {}).get("shape", state.get("shape", (0, 0, 0, 0))))
                dtype = (meta or {}).get("tensor_dtype") or state.get("tensor_dtype") or torch.bfloat16
                device = (meta or {}).get("device") or state.get("device") or "cpu"
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
        state = self._ensure_kivi_state(state, meta)
        return self._materialize_incremental(state, meta=meta)

    def quantize_kv(self, k, v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if k.ndim != 4 or v.shape != k.shape:
            raise ValueError(
                f"Expected K/V with matching [B, L, H, D] shapes, got {tuple(k.shape)} and {tuple(v.shape)}"
            )
        bf16_bytes = int(k.numel() * k.element_size() + v.numel() * v.element_size())
        with timed(k.device, enabled=self.stats.timing_enabled) as timer:
            tensor_dtype = (meta or {}).get("tensor_dtype", k.dtype)
            state_meta = dict(meta or {})
            state_meta.setdefault("shape", (k.shape[0], 0, k.shape[2], k.shape[3]))
            state_meta.setdefault("tensor_dtype", tensor_dtype)
            state_meta.setdefault("device", k.device)
            state = self.init_state(state_meta)
            self._commit_write(state, k, v, state_meta)
            if state_meta.get("absolute_end") is not None:
                state["committed_end"] = int(state_meta["absolute_end"])
            self._sync_state(state)
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
        state = self._ensure_kivi_state(state, meta)
        return self._materialize_incremental(state, meta=meta)

    def memory_bytes(self, state: Dict[str, Any]) -> int:
        state = self._ensure_kivi_state(state)
        total = sum(
            self._tensor_state_memory_bytes(item["state"])
            for item in state["k_segments"]
        )
        total += sum(
            self._tensor_state_memory_bytes(item["state"])
            for item in state["v_segments"]
        )
        for name in ("residual_k", "residual_v", "write_k", "write_v"):
            total += tensor_bytes(state.get(name))
        return int(total)

    def _slice_key_state_front(
        self,
        tensor_state: Dict[str, Any],
        tokens: int,
    ) -> None:
        tokens = int(tokens)
        if tokens % self.key_group_size:
            raise ValueError("KIVI K eviction must be key-group aligned")
        groups = tokens // self.key_group_size
        q_shape = list(int(dim) for dim in tensor_state["q_shape"])
        if groups <= 0 or groups >= q_shape[1]:
            raise ValueError("Partial KIVI K slice must leave at least one group")
        tensor_state["q"] = tensor_state["q"][:, groups:].contiguous()
        tensor_state["scale"] = tensor_state["scale"][:, groups:].contiguous()
        tensor_state["zero"] = tensor_state["zero"][:, groups:].contiguous()
        q_shape[1] -= groups
        tensor_state["q_shape"] = tuple(q_shape)
        tensor_state["q_numel"] = math.prod(q_shape)
        orig_shape = list(int(dim) for dim in tensor_state["orig_shape"])
        orig_shape[1] -= tokens
        tensor_state["orig_shape"] = tuple(orig_shape)
        tensor_state["pad_len"] = 0

    @staticmethod
    def _slice_value_state_front(
        tensor_state: Dict[str, Any],
        tokens: int,
    ) -> None:
        tokens = int(tokens)
        q_shape = list(int(dim) for dim in tensor_state["q_shape"])
        if tokens <= 0 or tokens >= q_shape[1]:
            raise ValueError("Partial KIVI V slice must leave at least one token")
        tensor_state["q"] = tensor_state["q"][:, tokens:].contiguous()
        tensor_state["scale"] = tensor_state["scale"][:, tokens:].contiguous()
        tensor_state["zero"] = tensor_state["zero"][:, tokens:].contiguous()
        q_shape[1] -= tokens
        tensor_state["q_shape"] = tuple(q_shape)
        tensor_state["q_numel"] = math.prod(q_shape)
        orig_shape = list(int(dim) for dim in tensor_state["orig_shape"])
        orig_shape[1] -= tokens
        tensor_state["orig_shape"] = tuple(orig_shape)

    def _evict_stream_prefix(
        self,
        state: Dict[str, Any],
        *,
        stream: str,
        residual: str,
        requested: int,
    ) -> int:
        items = state[stream]
        remaining = int(requested)
        removed = 0
        while items and remaining > 0:
            length = int(items[0]["length"])
            if remaining >= length:
                items.pop(0)
                remaining -= length
                removed += length
                continue
            if stream == "k_segments":
                self._slice_key_state_front(items[0]["state"], remaining)
            else:
                self._slice_value_state_front(items[0]["state"], remaining)
            items[0]["length"] = length - remaining
            removed += remaining
            remaining = 0

        residual_tensor = state.get(residual)
        if remaining > 0 and isinstance(residual_tensor, torch.Tensor):
            take = min(remaining, int(residual_tensor.shape[1]))
            state[residual] = residual_tensor[:, take:].contiguous()
            remaining -= take
            removed += take
        return removed

    def _evict_k_prefix(self, state: Dict[str, Any], requested: int) -> int:
        return self._evict_stream_prefix(
            state,
            stream="k_segments",
            residual="residual_k",
            requested=requested,
        )

    def _evict_v_prefix(self, state: Dict[str, Any], requested: int) -> int:
        return self._evict_stream_prefix(
            state,
            stream="v_segments",
            residual="residual_v",
            requested=requested,
        )

    def evict_prefix(self, state: Dict[str, Any], requested_tokens: int) -> int:
        state = self._ensure_kivi_state(state)
        requested = max(int(requested_tokens), 0)
        removable = (requested // self.key_group_size) * self.key_group_size
        committed = int(state["num_tokens"]) - (
            int(state["write_k"].shape[1])
            if isinstance(state.get("write_k"), torch.Tensor)
            else 0
        )
        removable = min(removable, (committed // self.key_group_size) * self.key_group_size)
        if removable <= 0:
            return 0
        removed_k = self._evict_k_prefix(state, removable)
        removed_v = self._evict_v_prefix(state, removable)
        if removed_k != removable or removed_v != removable:
            raise RuntimeError(
                f"KIVI eviction removed inconsistent lengths: K={removed_k}, V={removed_v}, "
                f"expected={removable}"
            )
        self._sync_state(state)
        return removable

    def evict_range(self, state: Dict[str, Any], start_tokens: int, requested_tokens: int) -> int:
        state = self._ensure_kivi_state(state)
        if int(start_tokens) != 0:
            return 0
        return self.evict_prefix(state, requested_tokens)

    def estimate_active_kv_bytes(
        self,
        active_tokens: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> int:
        active_tokens = max(int(active_tokens), 0)
        if self.residual_length > 0:
            quantized_k = (
                active_tokens // self.residual_length
            ) * self.residual_length
        else:
            quantized_k = (
                active_tokens // self.key_group_size
            ) * self.key_group_size
        quantized_v = max(active_tokens - self.residual_length, 0)
        key_blocks = quantized_k // self.key_group_size
        value_group_size = self._resolve_value_group_size(head_dim)
        value_groups = head_dim // value_group_size
        key_payload = (
            batch_size
            * key_blocks
            * packed_bytes(
                self.key_group_size * num_heads * head_dim,
                self.key_bits,
            )
        )
        value_payload = (
            batch_size
            * quantized_v
            * packed_bytes(num_heads * head_dim, self.value_bits)
        )
        key_params = batch_size * key_blocks * num_heads * head_dim * 2 * 2
        value_params = (
            batch_size * quantized_v * num_heads * value_groups * 2 * 2
        )
        residual_values = (
            (active_tokens - quantized_k) + (active_tokens - quantized_v)
        ) * batch_size * num_heads * head_dim
        bf16_residual = residual_values * 2
        key_bytes = key_payload + key_params
        value_bytes = value_payload + value_params
        return int(key_bytes + value_bytes + bf16_residual)
