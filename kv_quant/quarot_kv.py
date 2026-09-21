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
from .utils import (
    COMPARISON_GROUP_SIZE,
    SCALE_STORAGE_BYTES,
    SCALE_STORAGE_DTYPE,
    SIGNED_ZERO_STORAGE_DTYPE,
    ZERO_STORAGE_BYTES,
    fwht_last_dim,
    quarot_dequantize,
    quarot_find_params,
    quarot_find_params_groupwise,
    quarot_quantize,
    round_scale_to_storage,
    timed,
)


class QuaRotKVQuantizer(KVQuantizer):
    """KV-cache-only QuaRot, following spcl/QuaRot.

    The reference (``fake_quant/rotation_utils.py::QKRotationWrapper`` and
    ``fake_quant/rotation_utils.py::rotate_ov_proj``) does three things that
    together define the method:

    * Q and K are rotated by the same per-head Hadamard **after RoPE** and
      both stay rotated; the K cache holds the rotated keys.  QuaRot refuses
      pre-RoPE K quantization outright (``k_pre_rope`` raises).
    * V is produced already rotated, because a head-sized Hadamard is folded
      into ``v_proj``'s output weights and its inverse into ``o_proj``'s
      input weights.  With BF16 weights the identical, weight-free form is to
      cache the rotated V and undo the rotation on the attention output --
      attention is linear in V, so ``H(sum_j a_j v_j) == sum_j a_j (H v_j)``.
    * Both caches are quantized per token (``ActQuantizer`` only supports
      per-token), over one group of ``head_dim`` channels or over the whole
      hidden size.  K always goes through ``find_params`` with
      ``groupsize=-1`` (head-wise K is reshaped to ``[-1, head_dim]`` rows
      first), whose range is widened to include zero.  V is ``v_proj``'s
      output quantizer, so head-wise V takes
      ``find_params_per_token_groupwise`` and its range is *not* widened.

    Defaults follow the paper's KV setting (Section 5: "asymmetric
    quantization with a group size 128 with a constant clipping ratio of
    0.95"), with group size ``head_dim`` (128 for Wan and LongCat).

    ``materialize_kv`` therefore returns tensors in *attention space* when the
    state says so: the caller must rotate Q with :meth:`prepare_attention_qk`
    and pass the attention output through :meth:`restore_attention_output`.
    A caller that does neither gets the cache rotated back to the original
    basis instead, which stays numerically correct but is a plain
    rotate-quantize-unrotate round trip.
    """

    #: QuaRot asserts ``k_groupsize in [-1, head_dim]``.
    TOKEN_WISE_GROUP = -1
    #: Channels per scale in the shared comparison setting;
    #: ``channel_group_size=head_dim`` reproduces QuaRot's own KV setting.
    COMPARISON_GROUP = COMPARISON_GROUP_SIZE
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
        asym: bool = True,
        clip_ratio: float = 0.95,
        rotate_value: bool = True,
    ) -> None:
        key_bits = bits if key_bits is None else key_bits
        value_bits = bits if value_bits is None else value_bits
        if channel_group_size is not None and channel_group_size != self.TOKEN_WISE_GROUP:
            if channel_group_size <= 0:
                raise ValueError("channel_group_size must be > 0 or -1")
        if not 0.0 < float(clip_ratio) <= 1.0:
            raise ValueError("clip_ratio must be in (0, 1]")
        prefix = "QUAROT_KV" if rotate_value else "HADAMARD_K"
        resolved_name = name or (
            f"{prefix}_INT{bits}"
            if key_bits == value_bits == bits
            else f"{prefix}_K{key_bits}_V{value_bits}"
        )
        super().__init__(
            bits=bits,
            block_size=block_size,
            name=resolved_name,
            key_bits=key_bits,
            value_bits=value_bits,
        )
        self.channel_group_size = channel_group_size
        self.asym = bool(asym)
        self.sym = not self.asym
        self.clip_ratio = float(clip_ratio)
        self.rotate_value = bool(rotate_value)
        self.requires_special_attention_backend = True

    # -- rotation ---------------------------------------------------------

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        return fwht_last_dim(x)

    #: The normalized Hadamard is involutory, so one transform is its inverse.
    _inv_rotate = _rotate

    def prepare_attention_qk(
        self,
        roped_query: torch.Tensor,
        roped_key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate a post-RoPE Q/K pair, exactly as ``QKRotationWrapper`` does."""
        if roped_query.shape[-1] != roped_key.shape[-1]:
            raise ValueError("QuaRot requires query and key to share head_dim")
        return self._rotate(roped_query), self._rotate(roped_key)

    def prepare_value(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the ``v_proj`` output rotation that QuaRot folds into weights."""
        return self._rotate(value) if self.rotate_value else value

    def restore_attention_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Undo the V rotation, standing in for ``o_proj``'s fused inverse."""
        return self._inv_rotate(attn_output) if self.rotate_value else attn_output

    # -- quantization -----------------------------------------------------

    def _resolve_group(self, head_dim: int, num_heads: int) -> int:
        """Channels that share one scale.

        The Hadamard is always one ``head_dim`` block, which is what makes the
        rotation exact; the quantizer's group is a separate choice.  The
        default 64 is the shared comparison group, and ``head_dim`` reproduces
        QuaRot's own KV setting.
        """
        group = self.channel_group_size
        if group is None:
            return int(min(self.COMPARISON_GROUP, head_dim))
        if group == self.TOKEN_WISE_GROUP:
            return int(num_heads * head_dim)
        if group == num_heads * head_dim:
            return int(group)
        if group <= 0 or head_dim % group:
            raise ValueError(
                "QuaRot groups must be token-wise (-1) or divide head_dim; got "
                f"channel_group_size={group} for head_dim={head_dim}"
            )
        return int(group)

    def _grouped(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``[B, L, H, D]`` so the last dim is one quantization group."""
        b, l, h, d = x.shape
        group = self._resolve_group(d, h)
        return x.reshape(b, l, (h * d) // group, group)

    #: Values rotated and quantized per chunk.  The FWHT works in float32 and
    #: allocates per level, so a whole long cache at once is ~10x its size.
    _CHUNK_VALUES = 1 << 24

    def _quantize_rotated(
        self,
        x: torch.Tensor,
        bits: int,
        *,
        is_value: bool = False,
        rotate=None,
    ) -> Dict[str, Any]:
        """Quantize ``x`` per token, applying ``rotate`` chunk by chunk first.

        Every step is per token (the Hadamard acts on the last dimension and
        the ranges are per token group), so chunking along the sequence gives
        the same codes and scales as one full-size pass.
        """
        if is_value and self.channel_group_size != self.TOKEN_WISE_GROUP:
            find_params = quarot_find_params_groupwise
        else:
            find_params = quarot_find_params
        b, l, h, d = x.shape
        group = self._resolve_group(d, h)
        groups = (h * d) // group
        q = torch.empty((b, l, groups, group), dtype=torch.int8, device=x.device)
        scales, zeros = [], []
        tensor_dtype = x.dtype
        chunk = max(self._CHUNK_VALUES // max(b * h * d, 1), 1)
        for start in range(0, l, chunk):
            xc = x[:, start:start + chunk]
            if rotate is not None:
                xc = rotate(xc)
                tensor_dtype = xc.dtype
            xg = self._grouped(xc)
            scale, zero = find_params(xg, bits, self.sym, self.clip_ratio)
            scale = round_scale_to_storage(scale)
            q[:, start:start + chunk] = quarot_quantize(xg, scale, zero, bits, self.sym)
            scales.append(scale.to(SCALE_STORAGE_DTYPE))
            if not self.sym:
                zeros.append(zero.clamp(-128, 127).to(SIGNED_ZERO_STORAGE_DTYPE))
            del xc, xg, scale, zero
        empty_scale = torch.empty((b, 0, groups, 1), dtype=SCALE_STORAGE_DTYPE, device=x.device)
        empty_zero = torch.empty((b, 0, groups, 1), dtype=SIGNED_ZERO_STORAGE_DTYPE, device=x.device)
        state = {
            "q": pack_bits(q, bits, signed=self.sym),
            "q_shape": tuple(q.shape),
            "q_numel": int(q.numel()),
            "packed": True,
            "signed": self.sym,
            "scale": torch.cat(scales, dim=1) if scales else empty_scale,
            "orig_shape": tuple(x.shape),
            "bits": bits,
            "sym": self.sym,
            "channel_group_size": int(group),
            "axis": "post_rope_token_group",
            "rotated": True,
            "tensor_dtype": tensor_dtype,
        }
        if not self.sym:
            state["zero"] = torch.cat(zeros, dim=1) if zeros else empty_zero
        return state

    def _dequantize_rotated(self, state: Dict[str, Any]) -> torch.Tensor:
        q = state["q"]
        sym = bool(state.get("sym", True))
        if state.get("packed", False):
            q = unpack_bits(
                q,
                int(state["bits"]),
                state["q_shape"],
                int(state["q_numel"]),
                signed=bool(state.get("signed", sym)),
            )
        dtype = state.get("tensor_dtype", torch.bfloat16)
        zero = state.get("zero")
        if zero is None:
            zero = torch.zeros_like(state["scale"])
        x = quarot_dequantize(q, state["scale"], zero, sym, dtype=dtype)
        return x.reshape(tuple(int(dim) for dim in state["orig_shape"]))

    def _segment_memory_bytes(self, state: Dict[str, Any]) -> int:
        def bytes_for_tensor(tensor_state: Dict[str, Any]) -> int:
            q = tensor_state["q"]
            q_bytes = int(q.numel() * q.element_size()) if tensor_state.get("packed", False) else packed_bytes(
                int(q.numel()), int(tensor_state.get("bits", self.bits))
            )
            total = q_bytes
            for key in ("scale", "zero"):
                param = tensor_state.get(key)
                if isinstance(param, torch.Tensor):
                    total += int(param.numel() * param.element_size())
            return total

        return bytes_for_tensor(state["k"]) + bytes_for_tensor(state["v"])

    # -- incremental cache -------------------------------------------------

    def _commit_write(
        self,
        state: Dict[str, Any],
        write_k: torch.Tensor,
        write_v: torch.Tensor,
        meta: Dict[str, Any],
    ) -> None:
        if write_k.shape[1] == 0:
            return
        k_state = self._quantize_rotated(write_k, self.key_bits)
        v_state = self._quantize_rotated(write_v, self.value_bits, is_value=True)
        tensor_dtype = meta.get("tensor_dtype", write_k.dtype)
        k_state["tensor_dtype"] = tensor_dtype
        v_state["tensor_dtype"] = tensor_dtype
        append_segment(state, k_state, v_state, int(write_k.shape[1]))

    def init_state(self, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        state = new_state(meta)
        state["attention_space"] = bool((meta or {}).get("attention_space", False))
        return state

    def _rotate_new_kv(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        already_rotated: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if already_rotated:
            return new_k, new_v
        return self._rotate(new_k), self.prepare_value(new_v)

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
        bf16_bytes = int(
            new_k.numel() * new_k.element_size() + new_v.numel() * new_v.element_size()
        )
        already_rotated = bool(meta.get("already_rotated", False))
        new_k, new_v = self._rotate_new_kv(new_k, new_v, already_rotated)
        # A generic caller gets the original basis back from ``materialize_kv``.
        # The model runtime opts into attention space explicitly once it
        # rotates post-RoPE Q/K together and compensates the attention output.
        meta["attention_space"] = bool(meta.get("attention_space", already_rotated))
        before_segments = len(state["segments"])
        had_write = isinstance(state.get("write_k"), torch.Tensor)
        with timed(new_k.device, enabled=self.stats.timing_enabled) as timer:
            result = prepare_write(state, new_k, new_v, meta, self._commit_write)
        if result != "replace" and (had_write or len(state["segments"]) > before_segments):
            self.stats.record_quantize(timer)
            self.stats.quantize_calls += 1
        self.stats.bf16_kv_bytes = bf16_bytes
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
        # ``meta["start_token"]`` limits decoding to the attention window.
        start = max(int((meta or {}).get("start_token", 0)), 0)
        with timed(state_device(state), enabled=self.stats.timing_enabled) as timer:
            k_parts = []
            v_parts = []
            cursor = 0
            for segment in state["segments"]:
                length = int(segment["length"])
                if cursor + length > start:
                    skip = max(start - cursor, 0)
                    k_parts.append(self._dequantize_rotated(segment["k"])[:, skip:])
                    v_parts.append(self._dequantize_rotated(segment["v"])[:, skip:])
                cursor += length
            for name in ("residual", "write"):
                buffered_k = state.get(f"{name}_k")
                buffered_v = state.get(f"{name}_v")
                if isinstance(buffered_k, torch.Tensor) and buffered_k.shape[1] > 0:
                    length = int(buffered_k.shape[1])
                    if cursor + length > start:
                        skip = max(start - cursor, 0)
                        k_parts.append(buffered_k[:, skip:])
                        v_parts.append(buffered_v[:, skip:])
                    cursor += length
            if len(k_parts) == 1:
                k, v = k_parts[0], v_parts[0]
            elif k_parts:
                k, v = torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)
            else:
                shape = tuple(int(dim) for dim in (meta or {}).get("shape", state.get("shape", (0, 0, 0, 0))))
                dtype = (meta or {}).get("tensor_dtype") or state.get("tensor_dtype") or torch.bfloat16
                device = (meta or {}).get("device", "cpu")
                k = torch.empty((shape[0], 0, shape[2], shape[3]), dtype=dtype, device=device)
                v = k.clone()
            if not state.get("attention_space", False):
                k = self._inv_rotate(k)
                if self.rotate_value:
                    v = self._inv_rotate(v)
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

    # -- one-shot cache ----------------------------------------------------

    def quantize_kv(self, k, v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if k.ndim != 4 or v.shape != k.shape:
            raise ValueError(
                f"Expected K/V with matching [B, L, H, D] shapes, got {tuple(k.shape)} and {tuple(v.shape)}"
            )
        meta = dict(meta or {})
        already_rotated = bool(meta.get("already_rotated", False))
        bf16_bytes = int(k.numel() * k.element_size() + v.numel() * v.element_size())
        with timed(k.device, enabled=self.stats.timing_enabled) as timer:
            # Rotation runs inside the chunked quantization loop, so the full
            # rotated float32 cache never exists at once.
            k_state = self._quantize_rotated(
                k,
                self.key_bits,
                rotate=None if already_rotated else self._rotate,
            )
            v_state = self._quantize_rotated(
                v,
                self.value_bits,
                is_value=True,
                rotate=None if already_rotated else self.prepare_value,
            )
            tensor_dtype = meta.get("tensor_dtype", k.dtype)
            k_state["tensor_dtype"] = tensor_dtype
            v_state["tensor_dtype"] = tensor_dtype
            state = {
                "k": k_state,
                "v": v_state,
                # A caller may ask the one-shot cache to stay rotated without
                # having rotated the input itself, which is what LongCat's
                # condition cache does.  This mirrors ``append_kv``.
                "attention_space": bool(meta.get("attention_space", already_rotated)),
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
            k = self._dequantize_rotated(state["k"])
            v = self._dequantize_rotated(state["v"])
            if not state.get("attention_space", False):
                k = self._inv_rotate(k)
                if self.rotate_value:
                    v = self._inv_rotate(v)
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
        group = self._resolve_group(head_dim, num_heads)
        values = batch_size * active_tokens * num_heads * head_dim
        groups = batch_size * active_tokens * (num_heads * head_dim) // group
        # A BF16 scale, plus an 8-bit zero point when asymmetric.
        param_bytes = groups * (
            SCALE_STORAGE_BYTES + (ZERO_STORAGE_BYTES if self.asym else 0)
        )
        key_bytes = packed_bytes(values, self.key_bits) + param_bytes
        value_bytes = packed_bytes(values, self.value_bits) + param_bytes
        return int(key_bytes + value_bytes)


class HadamardKQuantizer(QuaRotKVQuantizer):
    """QuaRot with the V-side rotation removed.

    Kept as the K-only ablation of :class:`QuaRotKVQuantizer`: it isolates how
    much of QuaRot's benefit comes from the shared post-RoPE Q/K rotation
    alone.  It is not QuaRot and must not be reported as such.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["rotate_value"] = False
        super().__init__(*args, **kwargs)
