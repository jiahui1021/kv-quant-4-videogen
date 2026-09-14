"""Bridge LongCat's ``[B, H, S, D]`` cache layout to ``kv_quant``.

The shared quantizers deliberately use ``[B, S, H, D]`` so that their
sequence blocking is identical across models.  LongCat keeps the head axis
before the sequence axis, therefore this conversion must happen at the
adapter boundary and nowhere inside the quantizer implementations.
"""

from __future__ import annotations

from typing import Any

import torch


FORMAT_NAME = "shared_kv_quant_v1"


def longcat_to_shared(x: torch.Tensor) -> torch.Tensor:
    """Convert LongCat ``[B, H, S, D]`` to shared ``[B, S, H, D]``.

    Returns a permuted view.  A contiguous copy of the whole condition cache
    would be held for the full quantization call; the quantizers copy only
    what they reshape.
    """
    if x.ndim != 4:
        raise ValueError(
            f"Expected LongCat KV with shape [B, H, S, D], got {tuple(x.shape)}"
        )
    return x.permute(0, 2, 1, 3)


def shared_to_longcat(x: torch.Tensor) -> torch.Tensor:
    """Convert shared ``[B, S, H, D]`` to LongCat ``[B, H, S, D]``.

    Returns a permuted view; the attention concatenates it with the new
    tokens anyway, so a contiguous copy would only add a transient.
    """
    if x.ndim != 4:
        raise ValueError(
            f"Expected shared KV with shape [B, S, H, D], got {tuple(x.shape)}"
        )
    return x.permute(0, 2, 1, 3)


def encode_longcat_kv(k: torch.Tensor, v: torch.Tensor, quantizer) -> dict[str, Any]:
    if k.shape != v.shape:
        raise ValueError(f"K/V shapes must match, got {tuple(k.shape)} and {tuple(v.shape)}")
    dtype = k.dtype
    # QuaRot keeps the cache in the rotated attention space; the attention
    # module rotates Q to match and undoes the V rotation on its output.
    attention_space = bool(
        getattr(quantizer, "requires_special_attention_backend", False)
    )
    # The caller captures K after RoPE for QuaRot and for methods that declare
    # ``cache_space = "post_rope"`` (KIVI); the payload records which it is.
    cache_space = (
        "post_rope"
        if attention_space
        else getattr(quantizer, "cache_space", "pre_rope")
    )
    state = quantizer.quantize_kv(
        longcat_to_shared(k),
        longcat_to_shared(v),
        meta={
            "tensor_dtype": dtype,
            "device": k.device,
            "attention_space": attention_space,
        },
    )
    return {
        "format": FORMAT_NAME,
        "state": state,
        "dtype": dtype,
        "shape": tuple(k.shape),
        "attention_space": attention_space,
        "cache_space": cache_space,
    }


def decode_longcat_kv(payload: dict[str, Any], quantizer) -> tuple[torch.Tensor, torch.Tensor]:
    if not is_shared_quant_cache(payload):
        raise ValueError("Payload is not a shared LongCat KV cache")
    k_shared, v_shared = quantizer.dequantize_kv(
        payload["state"],
        meta={"tensor_dtype": payload["dtype"]},
    )
    k = shared_to_longcat(k_shared)
    v = shared_to_longcat(v_shared)
    expected_shape = tuple(payload.get("shape", k.shape))
    if tuple(k.shape) != expected_shape or tuple(v.shape) != expected_shape:
        raise RuntimeError(
            f"Decoded LongCat KV shape mismatch: expected {expected_shape}, "
            f"got {tuple(k.shape)} and {tuple(v.shape)}"
        )
    return k, v


def is_shared_quant_cache(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("format") == FORMAT_NAME and "state" in obj


def move_state_to(obj: Any, device: torch.device | str, _memo: dict | None = None) -> Any:
    """Recursively move tensors in a quantized payload without touching metadata.

    Quantizer states alias sub-objects (KIVI keeps ``segments``/``k``/``v``
    as aliases of ``k_segments``/``v_segments``), so every object is moved
    once and its aliases share the result; otherwise a cross-device move
    would store each packed payload several times.
    """
    memo = {} if _memo is None else _memo
    key = id(obj)
    if key in memo:
        return memo[key]
    if isinstance(obj, torch.Tensor):
        moved = obj.to(device)
    elif isinstance(obj, dict):
        moved = {}
        memo[key] = moved
        for name, value in obj.items():
            moved[name] = move_state_to(value, device, memo)
        return moved
    elif isinstance(obj, list):
        moved = []
        memo[key] = moved
        moved.extend(move_state_to(value, device, memo) for value in obj)
        return moved
    elif isinstance(obj, tuple):
        moved = tuple(move_state_to(value, device, memo) for value in obj)
    else:
        return obj
    memo[key] = moved
    return moved
