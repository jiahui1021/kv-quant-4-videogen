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
    """Convert LongCat ``[B, H, S, D]`` to shared ``[B, S, H, D]``."""
    if x.ndim != 4:
        raise ValueError(
            f"Expected LongCat KV with shape [B, H, S, D], got {tuple(x.shape)}"
        )
    return x.permute(0, 2, 1, 3).contiguous()


def shared_to_longcat(x: torch.Tensor) -> torch.Tensor:
    """Convert shared ``[B, S, H, D]`` to LongCat ``[B, H, S, D]``."""
    if x.ndim != 4:
        raise ValueError(
            f"Expected shared KV with shape [B, S, H, D], got {tuple(x.shape)}"
        )
    return x.permute(0, 2, 1, 3).contiguous()


def encode_longcat_kv(k: torch.Tensor, v: torch.Tensor, quantizer) -> dict[str, Any]:
    if k.shape != v.shape:
        raise ValueError(f"K/V shapes must match, got {tuple(k.shape)} and {tuple(v.shape)}")
    dtype = k.dtype
    state = quantizer.quantize_kv(
        longcat_to_shared(k),
        longcat_to_shared(v),
        meta={"tensor_dtype": dtype},
    )
    return {
        "format": FORMAT_NAME,
        "state": state,
        "dtype": dtype,
        "shape": tuple(k.shape),
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


def move_state_to(obj: Any, device: torch.device | str) -> Any:
    """Recursively move tensors in a quantized payload without touching metadata."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: move_state_to(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [move_state_to(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(move_state_to(value, device) for value in obj)
    return obj
