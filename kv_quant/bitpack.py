"""Physical bit packing for the integer KV-cache payloads."""

from __future__ import annotations

from typing import Sequence

import torch


def _validate_bits(bits: int) -> None:
    if bits not in (2, 4):
        raise ValueError(f"Only INT2 and INT4 packing is supported, got bits={bits}.")


def pack_bits(q: torch.Tensor, bits: int, *, signed: bool) -> torch.Tensor:
    """Pack an INT2/INT4 tensor into one byte per 8/bits values.

    ``signed=True`` stores the signed quantization range using an offset
    representation.  ``signed=False`` stores the unsigned KIVI range directly.
    The returned tensor is ``torch.uint8`` and contains the actual resident
    payload, rather than an INT8 tensor with a logical bit width.
    """
    _validate_bits(bits)
    if q.dtype != torch.int8:
        raise TypeError(f"Expected an int8 quantization tensor, got dtype={q.dtype}.")

    values_per_byte = 8 // bits
    value_mask = (1 << bits) - 1
    offset = (1 << (bits - 1)) if signed else 0
    values = q.reshape(-1).to(torch.int32) + offset
    values = values.clamp(0, value_mask)

    pad = (-values.numel()) % values_per_byte
    if pad:
        values = torch.cat(
            [values, torch.zeros(pad, dtype=values.dtype, device=values.device)]
        )

    shifts = torch.arange(
        values_per_byte, dtype=torch.int32, device=values.device
    ) * bits
    packed = (values.reshape(-1, values_per_byte) << shifts).sum(dim=1)
    return packed.to(torch.uint8)


def unpack_bits(
    packed: torch.Tensor,
    bits: int,
    shape: Sequence[int],
    numel: int,
    *,
    signed: bool,
) -> torch.Tensor:
    """Restore an INT2/INT4 tensor from a physically packed byte tensor."""
    _validate_bits(bits)
    if packed.dtype != torch.uint8:
        raise TypeError(f"Expected a uint8 packed tensor, got dtype={packed.dtype}.")
    if numel < 0:
        raise ValueError(f"numel must be non-negative, got {numel}.")

    shape = tuple(int(dim) for dim in shape)
    expected_numel = 1
    for dim in shape:
        expected_numel *= dim
    if expected_numel != numel:
        raise ValueError(
            f"Packed state shape={shape} contains {expected_numel} values, "
            f"but numel={numel}."
        )

    values_per_byte = 8 // bits
    required_bytes = (numel + values_per_byte - 1) // values_per_byte
    if packed.numel() < required_bytes:
        raise ValueError(
            f"Packed tensor has {packed.numel()} bytes, "
            f"but {required_bytes} are required for {numel} values."
        )

    value_mask = (1 << bits) - 1
    offset = (1 << (bits - 1)) if signed else 0
    shifts = torch.arange(
        values_per_byte, dtype=torch.int32, device=packed.device
    ) * bits
    values = (packed.reshape(-1).to(torch.int32)[:, None] >> shifts) & value_mask
    values = values.reshape(-1)[:numel] - offset
    return values.to(torch.int8).reshape(shape)
