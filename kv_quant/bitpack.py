"""Physical bit packing for the integer KV-cache payloads.

Packing and unpacking run on ``uint8``/``int8`` in bounded chunks.  Widening
to ``int32`` over the whole cache made the transient several times the size
of the BF16 cache it replaced, which is what ran long generations out of
memory.  The byte layout is unchanged: value ``j`` of a byte sits at bit
``j * bits``.
"""

from __future__ import annotations

from typing import Sequence

import torch

# Values processed per chunk; bounds the transient independently of cache size.
_CHUNK_VALUES = 1 << 24


def _validate_bits(bits: int) -> None:
    if bits not in (2, 4):
        raise ValueError(f"Only INT2 and INT4 packing is supported, got bits={bits}.")


def _codes(q: torch.Tensor, bits: int, signed: bool) -> torch.Tensor:
    """Map int8 quantization values to unsigned ``bits``-wide codes."""
    offset = (1 << (bits - 1)) if signed else 0
    mask = (1 << bits) - 1
    return (q.to(torch.int16) + offset).clamp_(0, mask).to(torch.uint8)


def _pack_codes_into(out: torch.Tensor, codes: torch.Tensor, bits: int) -> None:
    """OR ``codes[..., j]`` into ``out`` at bit ``j * bits``."""
    for j in range(codes.shape[-1]):
        out |= codes[..., j] << (j * bits)


def _unpack_codes(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """``[..., bytes]`` uint8 -> ``[..., bytes * values_per_byte]`` uint8 codes."""
    values_per_byte = 8 // bits
    mask = (1 << bits) - 1
    codes = torch.empty(
        (*packed.shape, values_per_byte), dtype=torch.uint8, device=packed.device
    )
    for j in range(values_per_byte):
        torch.bitwise_and(packed >> (j * bits), mask, out=codes[..., j])
    return codes.reshape(*packed.shape[:-1], -1)


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
    flat = q.reshape(-1)
    numel = flat.numel()
    packed = torch.zeros(
        (numel + values_per_byte - 1) // values_per_byte,
        dtype=torch.uint8,
        device=q.device,
    )
    chunk = (_CHUNK_VALUES // values_per_byte) * values_per_byte
    for start in range(0, numel, chunk):
        end = min(start + chunk, numel)
        codes = _codes(flat[start:end], bits, signed)
        pad = (-codes.numel()) % values_per_byte
        if pad:
            codes = torch.cat([codes, codes.new_zeros(pad)])
        out = packed[start // values_per_byte:(end + values_per_byte - 1) // values_per_byte]
        _pack_codes_into(out, codes.view(-1, values_per_byte), bits)
    return packed


def pack_bits_rows(
    q: torch.Tensor,
    bits: int,
    *,
    signed: bool,
    row_dims: int,
) -> torch.Tensor:
    """Pack the trailing dimensions independently for every leading row.

    KIVI has an intrinsic storage boundary for every key sequence group and
    every value token.  Padding each row independently makes the physical
    byte count independent of append boundaries and lets eviction slice whole
    rows without unpacking or re-packing older payloads.
    """
    _validate_bits(bits)
    if q.dtype != torch.int8:
        raise TypeError(f"Expected an int8 quantization tensor, got dtype={q.dtype}.")
    if row_dims <= 0 or row_dims >= q.ndim:
        raise ValueError(
            f"row_dims must keep between 1 and {q.ndim - 1} leading dimensions, "
            f"got {row_dims}."
        )

    leading_shape = tuple(int(dim) for dim in q.shape[:row_dims])
    values_per_row = 1
    for dim in q.shape[row_dims:]:
        values_per_row *= int(dim)
    rows = q.reshape(-1, values_per_row)

    values_per_byte = 8 // bits
    row_pad = (-values_per_row) % values_per_byte
    bytes_per_row = (values_per_row + row_pad) // values_per_byte
    packed = torch.zeros(
        (rows.shape[0], bytes_per_row), dtype=torch.uint8, device=q.device
    )
    chunk_rows = max(_CHUNK_VALUES // max(values_per_row, 1), 1)
    for start in range(0, rows.shape[0], chunk_rows):
        end = min(start + chunk_rows, rows.shape[0])
        codes = _codes(rows[start:end], bits, signed)
        if row_pad:
            codes = torch.cat([codes, codes.new_zeros(end - start, row_pad)], dim=1)
        _pack_codes_into(
            packed[start:end],
            codes.view(end - start, bytes_per_row, values_per_byte),
            bits,
        )
    return packed.reshape(*leading_shape, bytes_per_row)


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

    offset = (1 << (bits - 1)) if signed else 0
    flat = packed.reshape(-1)
    out = torch.empty(numel, dtype=torch.int8, device=packed.device)
    chunk_bytes = max(_CHUNK_VALUES // values_per_byte, 1)
    for byte_start in range(0, required_bytes, chunk_bytes):
        byte_end = min(byte_start + chunk_bytes, required_bytes)
        start = byte_start * values_per_byte
        end = min(byte_end * values_per_byte, numel)
        codes = _unpack_codes(flat[byte_start:byte_end], bits)[: end - start]
        target = out[start:end]
        target.copy_(codes)
        if offset:
            target.sub_(offset)
    return out.reshape(shape)


def unpack_bits_rows(
    packed: torch.Tensor,
    bits: int,
    shape: Sequence[int],
    *,
    signed: bool,
    row_dims: int,
) -> torch.Tensor:
    """Inverse of :func:`pack_bits_rows`."""
    _validate_bits(bits)
    if packed.dtype != torch.uint8:
        raise TypeError(f"Expected a uint8 packed tensor, got dtype={packed.dtype}.")
    shape = tuple(int(dim) for dim in shape)
    if row_dims <= 0 or row_dims >= len(shape):
        raise ValueError(
            f"row_dims must keep between 1 and {len(shape) - 1} leading dimensions, "
            f"got {row_dims}."
        )

    rows = 1
    for dim in shape[:row_dims]:
        rows *= dim
    values_per_row = 1
    for dim in shape[row_dims:]:
        values_per_row *= dim
    values_per_byte = 8 // bits
    required_row_bytes = (values_per_row + values_per_byte - 1) // values_per_byte
    if packed.numel() != rows * required_row_bytes:
        raise ValueError(
            f"Packed row tensor has {packed.numel()} bytes, but "
            f"{rows * required_row_bytes} are required for shape={shape}."
        )

    offset = (1 << (bits - 1)) if signed else 0
    packed_rows = packed.reshape(rows, required_row_bytes)
    out = torch.empty((rows, values_per_row), dtype=torch.int8, device=packed.device)
    chunk_rows = max(_CHUNK_VALUES // max(values_per_row, 1), 1)
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        codes = _unpack_codes(packed_rows[start:end], bits)[:, :values_per_row]
        target = out[start:end]
        target.copy_(codes)
        if offset:
            target.sub_(offset)
    return out.reshape(shape)
