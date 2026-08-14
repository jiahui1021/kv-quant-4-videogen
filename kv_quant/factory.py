from __future__ import annotations

import re


SUPPORTED_METHODS = (
    "BF16",
    "RTN_INT4",
    "RTN_INT2",
    "KIVI_INT4",
    "KIVI_INT2",
    "QUAROT_KV_INT4",
    "QUAROT_KV_INT2",
    "QVG_INT2",
    "QVG_INT4",
)


def create_quantizer(
    method: str,
    bits: int,
    block_size: int = 16,
    key_bits: int | None = None,
    value_bits: int | None = None,
    name: str | None = None,
    residual_length: int | None = None,
    value_group_size: int | None = None,
    channel_group_size: int | None = None,
):
    """Create one of the shared KV-cache quantizers."""
    method = method.upper()
    if method == "RTN":
        from .rtn import RTNQuantizer

        return RTNQuantizer(
            bits=bits,
            block_size=block_size,
            key_bits=key_bits,
            value_bits=value_bits,
            name=name,
        )
    if method == "KIVI":
        from .kivi import KIVIQuantizer

        return KIVIQuantizer(
            bits=bits,
            block_size=block_size,
            key_bits=key_bits,
            value_bits=value_bits,
            name=name,
            residual_length=residual_length,
            value_group_size=value_group_size,
        )
    if method == "QUAROT_KV":
        from .quarot_kv import QuaRotKVQuantizer

        return QuaRotKVQuantizer(
            bits=bits,
            block_size=block_size,
            key_bits=key_bits,
            value_bits=value_bits,
            name=name,
            channel_group_size=channel_group_size,
        )
    raise ValueError(f"Unsupported KV quantization method: {method}")


def parse_method(method: str, block_size: int = 16):
    """Return ``(canonical_name, quantizer_or_none)`` for a CLI method."""
    method = method.upper()
    if method == "BF16":
        return "BF16", None

    match = re.fullmatch(r"(RTN|KIVI|QUAROT_KV)_INT(2|4)", method)
    if match is None:
        supported = ", ".join(SUPPORTED_METHODS)
        raise ValueError(
            f"Unsupported method={method}. Expected one of: {supported}"
        )

    bits = int(match.group(2))
    return method, create_quantizer(match.group(1), bits, block_size=block_size)
