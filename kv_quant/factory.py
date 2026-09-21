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
    "HADAMARD_K_INT4",
    "HADAMARD_K_INT2",
    "QVG_INT2",
    "QVG_INT4",
)


def create_quantizer(
    method: str,
    bits: int,
    block_size: int | None = None,
    key_bits: int | None = None,
    value_bits: int | None = None,
    name: str | None = None,
    residual_length: int | None = None,
    key_group_size: int | None = None,
    value_group_size: int | None = None,
    channel_group_size: int | None = None,
    asym: bool | None = None,
    clip_ratio: float | None = None,
):
    """Create one of the shared KV-cache quantizers.

    ``block_size=None`` leaves every baseline at its own reference setting
    (RTN channel groups of 128, KIVI group 32 with a 128-token BF16 residual,
    QuaRot one ``head_dim`` group).  Passing a number overrides all of them at
    once, which is a matched-granularity ablation rather than the paper rows.
    """
    method = method.upper()
    paper_block_size = block_size is None
    # The exploratory methods below still take a number; only the three paper
    # baselines carry a published grouping of their own.
    block_size = 16 if block_size is None else int(block_size)
    # Each method exposes a different set of grouping knobs.  Silently
    # dropping one produces a quantizer that is not the configuration the
    # caller asked for, so an inapplicable option is refused here instead.
    _ignored = {
        "RTN": {"residual_length": residual_length,
                "key_group_size": key_group_size,
                "value_group_size": value_group_size,
                "asym": asym, "clip_ratio": clip_ratio},
        "KIVI": {"channel_group_size": channel_group_size,
                 "asym": asym, "clip_ratio": clip_ratio},
        "RTN_EXTRA": {},
        "HADAMARD_K": {"residual_length": residual_length,
                       "key_group_size": key_group_size,
                       "value_group_size": value_group_size},
        "QUAROT_KV": {"residual_length": residual_length,
                      "key_group_size": key_group_size,
                      "value_group_size": value_group_size},
    }.get(method, {})
    unused = sorted(name for name, value in _ignored.items() if value is not None)
    if unused:
        raise ValueError(
            f"{method} does not support {', '.join(unused)}; "
            "passing it would silently change nothing"
        )
    if method == "RTN":
        from .rtn import RTNQuantizer

        return RTNQuantizer(
            bits=bits,
            **({} if paper_block_size else {"block_size": block_size}),
            key_bits=key_bits,
            value_bits=value_bits,
            name=name,
            **({} if channel_group_size is None else {"channel_group_size": int(channel_group_size)}),
        )
    if method == "KIVI":
        from .kivi import KIVIQuantizer

        return KIVIQuantizer(
            bits=bits,
            **({} if paper_block_size else {"block_size": block_size}),
            key_bits=key_bits,
            value_bits=value_bits,
            name=name,
            residual_length=residual_length,
            key_group_size=key_group_size,
            value_group_size=value_group_size,
        )
    if method in ("QUAROT_KV", "HADAMARD_K"):
        from .quarot_kv import HadamardKQuantizer, QuaRotKVQuantizer

        cls = QuaRotKVQuantizer if method == "QUAROT_KV" else HadamardKQuantizer
        extra = {}
        if asym is not None:
            extra["asym"] = bool(asym)
        if clip_ratio is not None:
            extra["clip_ratio"] = float(clip_ratio)
        if channel_group_size is not None:
            extra["channel_group_size"] = int(channel_group_size)
        return cls(
            bits=bits,
            **({} if paper_block_size else {"block_size": block_size}),
            key_bits=key_bits,
            value_bits=value_bits,
            name=name,
            **extra,
        )
    raise ValueError(f"Unsupported KV quantization method: {method}")


def parse_method(
    method: str,
    block_size: int | None = None,
    channel_group_size: int | None = None,
    asym: bool | None = None,
    clip_ratio: float | None = None,
):
    """Return ``(canonical_name, quantizer_or_none)`` for a CLI method.

    ``asym`` and ``clip_ratio`` are QuaRot's ``--k_asym``/``--k_clip_ratio``
    knobs and only apply to the rotated backends; passing them to RTN or KIVI
    is refused by :func:`create_quantizer` rather than ignored.

    ``block_size=None`` keeps every baseline at its published grouping.
    """
    method = method.upper()
    if method == "BF16":
        return "BF16", None

    match = re.fullmatch(r"(RTN|KIVI|QUAROT_KV|HADAMARD_K)_INT(2|4)", method)
    if match is None:
        supported = ", ".join(
            name for name in SUPPORTED_METHODS if not name.startswith("QVG_")
        )
        raise ValueError(
            f"Unsupported method={method}. Expected one of: {supported} "
            "(QVG has its own entry point in the Causal-Forcing adapter)"
        )

    bits = int(match.group(2))
    return method, create_quantizer(
        match.group(1),
        bits,
        block_size=block_size,
        channel_group_size=channel_group_size,
        asym=asym,
        clip_ratio=clip_ratio,
    )
