"""KV-cache-only QuaRot, loaded from the repository-level implementation."""

from .shared import shared_quantizer_class


QuaRotKVQuantizer = shared_quantizer_class("QUAROT_KV")


def CompositeQuaRotKVQuantizer(**kwargs):
    """QuaRot as an inner quantizer of the composite methods (AGE_TIER, ...).

    Composites re-quantize the whole dequantized cache prefix on every call.
    The paper's clipping ratio (0.95) then shrinks the history range again on
    each call, so composites keep the unclipped symmetric range they were
    built with.  The standalone ``QUAROT_KV`` baseline keeps the paper setting
    and never re-quantizes history.
    """
    kwargs.setdefault("asym", False)
    kwargs.setdefault("clip_ratio", 1.0)
    return QuaRotKVQuantizer(**kwargs)


__all__ = ["CompositeQuaRotKVQuantizer", "QuaRotKVQuantizer"]
