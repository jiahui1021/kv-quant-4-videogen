"""Compatibility import for the repository-level QuaRot KV quantizer."""

from .shared import shared_quantizer_class


QuaRotKVQuantizer = shared_quantizer_class("QUAROT_KV")

__all__ = ["QuaRotKVQuantizer"]
