"""KV-cache-only QuaRot, loaded from the repository-level implementation."""

from .shared import shared_quantizer_class


QuaRotKVQuantizer = shared_quantizer_class("QUAROT_KV")

__all__ = ["QuaRotKVQuantizer"]
