"""Compatibility import for the repository-level KIVI quantizer."""

from .shared import shared_quantizer_class


KIVIQuantizer = shared_quantizer_class("KIVI")

__all__ = ["KIVIQuantizer"]
