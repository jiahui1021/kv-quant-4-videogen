"""Compatibility import for the repository-level RTN quantizer."""

from .shared import shared_quantizer_class


RTNQuantizer = shared_quantizer_class("RTN")

__all__ = ["RTNQuantizer"]
