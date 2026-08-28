"""Compatibility import for the repository-level Hadamard-K quantizer."""

from .shared import shared_quantizer_class


HadamardKQuantizer = shared_quantizer_class("HADAMARD_K")

__all__ = ["HadamardKQuantizer"]
