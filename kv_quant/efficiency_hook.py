"""Compatibility copy of the Self-Forcing block-boundary sampler.

The top-level ``kv_quant`` package is importable before ``Self-Forcing`` during
the repository's full test suite.  Keep the same transparent wrapper available
there while the canonical runtime copy remains under ``Self-Forcing/kv_quant``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple

from . import efficiency_record as efficiency
from .base import KVQuantizer


class SamplingQuantizer(KVQuantizer):
    """Forward one quantizer while sampling before and after each block."""

    def __init__(
        self,
        inner: KVQuantizer,
        cache_getter: Callable[[], Sequence[Dict[str, Any]]],
        num_layers: int,
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self._inner = inner
        self._cache_getter = cache_getter
        self._num_layers = int(num_layers)
        self._layer_cursor = 0
        self.sampler = efficiency.ResidentSampler()

    @property
    def stats(self):
        return self._inner.stats

    @property
    def bits(self) -> int:
        return self._inner.bits

    def name(self) -> str:
        return self._inner.name()

    def reset_stats(self) -> None:
        self._inner.reset_stats()

    def dequantize_kv(self, state, meta=None):
        return self._inner.dequantize_kv(state, meta=meta)

    def memory_bytes(self, state) -> int:
        return self._inner.memory_bytes(state)

    def estimate_active_kv_bytes(self, active_tokens, batch_size, num_heads, head_dim) -> int:
        return self._inner.estimate_active_kv_bytes(
            active_tokens=active_tokens,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )

    def finalize_state(self, state, meta=None):
        return self._inner.finalize_state(state, meta=meta)

    def reset_prompt_state(self) -> None:
        reset = getattr(self._inner, "reset_prompt_state", None)
        if callable(reset):
            reset()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def quantize_kv(self, k, v, meta=None):
        if self._layer_cursor == 0:
            self._sample()
        result = self._inner.quantize_kv(k, v, meta=meta)
        self._layer_cursor += 1
        if self._layer_cursor >= self._num_layers:
            self._layer_cursor = 0
            self._sample()
        return result

    def _sample(self) -> None:
        resident, equivalent = self.resident_kv_bytes()
        self.sampler.observe(resident, equivalent)

    def resident_kv_bytes(self) -> Tuple[int, int]:
        layers = list(self._cache_getter())
        if not layers:
            return 0, 0
        resident = 0
        tokens = 0
        for block in layers:
            for key in ("quant_state", "recent_k", "recent_v", "k", "v"):
                resident += efficiency.tensor_bytes(block.get(key))
            end_index = block.get("local_end_index")
            tokens = max(tokens, int(end_index) if end_index is not None else 0)
        first = layers[0]
        geometry = first.get("k")
        if geometry is None:
            geometry = first.get("recent_k")
        if geometry is not None and getattr(geometry, "ndim", 0) == 4:
            batch, _, heads, head_dim = geometry.shape
        else:
            batch = int(first.get("batch_size", 0))
            heads = int(first.get("num_heads", 0))
            head_dim = int(first.get("head_dim", 0))
        equivalent = efficiency.bf16_equivalent_bytes(
            int(batch), tokens, int(heads), int(head_dim)
        ) * len(layers)
        return resident, equivalent


__all__ = ["SamplingQuantizer"]
