"""Block-boundary resident-byte sampling for the Self-Forcing KV quantizers.

This repository's integration lives in the attention layer: the patch in
``docs/patches/self_forcing_kv_quant.patch`` dequantizes on read and quantizes
on write inside ``CausalWanSelfAttention``.  The pipeline's block loop is in
unpatched upstream code, so there is no block hook available without extending
that patch.

The wrapper below reconstructs the boundary instead: ``quantize_kv`` is called
once per layer per block, so a wrap of the layer counter marks a new block.
The sample is taken before the block's first layer is quantized and again
after, which is the same dense/packed state the other two repositories sample.
The equality gate in ``collect_efficiency.py`` -- every method's
``peak_bf16_equivalent_bytes`` must equal the BF16 run's
``peak_resident_bytes`` -- is what proves the reconstruction is correct.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple

from . import efficiency_record as efficiency
from .base import KVQuantizer


class SamplingQuantizer(KVQuantizer):
    """Transparent quantizer wrapper that samples the cache at block edges."""

    def __init__(
        self,
        inner: KVQuantizer,
        cache_getter: Callable[[], Sequence[Dict[str, Any]]],
        num_layers: int,
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        # KVQuantizer.__init__ is deliberately not called: this class owns no
        # quantization state of its own and forwards every setting to `inner`.
        self._inner = inner
        self._cache_getter = cache_getter
        self._num_layers = int(num_layers)
        self._layer_cursor = 0
        self.sampler = efficiency.ResidentSampler()

    # -- pass-through -----------------------------------------------------

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

    def estimate_active_kv_bytes(
        self, active_tokens, batch_size, num_heads, head_dim
    ) -> int:
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
        # Optional protocol members (finalize_state, reset_prompt_state, ...)
        # reach the wrapped quantizer untouched.  Only consulted for attributes
        # this class does not define, so it cannot shadow the methods above.
        return getattr(self._inner, name)

    # -- sampling ---------------------------------------------------------

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
        """Resident bytes and BF16 equivalent, per ``resident_analytic.v1``.

        Counts the packed ``quant_state``, the BF16 recent window some methods
        keep unquantized, and any dense ``k``/``v`` buffer still allocated.
        All are resident at the same time, so a figure built from only one of
        them understates the cache.
        """
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
        batch, heads, head_dim = _cache_geometry(layers[0])
        equivalent = efficiency.bf16_equivalent_bytes(
            batch=batch, tokens=tokens, heads=heads, head_dim=head_dim
        ) * len(layers)
        return resident, equivalent


def _cache_geometry(block: Dict[str, Any]) -> Tuple[int, int, int]:
    """Batch, heads, and head dimension, from whichever tensor the block has."""
    for key in ("k", "recent_k"):
        tensor = block.get(key)
        if tensor is not None and getattr(tensor, "ndim", 0) == 4:
            batch, _, heads, head_dim = tensor.shape
            return int(batch), int(heads), int(head_dim)
    return (
        int(block.get("batch_size", 0)),
        int(block.get("num_heads", 0)),
        int(block.get("head_dim", 0)),
    )


__all__ = ["SamplingQuantizer"]
