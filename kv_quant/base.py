from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch


@dataclass
class QuantizerStats:
    quantize_time_s: float = 0.0
    dequantize_time_s: float = 0.0
    quantize_calls: int = 0
    dequantize_calls: int = 0
    bf16_kv_bytes: int = 0
    compressed_kv_bytes: int = 0
    timing_enabled: bool = False
    _pending_quantize_timers: list[Any] = field(default_factory=list, repr=False)
    _pending_dequantize_timers: list[Any] = field(default_factory=list, repr=False)

    def record_quantize(self, timer: Any) -> None:
        if not timer.enabled:
            return
        if timer.is_cuda:
            self._pending_quantize_timers.append(timer)
        else:
            self.quantize_time_s += timer.resolve()

    def record_dequantize(self, timer: Any) -> None:
        if not timer.enabled:
            return
        if timer.is_cuda:
            self._pending_dequantize_timers.append(timer)
        else:
            self.dequantize_time_s += timer.resolve()

    def resolve_timing(self, synchronize: bool = True) -> None:
        """Resolve pending CUDA events, synchronizing at most once per device."""
        pending = self._pending_quantize_timers + self._pending_dequantize_timers
        if not pending:
            return

        if synchronize:
            devices = {timer.device for timer in pending if timer.device is not None}
            for device in devices:
                torch.cuda.synchronize(device)

        for timer in self._pending_quantize_timers:
            self.quantize_time_s += timer.resolve()
        for timer in self._pending_dequantize_timers:
            self.dequantize_time_s += timer.resolve()
        self._pending_quantize_timers.clear()
        self._pending_dequantize_timers.clear()


class KVQuantizer(ABC):
    def __init__(
        self,
        bits: int = 4,
        block_size: int = 16,
        name: str = "BASE",
        key_bits: int | None = None,
        value_bits: int | None = None,
    ) -> None:
        key_bits = bits if key_bits is None else key_bits
        value_bits = bits if value_bits is None else value_bits
        if key_bits not in (2, 4):
            raise ValueError(f"Unsupported key_bits={key_bits}; expected one of [2, 4].")
        if value_bits not in (2, 4):
            raise ValueError(f"Unsupported value_bits={value_bits}; expected one of [2, 4].")
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        self.bits = bits
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.block_size = block_size
        self._name = name
        self.stats = QuantizerStats()

    def name(self) -> str:
        return self._name

    def reset_stats(self) -> None:
        self.stats = QuantizerStats(timing_enabled=self.stats.timing_enabled)

    def set_timing_enabled(self, enabled: bool) -> None:
        """Enable optional quantize/dequantize timing breakdowns."""
        self.stats.timing_enabled = bool(enabled)

    @abstractmethod
    def quantize_kv(self, k, v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    # The one-shot methods above remain part of the public adapter contract
    # (LongCat uses them).  These runtime methods are implemented by the
    # shared RTN/KIVI/QuaRot baselines and are intentionally separate so
    # legacy experimental quantizers can keep their existing behavior.
    def init_state(self, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    def append_kv(
        self,
        state: Dict[str, Any],
        new_k,
        new_v,
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def materialize_kv(
        self,
        state: Dict[str, Any],
        meta: Dict[str, Any] | None = None,
    ) -> Tuple[Any, Any]:
        raise NotImplementedError

    def finalize_state(
        self,
        state: Dict[str, Any],
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def evict_prefix(self, state: Dict[str, Any], requested_tokens: int) -> int:
        raise NotImplementedError

    def evict_range(self, state: Dict[str, Any], start_tokens: int, requested_tokens: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def dequantize_kv(self, state: Dict[str, Any], meta: Dict[str, Any] | None = None) -> Tuple[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def memory_bytes(self, state: Dict[str, Any]) -> int:
        raise NotImplementedError

    @abstractmethod
    def estimate_active_kv_bytes(
        self,
        active_tokens: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> int:
        raise NotImplementedError
