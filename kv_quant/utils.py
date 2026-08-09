from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Iterator, Tuple

import torch

EPS = 1e-8


def _reshape_blocks(x: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, int]:
    # x shape: [B, L, H, D]
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor [B, L, H, D], got shape={tuple(x.shape)}")
    b, l, h, d = x.shape
    pad_len = (block_size - (l % block_size)) % block_size
    if pad_len:
        pad = torch.zeros((b, pad_len, h, d), device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)
    nb = x.shape[1] // block_size
    return x.view(b, nb, block_size, h, d), pad_len


def _unshape_blocks(xb: torch.Tensor, pad_len: int, orig_len: int) -> torch.Tensor:
    x = xb.reshape(xb.shape[0], xb.shape[1] * xb.shape[2], xb.shape[3], xb.shape[4])
    if pad_len:
        x = x[:, :orig_len]
    return x


def quantize_asym(x: torch.Tensor, bits: int, reduce_dims: Tuple[int, ...]):
    qmin, qmax = 0, (1 << bits) - 1
    x_min = x.amin(dim=reduce_dims, keepdim=True)
    x_max = x.amax(dim=reduce_dims, keepdim=True)
    scale = ((x_max - x_min) / max(qmax - qmin, 1)).clamp_min(EPS)
    zp = torch.round(qmin - x_min / scale).clamp(qmin, qmax)
    q = torch.round(x / scale + zp).clamp(qmin, qmax).to(torch.int8)
    return q, scale.to(torch.float16), zp.to(torch.float16)


def dequantize_asym(
    q: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return (q.to(dtype) - zp.to(dtype)) * scale.to(dtype)


def quantize_sym(x: torch.Tensor, bits: int, reduce_dims: Tuple[int, ...]):
    qmax = (1 << (bits - 1)) - 1
    x_abs = x.abs().amax(dim=reduce_dims, keepdim=True)
    scale = (x_abs / max(qmax, 1)).clamp_min(EPS)
    q = torch.round(x / scale).clamp(-qmax - 1, qmax).to(torch.int8)
    return q, scale.to(torch.float16)


def dequantize_sym(
    q: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return q.to(dtype) * scale.to(dtype)


def fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError("Last dimension must be power of two for Hadamard transform.")
    y = x
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], -1, h * 2)
        a = y[..., :h]
        b = y[..., h:]
        y = torch.cat((a + b, a - b), dim=-1)
        y = y.reshape(*x.shape)
        h <<= 1
    return y / math.sqrt(n)


class TimingResult:
    """A timer whose CUDA result can be resolved after generation completes."""

    def __init__(
        self,
        device: torch.device | str | None = None,
        enabled: bool = True,
    ) -> None:
        self.device = torch.device(device) if device is not None else None
        self._enabled = bool(enabled)
        self._use_cuda_events = (
            self._enabled
            and self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        self._start_event = None
        self._end_event = None
        self._start_cpu = None
        self._elapsed_s: float | None = None

    @property
    def is_cuda(self) -> bool:
        return self._use_cuda_events

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if not self._enabled:
            return
        if self._use_cuda_events:
            with torch.cuda.device(self.device):
                self._start_event = torch.cuda.Event(enable_timing=True)
                self._end_event = torch.cuda.Event(enable_timing=True)
                self._start_event.record()
        else:
            self._start_cpu = time.perf_counter()

    def stop(self) -> None:
        if not self._enabled:
            return
        if self._use_cuda_events:
            with torch.cuda.device(self.device):
                self._end_event.record()
        else:
            self._elapsed_s = time.perf_counter() - self._start_cpu

    def resolve(self, synchronize: bool = False) -> float:
        if not self._enabled:
            return 0.0
        if self._elapsed_s is not None:
            return self._elapsed_s
        if not self._use_cuda_events:
            raise RuntimeError("Timer was stopped without a CPU timestamp.")
        if synchronize:
            self._end_event.synchronize()
        self._elapsed_s = self._start_event.elapsed_time(self._end_event) / 1000.0
        return self._elapsed_s


@contextmanager
def timed(
    device: torch.device | str | None = None,
    enabled: bool = True,
) -> Iterator[TimingResult]:
    """Measure work without synchronizing CUDA on every call.

    CUDA event results are intentionally left unresolved. The caller should
    resolve them after one generation-level ``torch.cuda.synchronize()``.
    """
    timer = TimingResult(device, enabled=enabled)
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()
