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


def reshape_channel_groups(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Split a ``[B, L, H, D]`` tensor into per-token channel groups."""
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor [B, L, H, D], got shape={tuple(x.shape)}")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    b, l, h, d = x.shape
    if d % group_size:
        raise ValueError(
            f"head_dim={d} must be divisible by channel_group_size={group_size}"
        )
    return x.reshape(b, l, h, d // group_size, group_size)


def quantize_asym(x: torch.Tensor, bits: int, reduce_dims: Tuple[int, ...]):
    qmin, qmax = 0, (1 << bits) - 1
    x_min = x.amin(dim=reduce_dims, keepdim=True)
    x_max = x.amax(dim=reduce_dims, keepdim=True)
    scale = ((x_max - x_min) / max(qmax - qmin, 1)).clamp_min(EPS)
    zero = x_min
    q = torch.round((x - zero) / scale).clamp(qmin, qmax).to(torch.int8)
    return q, scale.to(torch.float16), zero.to(torch.float16)


def dequantize_asym(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return q.to(dtype) * scale.to(dtype) + zero.to(dtype)


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
    """Normalized Walsh-Hadamard transform over the last dimension.

    QuaRot runs its transform as ``hadamard_transform(x.float(), scale=1/sqrt(d))``
    and casts the result back (``fake_quant/rotation_utils.py``), so the
    butterfly accumulates in float32 here rather than in the cache dtype.
    """
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError("Last dimension must be power of two for Hadamard transform.")
    x_dtype = x.dtype
    y = x.float()
    work_shape = y.shape
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], -1, h * 2)
        a = y[..., :h]
        b = y[..., h:]
        y = torch.cat((a + b, a - b), dim=-1)
        y = y.reshape(*work_shape)
        h <<= 1
    return (y / math.sqrt(n)).to(x_dtype)


# --- QuaRot reference quantizer ------------------------------------------
#
# Ported from spcl/QuaRot ``fake_quant/quant_utils.py``.  QuaRot only ever
# quantizes activations per token, so ``find_params`` reduces over the last
# dimension of an already-reshaped tensor and the caller decides whether a
# row is one token or one (token, head) pair.


def quarot_minq_maxq(bits: int, sym: bool) -> Tuple[int, int]:
    if sym:
        maxq = 2 ** (bits - 1) - 1
        return -maxq - 1, maxq
    return 0, 2 ** bits - 1


def quarot_find_params(
    x: torch.Tensor,
    bits: int,
    sym: bool,
    clip_ratio: float = 1.0,
):
    """QuaRot ``ActQuantizer.find_params`` over the last dimension.

    Returns ``(scale, zero)``; ``zero`` is the rounded integer zero point for
    the asymmetric branch and all zeros for the symmetric one.
    """
    _, maxq = quarot_minq_maxq(bits, sym)
    work = x.float()
    zeros = torch.zeros_like(work[..., :1])
    xmin = torch.minimum(work.amin(dim=-1, keepdim=True), zeros) * clip_ratio
    xmax = torch.maximum(work.amax(dim=-1, keepdim=True), zeros) * clip_ratio
    if sym:
        xmax = torch.maximum(xmin.abs(), xmax)
        scale = xmax / maxq
        scale = torch.where(xmax == 0, torch.ones_like(scale), scale)
        return scale, torch.zeros_like(scale)
    degenerate = (xmin == 0) & (xmax == 0)
    xmin = torch.where(degenerate, -torch.ones_like(xmin), xmin)
    xmax = torch.where(degenerate, torch.ones_like(xmax), xmax)
    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)
    return scale, zero


def quarot_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    bits: int,
    sym: bool,
) -> torch.Tensor:
    """QuaRot ``sym_quant``/``asym_quant``; returns the integer codes."""
    minq, maxq = quarot_minq_maxq(bits, sym)
    work = torch.round(x.float() / scale)
    if not sym:
        work = work + zero
    return work.clamp(minq, maxq).to(torch.int8)


def quarot_dequantize(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    sym: bool,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """QuaRot ``sym_dequant``/``asym_dequant``."""
    values = q.to(torch.float32)
    if not sym:
        values = values - zero.to(torch.float32)
    return (values * scale.to(torch.float32)).to(dtype)


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
