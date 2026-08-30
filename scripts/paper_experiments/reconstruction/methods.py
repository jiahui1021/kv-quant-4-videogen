#!/usr/bin/env python3
"""Every comparison method behind one reconstruction interface.

The point of this module is that one collected trace is enough: BF16, RTN,
KIVI, QuaRot, Hadamard-K, QVG and TempoKV all take the same clean K/V tensors
and return their reconstruction, so a difference between the numbers is a
difference between the methods and not between two measurement harnesses.

A method that cannot run on this host reports *why* instead of being dropped.
QVG needs Triton, and TempoKV needs a checkout of its own repository plus a
calibration trace, so both are legitimately unavailable on a CPU-only box; a
silently missing row would read as "we did not test it".

Tensor layout
    The trace is ``[layers, records, frames, heads, dim]``.  One layer of it is
    ``[B, L, H, D]``, which is exactly what ``kv_quant`` expects, so no permute
    is needed there.  QVG wants ``[B, H, L, D]`` and gets one.  TempoKV takes
    the whole five-dimensional trace as it stands.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 1

# Where a sibling TempoKV checkout is expected when none is given.
DEFAULT_TEMPOKV_ROOT = REPO_ROOT.parent / "Tempokv"


class MethodUnavailable(RuntimeError):
    """Raised when a method cannot run here, carrying the reason."""


@dataclass
class Method:
    """One reconstruction method under comparison."""

    name: str
    label: str
    family: str
    bits: int | None
    build: Callable[["MethodContext"], Callable[[torch.Tensor, torch.Tensor], tuple]]
    needs_calibration: bool = False
    notes: str = ""


@dataclass
class MethodContext:
    """Everything a method might need beyond the trace itself."""

    stride: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    calibration: tuple[torch.Tensor, torch.Tensor] | None = None
    tempokv_root: Path | None = None
    fit_iterations: int = 2
    temporal_ridge: float = 1.0
    cross_ridge: float = 0.1


# --------------------------------------------------------------------------
# BF16 storage floor
# --------------------------------------------------------------------------


def _build_bf16(context: MethodContext):
    del context

    def run(keys: torch.Tensor, values: torch.Tensor):
        # Not a no-op: every method below also stores its reconstruction in
        # bfloat16, so this is the error floor none of them can beat.
        return keys.to(torch.bfloat16).float(), values.to(torch.bfloat16).float()

    return run


# --------------------------------------------------------------------------
# kv_quant baselines
# --------------------------------------------------------------------------


def _build_kv_quant(method_name: str) -> Callable[[MethodContext], Callable]:
    def build(context: MethodContext):
        try:
            from kv_quant.factory import parse_method
        except ImportError as error:  # pragma: no cover - depends on checkout
            raise MethodUnavailable(f"kv_quant is not importable: {error}") from error
        _, quantizer = parse_method(method_name)
        if quantizer is None:
            raise MethodUnavailable(f"{method_name} did not produce a quantizer")

        def run(keys: torch.Tensor, values: torch.Tensor):
            rebuilt_keys = torch.empty_like(keys)
            rebuilt_values = torch.empty_like(values)
            for layer in range(keys.shape[0]):
                # One layer is already [B, L, H, D].
                state = quantizer.quantize_kv(keys[layer], values[layer])
                layer_keys, layer_values = quantizer.dequantize_kv(state)
                rebuilt_keys[layer] = layer_keys.float()
                rebuilt_values[layer] = layer_values.float()
            return rebuilt_keys, rebuilt_values

        return run

    return build


# --------------------------------------------------------------------------
# QVG
# --------------------------------------------------------------------------


def _build_qvg(bits: int) -> Callable[[MethodContext], Callable]:
    def build(context: MethodContext):
        qvg_path = REPO_ROOT / "third_party" / "Quant-VideoGen"
        if str(qvg_path) not in sys.path:
            sys.path.insert(0, str(qvg_path))
        try:
            compress = importlib.import_module("quant_videogen.compress")
            uncompress = importlib.import_module("quant_videogen.uncompress")
        except ImportError as error:
            # The published QVG codec is a Triton kernel, so a CPU-only host
            # genuinely cannot run it.  Say so rather than substituting the
            # simulation path and labelling the result "QVG".
            raise MethodUnavailable(
                f"QVG needs its Triton codec, which is unavailable here: {error}"
            ) from error
        sys.path.insert(0, str(REPO_ROOT / "Causal-Forcing"))
        runtime = importlib.import_module("qvg_runtime")
        config = runtime.QVGConfig(bits=bits)
        quantize_fn = compress.get_quantize_fn(config.quant_type, config)

        def run(keys: torch.Tensor, values: torch.Tensor):
            rebuilt_keys = torch.empty_like(keys)
            rebuilt_values = torch.empty_like(values)
            for layer in range(keys.shape[0]):
                # QVG is driven in [B, H, L, D], the same permute the
                # Causal-Forcing runtime applies before compressing a span.
                layer_keys = keys[layer].permute(0, 2, 1, 3).contiguous()
                layer_values = values[layer].permute(0, 2, 1, 3).contiguous()
                packed_keys, packed_values = compress.compress_kv_cache(
                    layer_keys, layer_values, config.quant_type, config, quantize_fn
                )
                for packed in (packed_keys, packed_values):
                    if not isinstance(packed, dict):
                        raise MethodUnavailable(
                            "QVG real codec did not return packed states"
                        )
                    packed["info"] = {
                        "output_dtype": layer_keys.dtype,
                        "quant_config": config,
                    }
                out_keys, out_values = uncompress.uncompress_kv_cache(
                    packed_keys, packed_values
                )
                rebuilt_keys[layer] = out_keys.permute(0, 2, 1, 3).float()
                rebuilt_values[layer] = out_values.permute(0, 2, 1, 3).float()
            return rebuilt_keys, rebuilt_values

        return run

    return build


# --------------------------------------------------------------------------
# TempoKV
# --------------------------------------------------------------------------


def _build_tempokv(bits: int) -> Callable[[MethodContext], Callable]:
    def build(context: MethodContext):
        root = context.tempokv_root or DEFAULT_TEMPOKV_ROOT
        source = Path(root) / "src"
        if not source.is_dir():
            raise MethodUnavailable(
                f"TempoKV sources not found at {source}; pass --tempokv-root"
            )
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        try:
            config_module = importlib.import_module("tempokv.config")
            fitting = importlib.import_module("tempokv.fitting")
        except ImportError as error:
            raise MethodUnavailable(f"TempoKV is not importable: {error}") from error
        if context.calibration is None:
            # Unlike the quantizer baselines, TempoKV fits predictors.  Scoring
            # it on a trace it was fitted on would flatter it against methods
            # that never see the data twice.
            raise MethodUnavailable(
                "TempoKV needs a separate --calibration-trace to fit predictors"
            )
        config = config_module.config_from_mapping(
            {
                "k_bits": bits,
                "v_bits": bits,
                "anchor_bits": 4,
                "group_size": 64,
                "stride": context.stride,
                "residual_mode": "asymmetric",
                "scale_dtype": "bfloat16",
                "reconstruction_dtype": "bfloat16",
                "k_predictor_mode": "shared",
                "v_predictor_mode": "cross_kv",
                "k_history_length": 2,
                "v_history_length": 1,
                "cross_kv_weight_mode": "layer_shared",
                "cross_matmul_precision": "ieee",
                "fit_mode": "legacy",
            }
        )
        calibration_keys, calibration_values = context.calibration
        temporal, cross = fitting.fit_predictors(
            calibration_keys,
            calibration_values,
            config=config,
            iterations=context.fit_iterations,
            temporal_ridge=context.temporal_ridge,
            cross_ridge=context.cross_ridge,
            v_history_mode="diagonal",
            coordinate_system="pre_rope_normalized",
        )

        def run(keys: torch.Tensor, values: torch.Tensor):
            rebuilt_keys = fitting._reconstruct_temporal_trace(keys, temporal, config)
            rebuilt_values = fitting._reconstruct_cross_kv_trace(
                rebuilt_keys, values, cross, config
            )
            return rebuilt_keys, rebuilt_values

        return run

    return build


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def _methods() -> tuple[Method, ...]:
    entries: list[Method] = [
        Method("bf16", "BF16", "bf16", None, _build_bf16,
               notes="storage floor; no quantization beyond bfloat16")
    ]
    for bits in (2, 4):
        entries.extend(
            [
                Method(f"rtn_int{bits}", f"RTN INT{bits}", "rtn", bits,
                       _build_kv_quant(f"RTN_INT{bits}")),
                Method(f"kivi_int{bits}", f"KIVI INT{bits}", "kivi", bits,
                       _build_kv_quant(f"KIVI_INT{bits}")),
                Method(f"quarot_int{bits}", f"QuaRot INT{bits}", "quarot", bits,
                       _build_kv_quant(f"QUAROT_KV_INT{bits}")),
                Method(f"hadamard_int{bits}", f"Hadamard-K INT{bits}", "hadamard",
                       bits, _build_kv_quant(f"HADAMARD_K_INT{bits}")),
                Method(f"qvg_int{bits}", f"QVG INT{bits}", "qvg", bits,
                       _build_qvg(bits), notes="needs the Triton codec"),
                Method(f"tempokv_int{bits}", f"TempoKV INT{bits}", "tempokv", bits,
                       _build_tempokv(bits), needs_calibration=True,
                       notes="fits predictors on the calibration trace"),
            ]
        )
    return tuple(entries)


METHODS = _methods()
METHODS_BY_NAME = {item.name: item for item in METHODS}
DEFAULT_METHODS = (
    "bf16",
    "rtn_int2",
    "kivi_int2",
    "quarot_int2",
    "qvg_int2",
    "tempokv_int2",
)


def method(name: str) -> Method:
    try:
        return METHODS_BY_NAME[name]
    except KeyError:
        available = ", ".join(METHODS_BY_NAME)
        raise ValueError(f"unknown method {name!r}; known methods are {available}") from None


# --------------------------------------------------------------------------
# Per-frame error
# --------------------------------------------------------------------------


def frame_curve(clean: torch.Tensor, rebuilt: torch.Tensor) -> dict[str, Any]:
    """Per-frame, cumulative, and pooled NMSE along the frame axis.

    ``cumulative[t]`` covers frames 0..t, so its last entry *is* the pooled
    figure and a table cannot disagree with the plotted curve.
    """
    frames = int(clean.shape[2])
    per_frame: list[dict[str, Any]] = []
    cumulative: list[dict[str, Any]] = []
    running_error = 0.0
    running_reference = 0.0
    for index in range(frames):
        reference = clean[:, :, index].double()
        error = rebuilt[:, :, index].double() - reference
        error_energy = float(error.square().sum())
        reference_energy = float(reference.square().sum())
        running_error += error_energy
        running_reference += reference_energy
        per_frame.append(
            {
                "frame": index,
                "nmse": (
                    error_energy / reference_energy if reference_energy > 0.0 else None
                ),
            }
        )
        cumulative.append(
            {
                "frame": index,
                "nmse": (
                    running_error / running_reference
                    if running_reference > 0.0
                    else None
                ),
            }
        )
    return {
        "frames": frames,
        "per_frame": per_frame,
        "cumulative": cumulative,
        "pooled_nmse": (
            running_error / running_reference if running_reference > 0.0 else None
        ),
    }


def summarize_curve(curve: dict[str, Any], *, tail_fraction: float = 0.25) -> dict[str, Any]:
    values = [row["nmse"] for row in curve["per_frame"] if row["nmse"] is not None]
    if not values:
        raise ValueError("curve contains no finite frames")
    frames = len(values)
    tail_length = min(frames, max(1, int(round(frames * tail_fraction))))
    tail_start = frames - tail_length
    tail = values[tail_start:]
    # Everything before the tail window; when the tail is the whole sequence
    # there is no head to compare against and the ratio is 1 by construction.
    head = values[:tail_start] if tail_start > 0 else tail
    head_mean = sum(head) / len(head)
    tail_mean = sum(tail) / len(tail)
    return {
        "pooled_nmse": curve["pooled_nmse"],
        "first_frame_nmse": values[0],
        "last_frame_nmse": values[-1],
        "max_frame_nmse": max(values),
        "tail_start_frame": tail_start,
        "head_mean_nmse": head_mean,
        "tail_mean_nmse": tail_mean,
        "tail_over_head": tail_mean / head_mean if head_mean > 0.0 else None,
    }


def write_report(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
