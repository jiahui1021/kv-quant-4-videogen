#!/usr/bin/env python3
"""Reconstruction error for every KV quantization method on one trace.

One collected trace is scored by BF16, RTN, KIVI, QuaRot, Hadamard-K, QVG and
TempoKV through a single measurement path, reporting the error at each frame
index as well as the pooled figure.

    python scripts/paper_experiments/reconstruction/run_reconstruction.py \\
        --trace data/long-horizon/1401f/clean_trace.npz \\
        --calibration-trace data/self-forcing/trace/clean_trace.npz \\
        --tempokv-root ../Tempokv \\
        --output-dir results/reconstruction/1401f

Methods that cannot run on this host are reported with the reason rather than
dropped: QVG needs its Triton codec and TempoKV needs a calibration trace, so
a CPU-only box legitimately produces fewer rows and says why.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from methods import (  # noqa: E402
    DEFAULT_METHODS,
    METHODS,
    SCHEMA_VERSION,
    MethodContext,
    MethodUnavailable,
    frame_curve,
    method,
    summarize_curve,
    write_report,
)


def load_trace(path: Path, *, max_records: int | None = None):
    """Read a trace as [layers, records, frames, heads, dim] float32 plus stride."""
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in ("keys", "values") if name not in archive]
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
        keys = torch.from_numpy(np.array(archive["keys"], copy=True)).float()
        values = torch.from_numpy(np.array(archive["values"], copy=True)).float()
        stride = None
        for name in ("trace_stride", "stride"):
            if name in archive:
                stride = int(np.asarray(archive[name]).item())
                break
    if keys.shape != values.shape:
        raise ValueError(f"{path} K and V must share a shape")
    if stride is None:
        raise ValueError(f"{path} does not record a trace stride")
    # The two Forcing bridges can write a seven-dimensional event layout; its
    # records run event-major, so collapsing the extra axes preserves order.
    if keys.ndim == 7:
        layers, events, batch, tokens = keys.shape[:4]
        keys = keys.reshape(layers, events * batch * tokens, *keys.shape[4:])
        values = values.reshape(layers, events * batch * tokens, *values.shape[4:])
    if keys.ndim != 5:
        raise ValueError(
            f"{path} must use [layers, records, frames, heads, dim], got "
            f"{tuple(keys.shape)}"
        )
    if max_records is not None:
        if max_records < 1:
            raise ValueError("--max-records must be positive")
        keys = keys[:, :max_records].contiguous()
        values = values[:, :max_records].contiguous()
    return keys, values, stride


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--calibration-trace",
        type=Path,
        help="separate trace for methods that fit predictors (TempoKV)",
    )
    parser.add_argument("--tempokv-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="subset of: " + ", ".join(item.name for item in METHODS),
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--fit-iterations", type=int, default=2)
    parser.add_argument("--model", default="self-forcing")
    parser.add_argument("--report-name", default="reconstruction.json")
    return parser


def run(args: argparse.Namespace) -> dict:
    if not 0.0 < args.tail_fraction <= 1.0:
        raise ValueError("--tail-fraction must be in (0, 1]")
    keys, values, stride = load_trace(args.trace, max_records=args.max_records)
    calibration = None
    if args.calibration_trace is not None:
        calibration_keys, calibration_values, calibration_stride = load_trace(
            args.calibration_trace
        )
        if calibration_stride != stride:
            raise ValueError(
                f"calibration stride {calibration_stride} does not match trace "
                f"stride {stride}; the two traces come from different runs"
            )
        calibration = (calibration_keys, calibration_values)

    frames = int(keys.shape[2])
    print(
        f"[reconstruction] trace {tuple(keys.shape)}, {frames} frames, stride {stride}",
        file=sys.stderr,
        flush=True,
    )
    context = MethodContext(
        stride=stride,
        calibration=calibration,
        tempokv_root=args.tempokv_root,
        fit_iterations=args.fit_iterations,
    )

    results = []
    for name in args.methods:
        spec = method(name)
        print(f"[reconstruction] {spec.name}", file=sys.stderr, flush=True)
        try:
            runner = spec.build(context)
            rebuilt_keys, rebuilt_values = runner(keys, values)
        except MethodUnavailable as error:
            results.append(
                {
                    "method": spec.name,
                    "label": spec.label,
                    "family": spec.family,
                    "bits": spec.bits,
                    "status": "unavailable",
                    "reason": str(error),
                }
            )
            print(f"    unavailable: {error}", file=sys.stderr, flush=True)
            continue
        except Exception as error:  # noqa: BLE001 - one method must not stop the rest
            results.append(
                {
                    "method": spec.name,
                    "label": spec.label,
                    "family": spec.family,
                    "bits": spec.bits,
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(limit=6),
                }
            )
            print(f"    failed: {type(error).__name__}: {error}", file=sys.stderr)
            continue
        curves = {
            "k": frame_curve(keys, rebuilt_keys),
            "v": frame_curve(values, rebuilt_values),
        }
        # These are the largest objects in the process on a long trace; drop
        # them before the next method allocates its own.
        del rebuilt_keys, rebuilt_values
        gc.collect()
        results.append(
            {
                "method": spec.name,
                "label": spec.label,
                "family": spec.family,
                "bits": spec.bits,
                "status": "complete",
                "notes": spec.notes,
                "k": curves["k"],
                "v": curves["v"],
                "summary": {
                    tensor: summarize_curve(
                        curves[tensor], tail_fraction=args.tail_fraction
                    )
                    for tensor in ("k", "v")
                },
            }
        )

    payload = {
        "module": "reconstruction_comparison",
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "frames": frames,
        "stride": stride,
        "trace": str(args.trace.resolve()),
        "trace_shape": list(keys.shape),
        "calibration_trace": (
            str(args.calibration_trace.resolve())
            if args.calibration_trace is not None
            else None
        ),
        "max_records": args.max_records,
        "tail_fraction": args.tail_fraction,
        "methods": results,
    }
    write_report(args.output_dir / args.report_name, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run(args)
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        f"{'method':<18}{'tensor':>7}{'frame 0':>12}{'last':>12}"
        f"{'tail mean':>12}{'pooled':>12}"
    )
    for item in payload["methods"]:
        if item["status"] != "complete":
            print(f"{item['method']:<18}  {item['status']}: {item['reason'][:60]}")
            continue
        for tensor in ("k", "v"):
            summary = item["summary"][tensor]
            print(
                f"{item['method']:<18}{tensor.upper():>7}"
                f"{summary['first_frame_nmse']:>12.6f}"
                f"{summary['last_frame_nmse']:>12.6f}"
                f"{summary['tail_mean_nmse']:>12.6f}"
                f"{summary['pooled_nmse']:>12.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
