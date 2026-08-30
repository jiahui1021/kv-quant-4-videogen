#!/usr/bin/env python3
"""Minimal complete run of the cross-method reconstruction comparison.

Builds a short calibration trace and a longer evaluation trace, runs every
method, and renders the figure -- on CPU, in seconds.  Methods that need a GPU
codec report themselves unavailable, which is part of what this checks: a
method must never disappear silently.

    python scripts/paper_experiments/reconstruction/run_minimal.py \\
        --output-root /tmp/reconstruction-minimal

The numbers describe the synthetic generator and say nothing about any method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from methods import DEFAULT_TEMPOKV_ROOT, METHODS  # noqa: E402


LAYERS, RECORDS, HEADS, DIM, STRIDE = 3, 6, 2, 128, 4
CALIBRATION_FRAMES = 12
EVALUATION_FRAMES = 40
MINIMAL_METHODS = (
    "bf16",
    "rtn_int2",
    "kivi_int2",
    "quarot_int2",
    "hadamard_int2",
    "qvg_int2",
    "tempokv_int2",
)


def synthesize(path: Path, frames: int, *, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    shape = (LAYERS, RECORDS, HEADS, DIM)
    keys = np.zeros((LAYERS, RECORDS, frames, HEADS, DIM), dtype=np.float32)
    values = np.zeros_like(keys)
    keys[:, :, 0] = rng.normal(size=shape)
    values[:, :, 0] = rng.normal(size=shape)
    for frame in range(1, frames):
        keys[:, :, frame] = 0.9 * keys[:, :, frame - 1] + 0.1 * rng.normal(size=shape)
        values[:, :, frame] = (
            0.6 * keys[:, :, frame]
            + 0.3 * values[:, :, frame - 1]
            + 0.1 * rng.normal(size=shape)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        keys=keys,
        values=values,
        layer_ids=np.arange(LAYERS, dtype=np.int32),
        trace_stride=np.asarray(STRIDE, dtype=np.int32),
        coordinate_system=np.asarray("pre_rope_normalized"),
        cross_source_kind=np.asarray("normalized"),
        trace_role=np.asarray("calibration"),
    )
    return path


def _run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    if subprocess.run(command, text=True).returncode != 0:
        raise SystemExit(f"step failed: {' '.join(command)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tempokv-root", type=Path, default=DEFAULT_TEMPOKV_ROOT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--python-bin", default=sys.executable)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.output_root.expanduser().resolve()
    calibration = synthesize(
        root / "traces" / "calibration.npz", CALIBRATION_FRAMES, seed=args.seed
    )
    evaluation = synthesize(
        root / "traces" / "evaluation.npz", EVALUATION_FRAMES, seed=args.seed + 1
    )

    report_dir = root / "reconstruction"
    _run(
        [args.python_bin, str(HERE / "run_reconstruction.py"),
         "--trace", str(evaluation),
         "--calibration-trace", str(calibration),
         "--tempokv-root", str(args.tempokv_root),
         "--output-dir", str(report_dir),
         "--methods", *MINIMAL_METHODS]
    )
    _run(
        [args.python_bin, str(HERE / "plot_figure.py"),
         "--report", str(report_dir / "reconstruction.json"),
         "--output-dir", str(root / "figure")]
    )

    expected = [
        report_dir / "reconstruction.json",
        root / "figure" / "reconstruction_by_frame.csv",
        root / "figure" / "reconstruction_summary.csv",
    ]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise SystemExit(
            "the run finished but did not produce: "
            + ", ".join(str(path) for path in missing)
        )

    payload = json.loads(expected[0].read_text(encoding="utf-8"))
    covered = {item["method"] for item in payload["methods"]}
    if covered != set(MINIMAL_METHODS):
        raise SystemExit(f"report is missing methods: {sorted(set(MINIMAL_METHODS) - covered)}")
    # A method may legitimately be unavailable here, but never silently absent,
    # and it must never be reported as failed.
    failed = [
        item["method"] for item in payload["methods"] if item["status"] == "failed"
    ]
    if failed:
        raise SystemExit(f"methods failed rather than running or declining: {failed}")
    for item in payload["methods"]:
        if item["status"] == "unavailable":
            print(f"  {item['method']}: unavailable — {item['reason']}")

    print("\nminimal reconstruction comparison complete")
    for path in expected:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
