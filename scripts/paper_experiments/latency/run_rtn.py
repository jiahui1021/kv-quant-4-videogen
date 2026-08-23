#!/usr/bin/env python3
"""Run the Self-Forcing RTN latency adapter for the shared paper harness."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
SELF_FORCING_ROOT = REPO_ROOT / "Self-Forcing"


def pixel_to_latent_frames(pixel_frames: int) -> int:
    if pixel_frames <= 0 or (pixel_frames + 3) % 4:
        raise ValueError(
            "Self-Forcing requires pixel frames satisfying (frames + 3) % 4 == 0"
        )
    return (pixel_frames + 3) // 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("compilation", "pair"), required=True)
    parser.add_argument(
        "--method", choices=("RTN_INT2", "RTN_INT4"), required=True
    )
    parser.add_argument("--frames", type=int, required=True, help="pixel frames")
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--default-config-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--prompt-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--low-memory", action="store_true")
    return parser


def command(args: argparse.Namespace) -> list[str]:
    latent_frames = pixel_to_latent_frames(args.frames)
    seed = os.environ.get("PAPER_EXPERIMENT_SEED")
    if seed is None:
        raise ValueError("Missing PAPER_EXPERIMENT_SEED")
    result = [
        args.python,
        str(SELF_FORCING_ROOT / "scripts/01_generate.py"),
        "--config-path",
        str(args.config_path.expanduser().resolve()),
        "--default-config-path",
        str(args.default_config_path.expanduser().resolve()),
        "--checkpoint-path",
        str(args.checkpoint_path.expanduser().resolve()),
        "--prompt-path",
        str(args.prompt_path.expanduser().resolve()),
        "--results-root",
        str(args.output_dir.expanduser().resolve()),
        "--num-output-frames",
        str(latent_frames),
        "--method",
        args.method,
        "--block-size",
        str(args.block_size),
        "--local-attn-size",
        str(latent_frames),
        "--max-prompts",
        "1",
        "--seed",
        seed,
        "--no-log-vram-trace",
        "--paper-latency-phase",
        args.phase,
    ]
    if args.use_ema:
        result.append("--use-ema")
    if args.low_memory:
        result.append("--low-memory")
    return result


def validate_environment(args: argparse.Namespace) -> None:
    required = (
        "PAPER_EXPERIMENT_METHOD",
        "PAPER_EXPERIMENT_PHASE",
        "PAPER_EXPERIMENT_FRAMES",
        "PAPER_EXPERIMENT_REPEAT",
        "PAPER_EXPERIMENT_SEED",
        "PAPER_EXPERIMENT_SESSION_ID",
        "PAPER_EXPERIMENT_SESSION_SCOPE",
        "PAPER_EXPERIMENT_RUNTIME_INSTANCE_ID",
        "PAPER_EXPERIMENT_FROZEN_JSON",
    )
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise ValueError("Missing paper harness environment: " + ", ".join(missing))
    expected = {
        "PAPER_EXPERIMENT_METHOD": args.method,
        "PAPER_EXPERIMENT_FRAMES": str(args.frames),
        "PAPER_EXPERIMENT_PHASE": args.phase,
    }
    mismatches = {
        name: (value, os.environ.get(name))
        for name, value in expected.items()
        if os.environ.get(name) != value
    }
    if mismatches:
        rendered = ", ".join(
            f"{name}: expected {expected!r}, got {actual!r}"
            for name, (expected, actual) in mismatches.items()
        )
        raise ValueError("paper harness environment mismatch: " + rendered)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_environment(args)
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command(args), cwd=REPO_ROOT, env=os.environ.copy())
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
