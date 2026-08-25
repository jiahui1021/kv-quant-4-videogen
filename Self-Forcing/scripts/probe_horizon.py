#!/usr/bin/env python3
"""Bisect the largest feasible global-window cache for KVQuant4Video."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kv_quant import efficiency_record as efficiency


def bisect_horizon(
    fits: Callable[[int], bool], low: int, high: int
) -> tuple[int, list[dict]]:
    """Return the largest fitting point on the 12-frame horizon grid."""
    low = efficiency.largest_grid_horizon(int(low))
    high = efficiency.largest_grid_horizon(int(high))
    if low > high:
        return 0, []
    grid = list(range(low, high + 1, 12))
    log: list[dict] = []
    left, right = 0, len(grid) - 1
    best = -1
    while left <= right:
        middle = (left + right) // 2
        frames = grid[middle]
        fit = bool(fits(frames))
        log.append({"frames": frames, "fit": fit})
        if fit:
            best = middle
            left = middle + 1
        else:
            right = middle - 1
    return (grid[best] if best >= 0 else 0), log


def _load_generate_module():
    spec = importlib.util.spec_from_file_location(
        "kvquant_generate_for_horizon",
        REPO_ROOT / "scripts" / "01_generate.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load Self-Forcing/01_generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cache_fits(frames: int, args: argparse.Namespace) -> bool:
    """Construct the pipeline and allocate only the candidate cache."""
    generate = _load_generate_module()
    latent_frames = (int(frames) + 3) // 4
    try:
        pipeline = generate.initialize_pipeline(
            config_path=args.config_path,
            default_config_path=args.default_config_path,
            checkpoint_path=args.checkpoint_path,
            use_ema=True,
            device=torch.device("cuda"),
            low_memory=False,
            local_attn_size=latent_frames,
        )
        pipeline._initialize_kv_cache(
            batch_size=1,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        generate.ensure_kv_cache_capacity(
            pipeline,
            latent_frames,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _paper_label(method_name: str) -> tuple[str, int | None]:
    labels = {
        "BF16": ("BF16", None),
        "RTN_INT2": ("RTN", 2),
        "RTN_INT4": ("RTN", 4),
        "KIVI_INT2": ("KIVI", 2),
        "KIVI_INT4": ("KIVI", 4),
        "QUAROT_KV_INT2": ("QuaRot-KV", 2),
        "QUAROT_KV_INT4": ("QuaRot-KV", 4),
    }
    try:
        return labels[method_name]
    except KeyError as error:
        raise ValueError(f"unsupported horizon method: {method_name}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--default-config-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--probe-frames", type=int, default=3009)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-path", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    best, probes = bisect_horizon(
        lambda frames: _cache_fits(frames, args),
        low=9,
        high=args.probe_frames,
    )
    method, bits = _paper_label(args.method)
    latent_probe = (int(args.probe_frames) + 3) // 4
    prompt_indices = [0]
    prompt_hash = ""
    if args.prompt_path and args.prompt_path.is_file():
        import hashlib

        prompt_hash = hashlib.sha256(args.prompt_path.read_bytes()).hexdigest()
    payload = efficiency.build_horizon_payload(
        repo="KVQuant4Video",
        method=method,
        bits=bits,
        run={
            "model": "Self-Forcing",
            "width": 832,
            "height": 480,
            "frames": int(args.probe_frames),
            "latent_frames": latent_probe,
            "prompts_sha256": prompt_hash,
            "prompt_indices": prompt_indices,
            "seed": int(args.seed),
            "num_gpus": 1,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_uuid": "",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "",
            "local_attn_size": latent_probe,
        },
        budget={
            "device_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "exclusive_device": True,
        },
        probe_mode="bisect_on_init",
        max_feasible_frames=best,
        oom_reached=best < int(args.probe_frames),
        peak_allocated_bytes_at_max=int(torch.cuda.max_memory_allocated()),
        kv_resident_bytes_at_max=0,
        bisect_probes=probes,
    )
    efficiency.write_record(args.output, payload)
    print(f"{args.method}: max feasible horizon {best} frames ({len(probes)} probes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
