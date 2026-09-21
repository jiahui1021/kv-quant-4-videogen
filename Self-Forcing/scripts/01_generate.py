#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import threading
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF_FORCING_ROOT = REPO_ROOT / "third_party" / "Self-Forcing"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SELF_FORCING_ROOT) not in sys.path:
    sys.path.insert(0, str(SELF_FORCING_ROOT))

from kv_quant.factory import create_quantizer
from kv_quant import efficiency_record
from kv_quant.efficiency_hook import SamplingQuantizer
from pipeline import CausalInferencePipeline, CausalDiffusionInferencePipeline
from utils.misc import set_seed
from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb



def _quarot_kwargs(
    base: str,
    channel_group_size: Optional[int],
    asym: Optional[bool],
    clip_ratio: Optional[float],
) -> Dict[str, object]:
    """QuaRot's range knobs, which RTN and KIVI reject rather than ignore."""
    requested = {
        "--kv-channel-group-size": channel_group_size,
        "--kv-asym": asym,
        "--kv-clip-ratio": clip_ratio,
    }
    if base != "QUAROT_KV":
        named = sorted(flag for flag, value in requested.items() if value is not None)
        if named:
            raise ValueError(
                f"{', '.join(named)} only applies to QUAROT_KV, not {base}; "
                "passing it here would silently change nothing"
            )
        return {}
    extra: Dict[str, object] = {}
    if channel_group_size is not None:
        extra["channel_group_size"] = int(channel_group_size)
    if asym is not None:
        extra["asym"] = bool(asym)
    if clip_ratio is not None:
        extra["clip_ratio"] = float(clip_ratio)
    return extra


def parse_method(
    method: str,
    bits: Optional[int],
    block_size: Optional[int],
    kivi_residual_length: Optional[int] = None,
    quarot_channel_group_size: Optional[int] = None,
    quarot_asym: Optional[bool] = None,
    quarot_clip_ratio: Optional[float] = None,
):
    """Return ``(canonical_name, quantizer_or_none)`` for a CLI method.

    ``RTN_INT4`` and the other suffixed spellings carry their own width; a bare
    ``RTN`` takes it from ``--bits``.  ``--block-size`` left unset keeps each
    baseline at its published grouping.
    """
    method = method.upper()
    if method == "BF16":
        return "BF16", None

    base, parsed_bits = method, bits
    match = re.fullmatch(r"(RTN|KIVI|QUAROT_KV|HADAMARD_K)_INT(\d+)", method)
    if match is not None:
        base, parsed_bits = match.group(1), int(match.group(2))
    if base not in ("RTN", "KIVI", "QUAROT_KV", "HADAMARD_K"):
        raise ValueError(
            f"Unsupported method={method}. Expected BF16, RTN, KIVI, QUAROT_KV "
            "or HADAMARD_K, with or without an _INT<bits> suffix"
        )
    if parsed_bits is None:
        raise ValueError(f"method={method} needs --bits")

    return f"{base}_INT{parsed_bits}", create_quantizer(
        base,
        bits=parsed_bits,
        block_size=block_size,
        residual_length=kivi_residual_length if base == "KIVI" else None,
        **_quarot_kwargs(base, quarot_channel_group_size, quarot_asym, quarot_clip_ratio),
    )


_PAPER_METHODS = {
    "BF16": ("BF16", None),
    "RTN_INT2": ("RTN", 2), "RTN_INT4": ("RTN", 4),
    "KIVI_INT2": ("KIVI", 2), "KIVI_INT4": ("KIVI", 4),
    "QUAROT_KV_INT2": ("QuaRot", 2), "QUAROT_KV_INT4": ("QuaRot", 4),
}


def _paper_method_label(method_name: str):
    """Map a runtime method name onto its row in the paper's tables.

    Only the methods that keep the full sequence are accepted.  The pruning
    families evict tokens, which breaks the collector's requirement that every
    row's BF16 equivalent match the BF16 run's resident bytes, so they are
    rejected here rather than quietly producing an incomparable row.
    """
    try:
        return _PAPER_METHODS[method_name]
    except KeyError:
        raise ValueError(
            f"{method_name} is not one of the paper's efficiency rows; "
            "token-evicting methods need their own accounting (design spec 4.4)"
        ) from None


def _efficiency_gpu_uuid() -> str:
    """The device UUID, so the collector can prove one card ran every method.

    Best effort: this runs after a generation has already produced its videos,
    and no provenance lookup is worth losing that work to.  An empty result
    hides nothing -- the collector rejects a record whose ``gpu_uuid`` does not
    match the manifest, empty included.
    """
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
    except Exception:
        return ""


def load_prompts(prompt_path: Path, max_prompts: Optional[int]) -> List[Tuple[int, str]]:
    prompts: List[Tuple[int, str]] = []
    with prompt_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            prompt = line.strip()
            if not prompt:
                continue
            prompts.append((idx, prompt))
            if max_prompts is not None and len(prompts) >= max_prompts:
                break
    return prompts


def git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])  # noqa: S603
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def reset_kv_state(pipeline, quantizer):
    if pipeline.kv_cache1 is None:
        return
    if quantizer is not None and hasattr(quantizer, "reset_prompt_state"):
        quantizer.reset_prompt_state()
    for block in pipeline.kv_cache1:
        block["global_end_index"].fill_(0)
        block["local_end_index"].fill_(0)
        if isinstance(block.get("k"), torch.Tensor) and block["k"].numel() > 0:
            block["k"].zero_()
        if isinstance(block.get("v"), torch.Tensor) and block["v"].numel() > 0:
            block["v"].zero_()
        if quantizer is not None:
            block["quantizer"] = quantizer
            block["quant_state"] = None
            recent_k = block.get("recent_k")
            recent_v = block.get("recent_v")
            if isinstance(recent_k, torch.Tensor):
                block["recent_k"] = recent_k.new_empty((recent_k.shape[0], 0, recent_k.shape[2], recent_k.shape[3]))
            if isinstance(recent_v, torch.Tensor):
                block["recent_v"] = recent_v.new_empty((recent_v.shape[0], 0, recent_v.shape[2], recent_v.shape[3]))
            block["recent_start_index"] = 0
            block["recent_end_index"] = 0
            block["quantize_on_write"] = block.get("quantize_cadence", "per_step") == "per_step"


def finalize_kv_state(pipeline, quantizer) -> None:
    """Commit the final mutable write buffer for resident-byte reporting."""
    if quantizer is None or not callable(getattr(quantizer, "finalize_state", None)):
        return
    for block in getattr(pipeline, "kv_cache1", None) or []:
        state = block.get("quant_state")
        if state is None:
            continue
        write_k = state.get("write_k") if isinstance(state, dict) else None
        dtype = write_k.dtype if isinstance(write_k, torch.Tensor) else block.get("dtype", torch.bfloat16)
        quantizer.finalize_state(state, meta={"tensor_dtype": dtype})


def initialize_pipeline(
    config_path: Path,
    default_config_path: Path,
    checkpoint_path: Path,
    use_ema: bool,
    device: torch.device,
    low_memory: bool,
    local_attn_size: Optional[int] = None,
):
    config = OmegaConf.load(str(default_config_path))
    config = OmegaConf.merge(config, OmegaConf.load(str(config_path)))
    if local_attn_size is not None:
        config.model_kwargs.local_attn_size = int(local_attn_size)

    if hasattr(config, "denoising_step_list"):
        pipeline = CausalInferencePipeline(config, device=device)
    else:
        pipeline = CausalDiffusionInferencePipeline(config, device=device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    key = "generator_ema" if use_ema else "generator"
    if key not in state_dict:
        raise KeyError(f"Checkpoint missing key '{key}'. Keys: {list(state_dict.keys())[:10]}")
    pipeline.generator.load_state_dict(state_dict[key])

    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device)
    pipeline.generator.to(device)
    pipeline.vae.to(device)
    return pipeline


def tensor_shape_to_resolution(video: torch.Tensor) -> Tuple[int, int]:
    # [B, T, C, H, W]
    _, _, _, h, w = video.shape
    return int(h), int(w)


def ensure_kv_cache_capacity(pipeline, num_output_frames: int, dtype: torch.dtype, device: torch.device) -> None:
    if not hasattr(pipeline, "kv_cache1") or pipeline.kv_cache1 is None:
        return
    frame_seq_length = int(getattr(pipeline, "frame_seq_length", 0))
    if frame_seq_length <= 0:
        return
    required_tokens = int(num_output_frames) * frame_seq_length
    for block in pipeline.kv_cache1:
        k = block.get("k")
        v = block.get("v")
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor) or k.ndim != 4 or v.ndim != 4:
            continue
        current_tokens = int(k.shape[1])
        if required_tokens <= current_tokens:
            continue
        batch_size, _, num_heads, head_dim = k.shape
        new_k = torch.zeros([batch_size, required_tokens, num_heads, head_dim], dtype=dtype, device=device)
        new_v = torch.zeros_like(new_k)
        if current_tokens > 0:
            new_k[:, :current_tokens] = k
            new_v[:, :current_tokens] = v
        block["k"] = new_k
        block["v"] = new_v


def _current_active_kv_bytes(pipeline, quantizer) -> tuple[int, int]:
    """Return BF16-equivalent and resident bytes from live cache tensors.

    The old trace path reconstructed this number from cache policy metadata.
    Efficiency records use the sampling wrapper's actual tensor walk instead;
    the fallback below keeps the diagnostic trace useful for unwrapped callers
    without reintroducing a configuration-derived byte figure.
    """
    kv_cache = getattr(pipeline, "kv_cache1", None) or []
    if not kv_cache:
        return 0, 0

    resident_method = getattr(quantizer, "resident_kv_bytes", None)
    if callable(resident_method):
        resident, equivalent = resident_method()
        return int(equivalent), int(resident)

    from kv_quant.efficiency_record import tensor_bytes, bf16_equivalent_bytes

    resident = tensor_bytes(kv_cache)
    first = kv_cache[0]
    end_index = first.get("local_end_index", 0)
    tokens = int(end_index.item()) if isinstance(end_index, torch.Tensor) else int(end_index)
    geometry = first.get("k")
    if isinstance(geometry, torch.Tensor) and geometry.ndim == 4:
        batch, _, heads, head_dim = geometry.shape
    else:
        batch = int(first.get("batch_size", 0))
        heads = int(first.get("num_heads", 0))
        head_dim = int(first.get("head_dim", 0))
    equivalent = bf16_equivalent_bytes(
        int(batch), tokens, int(heads), int(head_dim)
    ) * len(kv_cache)
    if quantizer is None:
        # With BF16 caches the tensor walk is already the resident source.  The
        # equivalent is kept tied to the same live allocation for preallocated
        # cache implementations.
        equivalent = resident
    return int(equivalent), int(resident)


def _dense_kv_bytes(pipeline) -> int:
    """BF16 cost of the tokens the cache holds, not of its allocation.

    A preallocated cache is sized for the whole video before the first token is
    written, so its tensor bytes answer a different question than the token
    count the shared record compares across repositories.
    """
    from kv_quant.efficiency_record import bf16_equivalent_bytes

    kv_cache = getattr(pipeline, "kv_cache1", None) or []
    if not kv_cache:
        return 0
    first = kv_cache[0]
    end_index = first.get("local_end_index", 0)
    tokens = int(end_index.item()) if isinstance(end_index, torch.Tensor) else int(end_index)
    geometry = first.get("k")
    if isinstance(geometry, torch.Tensor) and geometry.ndim == 4 and geometry.shape[1] > 0:
        batch, _, heads, head_dim = geometry.shape
    else:
        batch = int(first.get("batch_size", 0))
        heads = int(first.get("num_heads", 0))
        head_dim = int(first.get("head_dim", 0))
    return bf16_equivalent_bytes(int(batch), tokens, int(heads), int(head_dim)) * len(kv_cache)


def _resident_analytic_bytes(pipeline, quantizer) -> Tuple[int, int]:
    """``(bf16_equivalent, resident)`` bytes under ``resident_analytic.v1``.

    A quantized run reports the peak its block-boundary sampler saw, which is
    the same pair the shared efficiency record carries, so this repository's
    own metrics file and the cross-repository record can no longer disagree.
    A BF16 run has no sampler: its dense cache is the baseline that every
    method's equivalent must match, so both halves are the analytic figure.
    """
    sampler = getattr(quantizer, "sampler", None)
    if sampler is not None and sampler.num_samples:
        return (
            int(sampler.peak_bf16_equivalent_bytes),
            int(sampler.peak_resident_bytes),
        )
    if quantizer is None:
        dense = _dense_kv_bytes(pipeline)
        return dense, dense
    equivalent, resident = _current_active_kv_bytes(pipeline, quantizer)
    return int(equivalent), int(resident)


def _sample_trace(device: torch.device, pipeline, quantizer, start_time: float, out_samples: List[Dict[str, float]]) -> None:
    bf16_kv_bytes, compressed_kv_bytes = _current_active_kv_bytes(pipeline, quantizer)
    out_samples.append(
        {
            "t_s": time.perf_counter() - start_time,
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "bf16_kv_bytes": bf16_kv_bytes,
            "compressed_kv_bytes": compressed_kv_bytes,
        }
    )


def collect_vram_trace(
    device: torch.device,
    pipeline,
    quantizer,
    interval_s: float,
    stop_event: threading.Event,
    out_samples: List[Dict[str, float]],
) -> None:
    start_time = time.perf_counter()
    _sample_trace(device, pipeline, quantizer, start_time, out_samples)
    while not stop_event.is_set():
        time.sleep(interval_s)
        _sample_trace(device, pipeline, quantizer, start_time, out_samples)
    _sample_trace(device, pipeline, quantizer, start_time, out_samples)


def downsample_trace(samples: List[Dict[str, float]], max_points: int) -> List[Dict[str, float]]:
    if max_points <= 0 or len(samples) <= max_points:
        return samples
    stride = int(math.ceil(len(samples) / max_points))
    reduced = samples[::stride]
    if reduced[-1]["t_s"] != samples[-1]["t_s"]:
        reduced.append(samples[-1])
    return reduced


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Self-Forcing generation.")

    device = torch.device(args.device)
    cache_policy = {"cadence": "per_step", "recent_blocks": 0}
    method_name, quantizer = parse_method(
        args.method,
        args.bits,
        args.block_size,
        kivi_residual_length=args.kivi_residual_length,
        quarot_channel_group_size=args.kv_channel_group_size,
        quarot_asym=args.kv_asym,
        quarot_clip_ratio=args.kv_clip_ratio,
    )

    if quantizer is not None and hasattr(quantizer, "set_timing_enabled"):
        quantizer.set_timing_enabled(args.profile_quant_timing)

    results_root = args.results_root if args.results_root.is_absolute() else (REPO_ROOT / args.results_root)
    output_dir = results_root / "videos" / method_name
    logs_dir = results_root / "logs"
    metrics_dir = results_root / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompt_path, args.max_prompts)
    if not prompts:
        raise RuntimeError(f"No prompts found in {args.prompt_path}")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "method": method_name,
                    "num_prompts": len(prompts),
                    "checkpoint": str(args.checkpoint_path),
                    "config": str(args.config_path),
                    "prompt_path": str(args.prompt_path),
                },
                indent=2,
            )
        )
        return

    set_seed(args.seed)
    low_memory = args.low_memory or get_cuda_free_memory_gb(device) < 40
    # Attend over the whole history, as the Tempokv and QVG runners do.  The
    # model's -1 default caps attention at 32760 tokens: the last 21 latent frames.
    local_attn_size = args.num_output_frames if args.local_attn_size is None else args.local_attn_size
    pipeline = initialize_pipeline(
        config_path=args.config_path,
        default_config_path=args.default_config_path,
        checkpoint_path=args.checkpoint_path,
        use_ema=args.use_ema,
        device=device,
        low_memory=low_memory,
        local_attn_size=local_attn_size,
    )
    if quantizer is not None and hasattr(quantizer, "set_runtime_context"):
        quantizer.set_runtime_context(frame_seq_length=int(getattr(pipeline, "frame_seq_length", 0)))
    num_frame_per_block = int(getattr(pipeline, "num_frame_per_block", 1))
    if args.num_output_frames % num_frame_per_block != 0:
        lower = args.num_output_frames - (args.num_output_frames % num_frame_per_block)
        upper = lower + num_frame_per_block
        raise ValueError(
            f"--num-output-frames={args.num_output_frames} must be divisible by num_frame_per_block={num_frame_per_block}. "
            f"Try {lower} or {upper}."
        )

    # Pre-initialize cache once so we can attach quantizer handles.
    pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    ensure_kv_cache_capacity(pipeline, args.num_output_frames, dtype=torch.bfloat16, device=device)

    # Resident-byte sampling rides on the quantizer because this repository has
    # no pipeline-level block hook: quantization happens inside the patched
    # attention layer.  Wrapping here, before reset_kv_state installs the
    # quantizer on every block, means every layer's call is seen.
    if quantizer is not None:
        quantizer = SamplingQuantizer(
            quantizer,
            cache_getter=lambda: getattr(pipeline, "kv_cache1", None) or [],
            num_layers=len(pipeline.kv_cache1 or []) or 30,
        )

    # VAE decode is the one latency segment this repository never measured.
    # decode_to_pixel is upstream Self-Forcing's own method, so the same wrap
    # target works in all three repositories.
    vae_decode = {"total_ms": 0.0, "calls": 0, "pending": []}
    _vae_type = type(pipeline.vae)
    _original_decode_to_pixel = _vae_type.decode_to_pixel

    def _timed_decode_to_pixel(self, *decode_args, **decode_kwargs):
        use_cuda_events = torch.cuda.is_available()
        if use_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            started = time.perf_counter()
        try:
            return _original_decode_to_pixel(self, *decode_args, **decode_kwargs)
        finally:
            if use_cuda_events:
                end_event.record()
                vae_decode["pending"].append((start_event, end_event))
            else:
                vae_decode["total_ms"] += (time.perf_counter() - started) * 1000.0
            vae_decode["calls"] += 1

    _vae_type.decode_to_pixel = _timed_decode_to_pixel
    if quantizer is not None:
        quantizer.reset_stats()
        num_layers = len(pipeline.kv_cache1)
        for block_idx, block in enumerate(pipeline.kv_cache1):
            block["kv_cache_size"] = int(block["k"].shape[1])
            block["batch_size"] = int(block["k"].shape[0])
            block["num_heads"] = int(block["k"].shape[2])
            block["head_dim"] = int(block["k"].shape[3])
            block["layer_id"] = int(block_idx)
            block["num_layers"] = int(num_layers)
            block["quantizer"] = quantizer
            block["quant_state"] = None
            block["quantize_cadence"] = cache_policy["cadence"]
            block["recent_blocks"] = int(cache_policy["recent_blocks"])
            block["frame_seq_length"] = int(getattr(pipeline, "frame_seq_length", 0))
            block["num_frame_per_block"] = int(num_frame_per_block)
            block["quantize_on_write"] = cache_policy["cadence"] == "per_step"
            block["recent_k"] = block["k"][:, :0].clone()
            block["recent_v"] = block["v"][:, :0].clone()
            block["recent_start_index"] = 0
            block["recent_end_index"] = 0
            # Keep quantized state as the primary cache representation.
            # This avoids persistent BF16 KV residency for quantized methods.
            block["k"] = torch.empty(0, dtype=torch.bfloat16, device=device)
            block["v"] = torch.empty(0, dtype=torch.bfloat16, device=device)

    if args.paper_latency_phase is not None:
        from paper_latency_runtime import run_paper_latency

        run_paper_latency(
            args,
            pipeline,
            quantizer,
            method_name,
            prompts,
            reset_kv_state,
            finalize_kv_state,
        )
        return

    run_log_path = logs_dir / f"generation_{method_name}.jsonl"
    run_log_f = run_log_path.open("a", encoding="utf-8")
    vram_trace_f = None
    if args.log_vram_trace:
        vram_trace_path = logs_dir / f"vram_trace_{method_name}.jsonl"
        vram_trace_f = vram_trace_path.open("a", encoding="utf-8")

    total_runtime_s = 0.0
    per_prompt_runtime_s: List[float] = []
    per_prompt_peak_bytes: List[int] = []
    per_prompt_reserved_bytes: List[int] = []
    peak_vram_bytes = 0
    first_video_shape = None

    try:
        for prompt_id, prompt in prompts:
            # Tempokv and QVG seed prompt i with seed + i * 1_000_003, so one prompt
            # index draws the same noise in all three repositories.
            prompt_seed = args.seed + prompt_id * 1_000_003
            set_seed(prompt_seed)
            reset_kv_state(pipeline, quantizer)

            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames, 16, 60, 104],
                device=device,
                dtype=torch.bfloat16,
            )

            torch.cuda.reset_peak_memory_stats(device)
            vram_samples: List[Dict[str, float]] = []
            trace_stop_event = threading.Event()
            trace_thread = None
            if args.log_vram_trace and args.vram_sample_interval_s > 0:
                trace_thread = threading.Thread(
                    target=collect_vram_trace,
                    args=(device, pipeline, quantizer, args.vram_sample_interval_s, trace_stop_event, vram_samples),
                    daemon=True,
                )
                trace_thread.start()
            start = time.perf_counter()
            with torch.no_grad():
                video, latents = pipeline.inference(
                    noise=sampled_noise,
                    text_prompts=[prompt] * args.num_samples,
                    return_latents=True,
                    low_memory=low_memory,
                )
            finalize_kv_state(pipeline, quantizer)
            runtime_s = time.perf_counter() - start
            if trace_thread is not None:
                trace_stop_event.set()
                trace_thread.join(timeout=5.0)
            peak = int(torch.cuda.max_memory_allocated(device))
            reserved_peak = int(torch.cuda.max_memory_reserved(device))
            if not vram_samples:
                vram_samples = [
                    {
                        "t_s": 0.0,
                        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                        "bf16_kv_bytes": 0,
                        "compressed_kv_bytes": 0,
                    }
                ]
            vram_samples = downsample_trace(vram_samples, args.vram_max_points)

            total_runtime_s += runtime_s
            per_prompt_runtime_s.append(float(runtime_s))
            # Peak stats are reset before every prompt above, so this is that
            # prompt's own peak rather than a running maximum.
            per_prompt_peak_bytes.append(int(peak))
            per_prompt_reserved_bytes.append(reserved_peak)
            peak_vram_bytes = max(peak_vram_bytes, peak)
            first_video_shape = tuple(video.shape)

            video_uint8 = (255.0 * rearrange(video, "b t c h w -> b t h w c")).clamp(0, 255).to(torch.uint8).cpu()
            h, w = tensor_shape_to_resolution(video)

            for sample_idx in range(args.num_samples):
                out_path = output_dir / f"prompt_{prompt_id:04d}_seed_{prompt_seed + sample_idx}.mp4"
                write_video(str(out_path), video_uint8[sample_idx], fps=args.fps)

                record = {
                    "method": method_name,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "seed": prompt_seed + sample_idx,
                    "model_config": str(args.config_path),
                    "checkpoint_path": str(args.checkpoint_path),
                    "git_commit_hash": git_commit_hash(),
                    "total_frames": int(video.shape[1]),
                    "resolution": [h, w],
                    "wall_clock_runtime_s": runtime_s,
                    "peak_vram_bytes": peak,
                    "output_video": str(out_path.relative_to(REPO_ROOT)),
                    "latents_shape": list(latents.shape),
                    "vram_trace_points": len(vram_samples),
                }
                run_log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                run_log_f.flush()
                if vram_trace_f is not None:
                    vram_trace_f.write(
                        json.dumps(
                            {
                                "method": method_name,
                                "prompt_id": prompt_id,
                                "seed": prompt_seed + sample_idx,
                                "runtime_s": runtime_s,
                                "peak_vram_bytes": peak,
                                "samples": vram_samples,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    vram_trace_f.flush()

    finally:
        run_log_f.close()
        if vram_trace_f is not None:
            vram_trace_f.close()

    if vae_decode["pending"]:
        torch.cuda.synchronize(device)
        vae_decode["total_ms"] += sum(
            start.elapsed_time(end) for start, end in vae_decode["pending"]
        )
        vae_decode["pending"].clear()

    # Both files report one definition, ``resident_analytic.v1``: the BF16 cost
    # of the tokens the cache actually holds against the bytes resident at the
    # same moment.  Mixing the preallocated capacity into the numerator and a
    # live byte walk into the denominator made the ratio depend on the cache
    # allocation rather than on the method.
    bf16_kv_bytes, compressed_kv_bytes = _resident_analytic_bytes(
        pipeline, quantizer
    )
    if quantizer is None:
        quant_time = 0.0
        dequant_time = 0.0
    else:
        quant_time = float(quantizer.stats.quantize_time_s)
        dequant_time = float(quantizer.stats.dequantize_time_s)

    efficiency = {
        "method": method_name,
        "num_prompts": len(prompts),
        "num_samples": args.num_samples,
        "total_runtime_s": total_runtime_s,
        "avg_runtime_s_per_prompt": total_runtime_s / max(len(prompts), 1),
        "peak_vram_bytes": peak_vram_bytes,
        "quantize_time_s": quant_time,
        "dequantize_time_s": dequant_time,
        "quantize_calls": 0.0 if quantizer is None else int(quantizer.stats.quantize_calls),
        "dequantize_calls": 0.0 if quantizer is None else int(quantizer.stats.dequantize_calls),
        "bf16_kv_bytes": bf16_kv_bytes,
        "compressed_kv_bytes": compressed_kv_bytes,
        "effective_kv_bits_per_value": (
            compressed_kv_bytes * 8 / max(bf16_kv_bytes / 2, 1)
            if compressed_kv_bytes
            else 16.0
        ),
        "compression_ratio": (bf16_kv_bytes / compressed_kv_bytes) if compressed_kv_bytes > 0 else 0.0,
        "first_video_shape": list(first_video_shape) if first_video_shape is not None else None,
        "cache_policy": cache_policy,
    }
    if quantizer is not None and hasattr(quantizer, "diagnostics"):
        extra_metrics = quantizer.diagnostics()
        if isinstance(extra_metrics, dict):
            efficiency.update(extra_metrics)

    metrics_path = metrics_dir / f"efficiency_{method_name}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(efficiency, f, indent=2)

    _write_shared_efficiency_record(
        Path(args.efficiency_output).expanduser()
        if args.efficiency_output
        else metrics_dir / "efficiency.json",
        args=args,
        method_name=method_name,
        prompts=prompts,
        quantizer=quantizer,
        pipeline=pipeline,
        per_prompt_runtime_s=per_prompt_runtime_s,
        per_prompt_peak_bytes=per_prompt_peak_bytes,
        per_prompt_reserved_bytes=per_prompt_reserved_bytes,
        vae_decode=vae_decode,
        device=device,
    )

    print(json.dumps(efficiency, indent=2))


def _write_shared_efficiency_record(
    path: Path,
    *,
    args: argparse.Namespace,
    method_name: str,
    prompts: List[Tuple[int, str]],
    quantizer,
    pipeline,
    per_prompt_runtime_s: List[float],
    per_prompt_peak_bytes: List[int],
    per_prompt_reserved_bytes: List[int],
    vae_decode: Dict[str, float],
    device: torch.device,
) -> None:
    """Emit the cross-repository ``efficiency.v1`` record.

    Written beside this repository's own ``efficiency_<method>.json``, which
    keeps its existing shape: that file answers questions specific to these
    baselines, this one is the row the shared collector reads.
    """
    import hashlib

    label, bits = _paper_method_label(method_name)
    prompt_indices = [prompt_id for prompt_id, _ in prompts]
    sampler = (
        quantizer.sampler
        if isinstance(quantizer, SamplingQuantizer)
        else efficiency_record.ResidentSampler()
    )
    quantized = quantizer is not None
    stats = quantizer.stats if quantized else None
    segments = {
        # Per-prompt wall clock is the only end-to-end figure this runner has;
        # the reported horizon-level number is the median prompt, matching the
        # generation_seconds_median the other two repositories report.
        "e2e": (
            statistics.median(per_prompt_runtime_s[1:]) * 1000.0
            if len(per_prompt_runtime_s) > 1
            else (per_prompt_runtime_s[0] * 1000.0 if per_prompt_runtime_s else None)
        ),
        "cache_encode": float(stats.quantize_time_s) * 1000.0 if quantized else 0.0,
        "cache_decode": float(stats.dequantize_time_s) * 1000.0 if quantized else 0.0,
        "vae_decode": vae_decode["total_ms"] if vae_decode["calls"] else None,
    }
    instrumentation = {
        "e2e": "patched" if per_prompt_runtime_s else "missing: no prompt completed",
        "cache_encode": "patched" if quantized else "absent: no codec",
        "cache_decode": "patched" if quantized else "absent: no codec",
        "vae_decode": (
            "patched" if vae_decode["calls"] else "missing: decode_to_pixel"
        ),
    }
    prompt_bytes = Path(args.prompt_path).expanduser().read_bytes()
    # Prompt 0 is the warm-up: peak stats are reset before every prompt, so the
    # reported peak is the largest of the prompts after it.
    measured_peaks = per_prompt_peak_bytes[1:] or per_prompt_peak_bytes
    efficiency_record.write_record(
        path,
        efficiency_record.build_efficiency_payload(
            repo="kv-quant-4-videogen",
            method=label,
            bits=bits,
            run={
                "model": "Self-Forcing",
                "width": 832,
                "height": 480,
                "frames": int(args.num_output_frames) * 4 - 3,
                "latent_frames": int(args.num_output_frames),
                "prompts_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                "prompt_indices": prompt_indices,
                "seed": int(args.seed),
                "num_gpus": 1,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_uuid": _efficiency_gpu_uuid(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda or "",
                "local_attn_size": int(
                    pipeline.generator.model.local_attn_size
                ),
            },
            sampler=sampler,
            cache_shape={
                "num_layers": 30,
                "num_heads": 12,
                "head_dim": 128,
                "frame_seq_length": 1560,
                "batch_size": 1,
            },
            fake_quant=False,
            predictor_parameter_bytes=0,
            gpu_memory={
                "peak_allocated_bytes": max(measured_peaks) if measured_peaks else 0,
                "peak_reserved_bytes": max(
                    per_prompt_reserved_bytes[1:] or per_prompt_reserved_bytes or [
                        int(torch.cuda.max_memory_reserved(device))
                    ]
                ),
                "peak_reset_after_prompt_index": (
                    prompt_indices[0] if prompt_indices else 0
                ),
            },
            per_prompt_generation_seconds=per_prompt_runtime_s,
            prompt_indices=prompt_indices,
            segments_ms=segments,
            instrumentation=instrumentation,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate videos with BF16 or KV quantization baselines.")
    parser.add_argument(
        "--method",
        type=str,
        default="BF16",
        help="BF16, RTN_INT4, RTN_INT2, KIVI_INT4, KIVI_INT2, QUAROT_KV_INT4, QUAROT_KV_INT2, HADAMARD_K_INT4",
    )
    parser.add_argument(
        "--bits", type=int, default=None, help="Bit width for a bare RTN/KIVI/QUAROT_KV method name"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help=(
            "Override every method's quantization group size.  Unset keeps each "
            "baseline at its published setting: RTN and QuaRot groups of 64, "
            "KIVI group 32 with a 128-token BF16 residual."
        ),
    )
    parser.add_argument(
        "--kv-channel-group-size",
        type=int,
        default=None,
        help=(
            "QuaRot only: quantization group in channels. The reference accepts "
            "-1 (token-wise) or head_dim; RTN/KIVI use --block-size instead"
        ),
    )
    parser.add_argument(
        "--kv-asym",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "QuaRot only: asymmetric KV quantization (--k_asym/--v_asym upstream). "
            "On by default, following the QuaRot paper's KV setting"
        ),
    )
    parser.add_argument(
        "--kv-clip-ratio",
        type=float,
        default=None,
        help=(
            "QuaRot only: shrink the quantization range (--k_clip_ratio upstream). "
            "0.95 by default, following the QuaRot paper"
        ),
    )
    parser.add_argument(
        "--kivi-residual-length",
        type=int,
        default=None,
        help="Recent BF16 token count kept by incremental KIVI; defaults to block size.",
    )
    parser.add_argument("--config-path", type=Path, default=SELF_FORCING_ROOT / "configs" / "self_forcing_dmd.yaml")
    parser.add_argument("--default-config-path", type=Path, default=SELF_FORCING_ROOT / "configs" / "default_config.yaml")
    parser.add_argument("--checkpoint-path", type=Path, default=REPO_ROOT / "checkpoints" / "self_forcing_dmd.pt")
    # MovieGen-128 at 180 latent frames (717 video frames) is the protocol the
    # Tempokv and QVG Self-Forcing runners share.
    parser.add_argument("--prompt-path", type=Path, default=REPO_ROOT / "prompts" / "moviegen_128.txt")
    parser.add_argument("--num-output-frames", type=int, default=180)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--local-attn-size",
        type=int,
        default=None,
        help="Causal attention window in latent frames (default: the full history, --num-output-frames).",
    )
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument(
        "--efficiency-output",
        type=Path,
        default=None,
        help="Optional path for the shared efficiency.v1 record.",
    )
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--low-memory", action="store_true", help="Enable official dynamic-swap low-memory mode.")
    parser.add_argument(
        "--profile-quant-timing",
        action="store_true",
        help="Record optional quantize/dequantize CUDA-event breakdowns",
    )
    parser.add_argument(
        "--log-vram-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-prompt VRAM usage trace logging to logs/vram_trace_<method>.jsonl",
    )
    parser.add_argument("--vram-sample-interval-s", type=float, default=0.2, help="VRAM trace sampling interval in seconds.")
    parser.add_argument("--vram-max-points", type=int, default=1000, help="Maximum stored points per prompt trace.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--paper-latency-phase",
        choices=("compilation", "pair"),
        default=None,
        help="Run the formal paired RTN latency protocol without writing video.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
