"""Formal paired latency runtime for Self-Forcing RTN baselines."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import time
import types

import torch


RUN_SCHEMA = "tempokv.latency-run.v1"
COMPILATION_SCHEMA = "tempokv.compilation-run.v1"
LIFECYCLE = (
    "self_forcing_global_causal_attention_full_resident_history_"
    "read_each_denoising_step"
)


def pixel_to_latent_frames(pixel_frames: int) -> int:
    if pixel_frames <= 0 or (pixel_frames + 3) % 4:
        raise ValueError(
            "Self-Forcing requires pixel frames satisfying (frames + 3) % 4 == 0"
        )
    return (pixel_frames + 3) // 4


def _write_json(path: str, payload: dict) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)


def _gpu_uuid(device: torch.device) -> str:
    properties = torch.cuda.get_device_properties(device)
    value = getattr(properties, "uuid", None)
    if value:
        return str(value)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    selected = visible if visible else str(device.index or 0)
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={selected}",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = completed.stdout.strip().splitlines()
    if len(values) != 1 or not values[0].startswith("GPU-"):
        raise RuntimeError("Could not observe one unambiguous GPU UUID")
    return values[0]


def observed_frozen(prompt: str, seed: int, device: torch.device) -> dict:
    causal_model = importlib.import_module("wan.modules.causal_model")
    if not callable(getattr(causal_model, "flex_attention", None)):
        raise RuntimeError("Self-Forcing causal model has no flex_attention backend")
    return {
        "model": "Self-Forcing",
        "width": 832,
        "height": 480,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "seed": int(seed),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_uuid": _gpu_uuid(device),
        "cuda_version": str(torch.version.cuda),
        "attention_backend": "torch_compiled_flex_attention",
        "attention_lifecycle": LIFECYCLE,
    }


class StrictSections:
    """Synchronize and accumulate every cache-codec and VAE segment."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cache_encode = 0.0
        self.cache_decode = 0.0
        self.vae_decode = 0.0
        self.cache_encode_calls = 0
        self.cache_decode_calls = 0
        self.vae_decode_calls = 0

    def reset(self) -> None:
        self.cache_encode = self.cache_decode = self.vae_decode = 0.0
        self.cache_encode_calls = 0
        self.cache_decode_calls = 0
        self.vae_decode_calls = 0

    def wrap(self, owner, method_name: str, bucket: str) -> bool:
        original = getattr(owner, method_name, None)
        if not callable(original):
            return False

        def measured(_owner, *args, **kwargs):
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            result = original(*args, **kwargs)
            torch.cuda.synchronize(self.device)
            setattr(self, bucket, getattr(self, bucket) + time.perf_counter() - started)
            calls = bucket + "_calls"
            setattr(self, calls, getattr(self, calls) + 1)
            return result

        setattr(owner, method_name, types.MethodType(measured, owner))
        return True


def make_run_report(
    *,
    method: str,
    phase: str,
    repeat: int,
    requested_frames: int,
    effective_frames: int,
    session_id: str,
    session_scope: str,
    runtime_instance_id: str,
    frozen: dict,
    timings: dict,
) -> dict:
    return {
        "schema_version": RUN_SCHEMA,
        "method": method,
        "phase": phase,
        "repeat": repeat,
        "requested_frames": requested_frames,
        "effective_frames": effective_frames,
        "session_id": session_id,
        "session_scope": session_scope,
        "runtime_instance_id": runtime_instance_id,
        "frozen": frozen,
        "timings_s": timings,
        "synchronization": {
            "api": "torch.cuda.synchronize",
            "before_each_segment": True,
            "after_each_segment": True,
        },
        "exclusions": {"model_loading": True, "compilation": True},
    }


def actual_local_attn_size(pipeline) -> int:
    value = getattr(pipeline, "local_attn_size", None)
    if value is None:
        generator = getattr(pipeline, "generator", None)
        model = getattr(generator, "model", None)
        value = getattr(model, "local_attn_size", -1)
    return int(value)


def run_paper_latency(args, pipeline, quantizer, method_name, prompts, reset_kv_state, finalize_kv_state) -> None:
    if method_name not in {"RTN_INT2", "RTN_INT4"} or quantizer is None:
        raise ValueError("Formal RTN latency supports RTN_INT2 and RTN_INT4 only")
    if len(prompts) != 1 or args.num_samples != 1:
        raise ValueError("Formal latency requires exactly one prompt and one sample")
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
        raise RuntimeError("Missing paper environment: " + ", ".join(missing))
    requested_frames = int(os.environ["PAPER_EXPERIMENT_FRAMES"])
    expected_latent = pixel_to_latent_frames(requested_frames)
    if args.num_output_frames != expected_latent:
        raise ValueError(
            f"{requested_frames} pixel frames require {expected_latent} latent "
            f"frames, got {args.num_output_frames}"
        )
    actual_window = actual_local_attn_size(pipeline)
    if actual_window != expected_latent:
        raise ValueError(
            f"Formal full-history protocol requires local_attn_size={expected_latent}, "
            f"runtime uses {actual_window}"
        )
    if os.environ["PAPER_EXPERIMENT_METHOD"] != method_name:
        raise ValueError("Harness method differs from Self-Forcing method")
    if os.environ["PAPER_EXPERIMENT_PHASE"] != args.paper_latency_phase:
        raise ValueError("Harness phase differs from Self-Forcing phase")
    if int(os.environ["PAPER_EXPERIMENT_SEED"]) != int(args.seed):
        raise ValueError("Harness seed differs from Self-Forcing seed")

    device = torch.device(args.device)
    _prompt_id, prompt = prompts[0]
    frozen = observed_frozen(prompt, args.seed, device)
    expected_frozen = os.environ["PAPER_EXPERIMENT_FROZEN_JSON"]
    if json.loads(expected_frozen) != frozen:
        raise ValueError("Observed prompt/hardware/backend differs from manifest")

    sections = StrictSections(device)
    encode_wrapped = [
        name
        for name in ("append_kv", "finalize_state", "quantize_kv")
        if sections.wrap(quantizer, name, "cache_encode")
    ]
    decode_wrapped = [
        name
        for name in ("materialize_kv", "dequantize_kv")
        if sections.wrap(quantizer, name, "cache_decode")
    ]
    if not encode_wrapped or not decode_wrapped:
        raise RuntimeError(
            "RTN runtime exposes no instrumentable encode/decode API: "
            f"encode={encode_wrapped}, decode={decode_wrapped}"
        )
    if not sections.wrap(pipeline.vae, "decode_to_pixel", "vae_decode"):
        raise RuntimeError("Self-Forcing VAE exposes no decode_to_pixel method")

    torch.manual_seed(args.seed)
    sampled_noise = torch.randn(
        [1, args.num_output_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16,
    )

    def one_run() -> tuple[dict[str, float], int]:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        reset_kv_state(pipeline, quantizer)
        quantizer.reset_stats()
        sections.reset()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            video, _ = pipeline.inference(
                noise=sampled_noise,
                text_prompts=[prompt],
                return_latents=True,
                low_memory=args.low_memory,
            )
        finalize_kv_state(pipeline, quantizer)
        torch.cuda.synchronize(device)
        e2e = time.perf_counter() - started
        accounted = sections.cache_encode + sections.cache_decode + sections.vae_decode
        if sections.cache_encode_calls <= 0 or sections.cache_decode_calls <= 0:
            raise RuntimeError(
                "RTN cache instrumentation recorded no encode/decode calls: "
                f"encode={sections.cache_encode_calls}, "
                f"decode={sections.cache_decode_calls}"
            )
        if sections.vae_decode_calls != 1:
            raise RuntimeError(
                f"Expected one VAE decode, observed {sections.vae_decode_calls}"
            )
        tolerance = max(0.005, e2e * 0.01)
        if accounted > e2e + tolerance:
            raise RuntimeError("Timing sections exceed E2E")
        return {
            "e2e": e2e,
            "cache_encode": sections.cache_encode,
            "cache_decode": sections.cache_decode,
            "vae_decode": sections.vae_decode,
            "denoising_other": max(0.0, e2e - accounted),
        }, int(video.shape[1])

    runtime_id = os.environ["PAPER_EXPERIMENT_RUNTIME_INSTANCE_ID"]
    first, first_frames = one_run()
    second, second_frames = one_run()
    if first_frames != requested_frames or second_frames != requested_frames:
        raise RuntimeError(
            f"Runtime produced {first_frames}/{second_frames} frames, requested {requested_frames}"
        )

    if args.paper_latency_phase == "compilation":
        report_path = os.environ.get("PAPER_EXPERIMENT_REPORT")
        if not report_path:
            raise RuntimeError("Compilation phase requires PAPER_EXPERIMENT_REPORT")
        _write_json(
            report_path,
            {
                "schema_version": COMPILATION_SCHEMA,
                "method": method_name,
                "one_time_compilation_s": max(0.0, first["e2e"] - second["e2e"]),
                "session_id": os.environ["PAPER_EXPERIMENT_SESSION_ID"],
                "session_scope": os.environ["PAPER_EXPERIMENT_SESSION_SCOPE"],
                "frozen": frozen,
            },
        )
        return

    warmup_path = os.environ.get("PAPER_EXPERIMENT_WARMUP_REPORT")
    measure_path = os.environ.get("PAPER_EXPERIMENT_MEASURE_REPORT")
    if not warmup_path or not measure_path:
        raise RuntimeError("Pair phase requires warmup and measure report paths")
    common = {
        "method": method_name,
        "repeat": int(os.environ["PAPER_EXPERIMENT_REPEAT"]),
        "requested_frames": requested_frames,
        "effective_frames": second_frames,
        "session_id": os.environ["PAPER_EXPERIMENT_SESSION_ID"],
        "session_scope": os.environ["PAPER_EXPERIMENT_SESSION_SCOPE"],
        "runtime_instance_id": runtime_id,
        "frozen": frozen,
    }
    _write_json(warmup_path, make_run_report(phase="warmup", timings=first, **common))
    _write_json(measure_path, make_run_report(phase="measure", timings=second, **common))
