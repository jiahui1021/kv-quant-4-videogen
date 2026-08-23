from __future__ import annotations

import importlib.util
import argparse
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "paper_latency_runtime.py"
SPEC = importlib.util.spec_from_file_location("paper_latency_runtime", MODULE_PATH)
runtime = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runtime)

RUNNER_PATH = ROOT.parent / "scripts/paper_experiments/latency/run_rtn.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_rtn", RUNNER_PATH)
runner = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(runner)


def test_pixel_to_latent_frames() -> None:
    assert runtime.pixel_to_latent_frames(717) == 180
    assert runtime.pixel_to_latent_frames(1401) == 351


@pytest.mark.parametrize("frames", [0, 180, 360, 540, 1050, 1400])
def test_rejects_inexact_pixel_horizons(frames: int) -> None:
    with pytest.raises(ValueError):
        runtime.pixel_to_latent_frames(frames)


def test_report_contract_has_complete_timing_split() -> None:
    frozen = {
        "model": "Self-Forcing",
        "width": 832,
        "height": 480,
        "prompt_sha256": "a" * 64,
        "seed": 0,
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "gpu_uuid": "GPU-test",
        "cuda_version": "12.1",
        "attention_backend": "torch_compiled_flex_attention",
        "attention_lifecycle": runtime.LIFECYCLE,
    }
    timings = {
        "e2e": 10.0,
        "cache_encode": 1.0,
        "cache_decode": 2.0,
        "vae_decode": 3.0,
        "denoising_other": 4.0,
    }
    report = runtime.make_run_report(
        method="RTN_INT2",
        phase="measure",
        repeat=0,
        requested_frames=717,
        effective_frames=717,
        session_id="session",
        session_scope="coordinator_session_multiple_processes",
        runtime_instance_id="runtime",
        frozen=frozen,
        timings=timings,
    )
    assert set(report["timings_s"]) == {
        "e2e",
        "cache_encode",
        "cache_decode",
        "vae_decode",
        "denoising_other",
    }
    assert report["synchronization"]["before_each_segment"] is True
    assert report["synchronization"]["after_each_segment"] is True


def test_strict_sections_require_real_method_calls(monkeypatch) -> None:
    monkeypatch.setattr(runtime.torch.cuda, "synchronize", lambda _device: None)

    class Codec:
        def append_kv(self, value):
            return value + 1

        def materialize_kv(self, value):
            return value - 1

    codec = Codec()
    sections = runtime.StrictSections(runtime.torch.device("cpu"))
    assert sections.wrap(codec, "append_kv", "cache_encode") is True
    assert sections.wrap(codec, "materialize_kv", "cache_decode") is True
    assert sections.wrap(codec, "missing", "cache_decode") is False
    assert codec.append_kv(2) == 3
    assert codec.materialize_kv(2) == 1
    assert sections.cache_encode_calls == 1
    assert sections.cache_decode_calls == 1
    assert sections.cache_encode > 0
    assert sections.cache_decode > 0


def test_runner_calls_self_forcing_with_latent_length(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PAPER_EXPERIMENT_SEED", "23")
    args = argparse.Namespace(
        phase="pair",
        method="RTN_INT4",
        frames=717,
        config_path=tmp_path / "config.yaml",
        default_config_path=tmp_path / "default.yaml",
        checkpoint_path=tmp_path / "model.pt",
        prompt_path=tmp_path / "prompt.txt",
        output_dir=tmp_path / "output",
        block_size=16,
        python="python",
        use_ema=False,
        low_memory=False,
    )
    command = runner.command(args)
    assert "Self-Forcing/scripts/01_generate.py" in command[1]
    assert command[command.index("--num-output-frames") + 1] == "180"
    assert command[command.index("--local-attn-size") + 1] == "180"
    assert command[command.index("--method") + 1] == "RTN_INT4"
    assert command[command.index("--seed") + 1] == "23"


def test_pipeline_window_read_does_not_require_generator() -> None:
    class Pipeline:
        local_attn_size = 180

    assert runtime.actual_local_attn_size(Pipeline()) == 180
