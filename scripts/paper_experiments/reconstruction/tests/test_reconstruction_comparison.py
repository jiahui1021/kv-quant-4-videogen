from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


RECONSTRUCTION = Path(__file__).resolve().parents[1]
if str(RECONSTRUCTION) not in sys.path:
    sys.path.insert(0, str(RECONSTRUCTION))


def _module(name: str):
    path = RECONSTRUCTION / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"reconstruction_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


methods = _module("methods")
plot_figure = _module("plot_figure")
run_reconstruction = _module("run_reconstruction")


def _trace(path: Path, *, frames: int, layers: int = 2, records: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    shape = (layers, records, 2, 128)
    keys = np.zeros((layers, records, frames, 2, 128), dtype=np.float32)
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
    np.savez(
        path,
        keys=keys,
        values=values,
        layer_ids=np.arange(layers, dtype=np.int32),
        trace_stride=np.asarray(4, dtype=np.int32),
        coordinate_system=np.asarray("pre_rope_normalized"),
    )
    return path


# --------------------------------------------------------------------------
# The registry covers every method family the comparison claims
# --------------------------------------------------------------------------


def test_every_named_family_is_registered() -> None:
    families = {item.family for item in methods.METHODS}
    assert families == {
        "bf16",
        "rtn",
        "kivi",
        "quarot",
        "hadamard",
        "qvg",
        "tempokv",
    }


def test_both_bit_widths_exist_for_every_quantized_family() -> None:
    for family in ("rtn", "kivi", "quarot", "hadamard", "qvg", "tempokv"):
        bits = {item.bits for item in methods.METHODS if item.family == family}
        assert bits == {2, 4}, family


def test_only_tempokv_declares_a_calibration_requirement() -> None:
    """It is the one method that fits anything, and that asymmetry is reported."""
    fitting = {item.family for item in methods.METHODS if item.needs_calibration}
    assert fitting == {"tempokv"}


def test_unknown_methods_name_what_is_available() -> None:
    with pytest.raises(ValueError, match="kivi_int2"):
        methods.method("awq_int2")


# --------------------------------------------------------------------------
# BF16 is the floor, not a no-op
# --------------------------------------------------------------------------


def test_bf16_reports_the_storage_floor_rather_than_zero() -> None:
    torch.manual_seed(0)
    keys = torch.randn(2, 3, 8, 2, 128)
    values = torch.randn_like(keys)
    run = methods.method("bf16").build(methods.MethodContext(stride=4))
    rebuilt_keys, rebuilt_values = run(keys, values)
    curve = methods.frame_curve(keys, rebuilt_keys)
    assert curve["pooled_nmse"] > 0.0
    assert curve["pooled_nmse"] < 1e-4
    assert rebuilt_values.shape == values.shape


# --------------------------------------------------------------------------
# The kv_quant baselines run on the trace layout unchanged
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["rtn_int2", "kivi_int2", "quarot_int2", "hadamard_int2"])
def test_baselines_consume_one_trace_layer_without_a_permute(name: str) -> None:
    """A layer of the trace is already [B, L, H, D]; a permute here would be a bug."""
    torch.manual_seed(0)
    keys = torch.randn(2, 3, 16, 2, 128)
    values = torch.randn_like(keys)
    run = methods.method(name).build(methods.MethodContext(stride=4))
    rebuilt_keys, rebuilt_values = run(keys, values)
    assert rebuilt_keys.shape == keys.shape
    assert rebuilt_values.shape == values.shape
    # Each rebuilt tensor must track its own reference, never the other one.
    def error(rebuilt, reference):
        return float((rebuilt.double() - reference.double()).square().sum())

    assert error(rebuilt_keys, keys) < error(rebuilt_keys, values)
    assert error(rebuilt_values, values) < error(rebuilt_values, keys)


# --------------------------------------------------------------------------
# A method that cannot run says so
# --------------------------------------------------------------------------


def test_tempokv_without_a_calibration_trace_declines_with_a_reason() -> None:
    """Scoring it on the trace it was fitted on would flatter it."""
    context = methods.MethodContext(stride=4, calibration=None)
    with pytest.raises(methods.MethodUnavailable, match="calibration-trace"):
        methods.method("tempokv_int2").build(context)


def test_tempokv_without_a_checkout_declines_with_a_reason(tmp_path: Path) -> None:
    context = methods.MethodContext(stride=4, tempokv_root=tmp_path / "absent")
    with pytest.raises(methods.MethodUnavailable, match="tempokv-root"):
        methods.method("tempokv_int2").build(context)


def test_an_unavailable_method_is_recorded_not_dropped(tmp_path: Path) -> None:
    """An absent row would read as an untested method."""
    trace = _trace(tmp_path / "trace.npz", frames=8)
    args = run_reconstruction.build_parser().parse_args(
        [
            "--trace", str(trace),
            "--output-dir", str(tmp_path / "out"),
            "--methods", "bf16", "tempokv_int2",
        ]
    )
    payload = run_reconstruction.run(args)
    by_name = {item["method"]: item for item in payload["methods"]}
    assert set(by_name) == {"bf16", "tempokv_int2"}
    assert by_name["bf16"]["status"] == "complete"
    assert by_name["tempokv_int2"]["status"] == "unavailable"
    assert "calibration" in by_name["tempokv_int2"]["reason"]


def test_the_figure_names_methods_that_did_not_run() -> None:
    rows = plot_figure.summary_rows(
        {
            "methods": [
                {
                    "method": "qvg_int2",
                    "family": "qvg",
                    "bits": 2,
                    "status": "unavailable",
                    "reason": "needs Triton",
                }
            ]
        }
    )
    assert rows[0]["status"] == "unavailable"
    assert rows[0]["reason"] == "needs Triton"


# --------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------


def _curve(errors: list[float]) -> dict:
    clean = torch.ones(1, 1, len(errors), 1, 1)
    rebuilt = clean.clone()
    for index, value in enumerate(errors):
        rebuilt[0, 0, index, 0, 0] = 1.0 + value
    return methods.frame_curve(clean, rebuilt)


def test_the_cumulative_curve_ends_at_the_pooled_number() -> None:
    curve = _curve([0.1, 0.2, 0.3, 0.4])
    assert curve["cumulative"][-1]["nmse"] == pytest.approx(curve["pooled_nmse"])


def test_the_head_window_is_everything_before_the_tail() -> None:
    summary = methods.summarize_curve(_curve([0.1] * 6 + [1.0] * 2), tail_fraction=0.25)
    assert summary["tail_start_frame"] == 6
    assert summary["head_mean_nmse"] == pytest.approx(0.01, rel=1e-6)


def test_a_late_degrading_method_is_distinguished_from_a_uniform_one() -> None:
    late = methods.summarize_curve(_curve([0.1] * 6 + [1.0] * 2), tail_fraction=0.25)
    uniform = methods.summarize_curve(_curve([0.25] * 8), tail_fraction=0.25)
    assert uniform["tail_over_head"] == pytest.approx(1.0)
    assert late["tail_over_head"] > 5.0


# --------------------------------------------------------------------------
# Trace loading
# --------------------------------------------------------------------------


def test_max_records_trims_records_and_never_frames(tmp_path: Path) -> None:
    trace = _trace(tmp_path / "trace.npz", frames=20, records=8)
    keys, _, _ = run_reconstruction.load_trace(trace, max_records=3)
    assert keys.shape[1] == 3
    assert keys.shape[2] == 20


def test_a_calibration_trace_from_another_run_is_refused(tmp_path: Path) -> None:
    trace = _trace(tmp_path / "trace.npz", frames=10)
    calibration = tmp_path / "calibration.npz"
    rng = np.random.default_rng(1)
    shape = (2, 3, 6, 2, 128)
    np.savez(
        calibration,
        keys=rng.normal(size=shape).astype(np.float32),
        values=rng.normal(size=shape).astype(np.float32),
        layer_ids=np.arange(2, dtype=np.int32),
        trace_stride=np.asarray(8, dtype=np.int32),
    )
    args = run_reconstruction.build_parser().parse_args(
        [
            "--trace", str(trace),
            "--calibration-trace", str(calibration),
            "--output-dir", str(tmp_path / "out"),
            "--methods", "bf16",
        ]
    )
    with pytest.raises(ValueError, match="different runs"):
        run_reconstruction.run(args)
