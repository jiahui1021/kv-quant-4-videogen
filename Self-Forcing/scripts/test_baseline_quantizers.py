#!/usr/bin/env python3
"""Static RTN/KIVI/Hadamard-K checks; this script never loads a video model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from kv_quant.bitpack import unpack_bits, unpack_bits_rows
from kv_quant.factory import create_quantizer


def _append_meta(start: int, length: int, dtype: torch.dtype, attention_space: bool = False) -> dict:
    return {
        "absolute_start": start,
        "absolute_end": start + length,
        "tensor_dtype": dtype,
        "already_rotated": attention_space,
        "attention_space": attention_space,
    }


def _payload_bytes(state: dict) -> int:
    total = 0
    for key in ("k", "v"):
        payload = state[key]["q"]
        total += int(payload.numel() * payload.element_size())
    return total


def _codebook(tensor_state: dict) -> list[int]:
    if tensor_state.get("packed_rows") is not None:
        values = unpack_bits_rows(
            tensor_state["q"],
            int(tensor_state["bits"]),
            tensor_state["q_shape"],
            signed=bool(tensor_state.get("signed", False)),
            row_dims=int(tensor_state["packed_rows"]),
        )
    else:
        values = unpack_bits(
            tensor_state["q"],
            int(tensor_state["bits"]),
            tensor_state["q_shape"],
            int(tensor_state["q_numel"]),
            signed=bool(tensor_state.get("signed", False)),
        )
    return [int(value) for value in torch.unique(values).cpu().tolist()]


def _append_invariance(method: str, bits: int, block_size: int, seed: int) -> dict:
    torch.manual_seed(seed)
    attention_space = method == "HADAMARD_K"
    quantizer = create_quantizer(
        method,
        bits=bits,
        block_size=block_size,
        residual_length=block_size if method == "KIVI" else None,
    )
    state = quantizer.init_state(
        {
            "shape": (1, 0, 3, 16),
            "tensor_dtype": torch.bfloat16,
            "device": "cpu",
            "attention_space": attention_space,
        }
    )
    saved = None
    for index in range(4):
        k = torch.randn(1, block_size, 3, 16, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        quantizer.append_kv(
            state,
            k,
            v,
            _append_meta(index * block_size, block_size, k.dtype, attention_space),
        )
        if saved is None and state["segments"]:
            saved = state["segments"][0]["k"]["q"].clone()
    if saved is None:
        raise AssertionError(f"{method} did not produce a quantized segment")
    if not torch.equal(saved, state["segments"][0]["k"]["q"]):
        raise AssertionError(f"{method} changed an old packed payload after append")
    k_hat, v_hat = quantizer.materialize_kv(
        state,
        {"tensor_dtype": torch.bfloat16, "attention_space": attention_space},
    )
    return {
        "method": method,
        "append_tokens": int(state["num_tokens"]),
        "quantized_tokens": int(state["quantized_tokens"]),
        "segments": len(state["segments"]),
        "materialized_shape": list(k_hat.shape),
        "materialized_v_shape": list(v_hat.shape),
        "resident_bytes": int(quantizer.memory_bytes(state)),
    }


def _reconstruction(method: str, bits: int, block_size: int, k: torch.Tensor, v: torch.Tensor) -> dict:
    quantizer = create_quantizer(
        method,
        bits=bits,
        block_size=block_size,
        residual_length=block_size if method == "KIVI" else None,
    )
    state = quantizer.quantize_kv(k, v, meta={"tensor_dtype": k.dtype})
    k_hat, v_hat = quantizer.dequantize_kv(state)
    denom = max(float(k.numel() + v.numel()), 1.0)
    return {
        "mse_k": float((k.float() - k_hat.float()).square().mean()),
        "mse_v": float((v.float() - v_hat.float()).square().mean()),
        "mae": float((k.float() - k_hat.float()).abs().sum() + (v.float() - v_hat.float()).abs().sum()) / denom,
        "cosine_k": float(torch.nn.functional.cosine_similarity(k.float().flatten(), k_hat.float().flatten(), dim=0)),
        "cosine_v": float(torch.nn.functional.cosine_similarity(v.float().flatten(), v_hat.float().flatten(), dim=0)),
        "max_error": float(max((k.float() - k_hat.float()).abs().max(), (v.float() - v_hat.float()).abs().max())),
        "payload_bytes": _payload_bytes(state),
        "codebook_k": _codebook(state["k"]),
        "codebook_v": _codebook(state["v"]),
        "resident_bytes": int(quantizer.memory_bytes(state)),
        "effective_bits_per_value": float(quantizer.memory_bytes(state) * 8 / denom),
    }


def _kivi_regressions() -> dict:
    quantizer = create_quantizer(
        "KIVI",
        bits=4,
        block_size=4,
        residual_length=8,
        value_group_size=4,
    )
    k = torch.randn(1, 12, 1, 4, dtype=torch.bfloat16)
    state = quantizer.quantize_kv(k, k.clone())
    expected = {
        "quantized_k_tokens": 8,
        "residual_k_tokens": 4,
        "quantized_v_tokens": 4,
        "residual_v_tokens": 8,
    }
    actual = {
        "quantized_k_tokens": int(state["quantized_k_tokens"]),
        "residual_k_tokens": int(state["residual_k"].shape[1]),
        "quantized_v_tokens": int(state["quantized_v_tokens"]),
        "residual_v_tokens": int(state["residual_v"].shape[1]),
    }
    if actual != expected:
        raise AssertionError(f"KIVI residual lifecycle mismatch: {actual} != {expected}")
    estimate = quantizer.estimate_active_kv_bytes(12, 1, 1, 4)
    resident = quantizer.memory_bytes(state)
    if estimate != resident:
        raise AssertionError(f"KIVI byte estimate mismatch: {estimate} != {resident}")

    eviction_quantizer = create_quantizer(
        "KIVI",
        bits=4,
        block_size=3,
        residual_length=0,
        value_group_size=3,
    )
    eviction_input = torch.randn(1, 6, 1, 3, dtype=torch.bfloat16)
    eviction_state = eviction_quantizer.quantize_kv(
        eviction_input,
        eviction_input.clone(),
    )
    removed = eviction_quantizer.evict_prefix(eviction_state, 3)
    if removed != 3 or eviction_state["num_tokens"] != 3:
        raise AssertionError(
            f"KIVI group eviction failed: removed={removed}, "
            f"remaining={eviction_state['num_tokens']}"
        )

    high = torch.tensor(
        [1e5, 1.2e5, 1.5e5, 2e5],
        dtype=torch.bfloat16,
    ).reshape(1, 4, 1, 1).repeat(1, 1, 1, 4)
    overflow_quantizer = create_quantizer(
        "KIVI",
        bits=4,
        block_size=4,
        residual_length=0,
        value_group_size=4,
    )
    high_state = overflow_quantizer.quantize_kv(high, high.clone())
    high_hat, _ = overflow_quantizer.dequantize_kv(high_state)
    if not torch.isfinite(high_hat).all():
        raise AssertionError("KIVI finite BF16 input reconstructed as inf/nan")

    return {
        **actual,
        "resident_bytes": int(resident),
        "estimated_bytes": int(estimate),
        "evicted_tokens": int(removed),
        "remaining_tokens": int(eviction_state["num_tokens"]),
        "finite_bf16_range": True,
    }


def _self_forcing_patch_regression() -> dict:
    patch_path = (
        REPO_ROOT
        / "Self-Forcing"
        / "docs"
        / "patches"
        / "self_forcing_kv_quant.patch"
    )
    text = patch_path.read_text(encoding="utf-8")
    required = (
        'cache_k[:, :local_end_index]',
        'cache_v[:, :local_end_index]',
        'active_k, active_v = quantizer.dequantize_kv',
        'cache_k = torch.zeros(',
    )
    missing = [snippet for snippet in required if snippet not in text]
    if missing:
        raise AssertionError(
            "Self-Forcing patch does not preserve active-prefix KIVI semantics: "
            + ", ".join(missing)
        )
    return {
        "compresses_active_prefix_only": True,
        "restores_dense_work_capacity": True,
    }


def run(seed: int, block_size: int) -> dict:
    torch.manual_seed(seed)
    k = torch.randn(2, 32, 3, 16, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    methods = ("RTN", "KIVI", "HADAMARD_K")
    result = {"append_invariance": [], "reconstruction": {}}
    for method in methods:
        result["append_invariance"].append(_append_invariance(method, 4, block_size, seed + 1))
        rows = {}
        for bits in (2, 4):
            rows[str(bits)] = _reconstruction(method, bits, block_size, k, v)
        if rows["4"]["mse_k"] >= rows["2"]["mse_k"] or rows["4"]["mse_v"] >= rows["2"]["mse_v"]:
            raise AssertionError(f"INT4 must reconstruct better than INT2 for {method}: {rows}")
        result["reconstruction"][method] = rows

    quantizer = create_quantizer("HADAMARD_K", bits=4, block_size=block_size)
    q = torch.randn(1, 5, 3, 16, dtype=torch.float32)
    key = torch.randn_like(q)
    rotated_q, rotated_key = quantizer.prepare_attention_qk(q, key)
    original = torch.matmul(q, key.transpose(-1, -2))
    rotated = torch.matmul(rotated_q, rotated_key.transpose(-1, -2))
    max_abs_error = float((original - rotated).abs().max())
    if max_abs_error >= 1e-5:
        raise AssertionError(f"Hadamard-K BF16 rotation equivalence failed: max_abs_error={max_abs_error}")
    result["hadamard_k_rotation_equivalence"] = {
        "max_abs_error": max_abs_error,
        "cosine": float(torch.nn.functional.cosine_similarity(original.flatten(), rotated.flatten(), dim=0)),
    }
    result["kivi_regressions"] = _kivi_regressions()
    result["self_forcing_patch_regression"] = _self_forcing_patch_regression()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--block-size", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.block_size), indent=2))


if __name__ == "__main__":
    main()
