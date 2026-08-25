"""Resident KV memory accounting for the integrated Forcing-KV runtime.

Forcing-KV compresses by assigning different head groups different resident
history lengths. The ratio below therefore comes from the tensors that are
actually alive in ``pipeline.kv_cache1`` and a BF16 full-head equivalent for
the same logical resident token count.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

import torch


def _is_kv_name(name: object) -> bool:
    if not isinstance(name, str):
        return False
    lowered = name.lower()
    return lowered in {"k", "v"} or lowered.endswith(("_k", "_v"))


def _tensor_bytes(node: Any, *, key: str | None, seen: set[tuple[int, int]]) -> int:
    if isinstance(node, torch.Tensor):
        if key is not None and not _is_kv_name(key):
            return 0
        storage = node.untyped_storage()
        identity = (int(storage.data_ptr()), int(storage.nbytes()))
        if identity in seen:
            return 0
        seen.add(identity)
        return identity[1]
    if isinstance(node, dict):
        return sum(
            _tensor_bytes(value, key=name, seen=seen)
            for name, value in node.items()
        )
    if isinstance(node, (list, tuple)):
        return sum(_tensor_bytes(value, key=key, seen=seen) for value in node)
    return 0


def _scalar(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value or 0)


def _geometry(cache: dict[str, Any]) -> tuple[int, int, int]:
    for name, value in cache.items():
        if _is_kv_name(name) and isinstance(value, torch.Tensor) and value.ndim == 4:
            return int(value.shape[0]), int(value.shape[2]), int(value.shape[3])
    return (
        int(cache.get("batch_size", 1)),
        int(cache.get("num_heads", 12)),
        int(cache.get("head_dim", 128)),
    )


def measure_cache(
    kv_cache: list[dict[str, Any]],
    *,
    default_num_heads: int = 12,
    default_head_dim: int = 128,
) -> dict[str, Any]:
    """Measure resident bytes and a full-head BF16 equivalent."""
    if not kv_cache:
        return {
            "num_layers": 0,
            "resident_kv_bytes": 0,
            "bf16_equivalent_bytes": 0,
            "compression_ratio": None,
        }

    seen: set[tuple[int, int]] = set()
    resident_bytes = _tensor_bytes(kv_cache, key=None, seen=seen)
    equivalent_bytes = 0
    for cache in kv_cache:
        batch, heads, head_dim = _geometry(cache)
        heads = heads or default_num_heads
        head_dim = head_dim or default_head_dim
        # local_end_index is the number of resident token positions after a
        # local/history policy. Fall back to the absolute cursor for caches
        # that keep the complete sequence.
        tokens = _scalar(cache.get("local_end_index"))
        if tokens <= 0:
            tokens = _scalar(cache.get("global_end_index"))
        equivalent_bytes += batch * tokens * heads * head_dim * 2 * 2

    return {
        "num_layers": len(kv_cache),
        "resident_kv_bytes": int(resident_bytes),
        "bf16_equivalent_bytes": int(equivalent_bytes),
        "compression_ratio": (
            float(equivalent_bytes / resident_bytes) if resident_bytes else None
        ),
    }


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


__all__ = ["measure_cache", "write_report"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(
        f"{report.get('workload', 'Forcing-KV')}: "
        f"compression_ratio={report.get('compression_ratio')} "
        f"({report.get('bf16_equivalent_bytes')} / "
        f"{report.get('resident_kv_bytes')} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
