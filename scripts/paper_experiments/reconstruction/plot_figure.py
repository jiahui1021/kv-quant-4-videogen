#!/usr/bin/env python3
"""Render the cross-method reconstruction figure and its backing CSVs.

    (a) K NMSE per frame      (b) V NMSE per frame
    (c) K NMSE cumulative     (d) V NMSE cumulative

Methods that could not run are listed under the figure rather than omitted, so
a missing method is visibly missing instead of quietly absent.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


# Cycled so methods whose curves coincide stay distinguishable.
_DASHES = ((), (6, 2), (2, 2), (6, 2, 1, 2), (1, 1.6), (10, 3))


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid report {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"report must be a JSON object: {path}")
    return payload


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _complete(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in report["methods"] if item["status"] == "complete"]


def curve_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in _complete(report):
        for tensor in ("k", "v"):
            cumulative = {row["frame"]: row["nmse"] for row in item[tensor]["cumulative"]}
            for row in item[tensor]["per_frame"]:
                rows.append(
                    {
                        "method": item["method"],
                        "family": item["family"],
                        "bits": item["bits"],
                        "tensor": tensor.upper(),
                        "frame": row["frame"],
                        "nmse": row["nmse"],
                        "cumulative_nmse": cumulative.get(row["frame"]),
                    }
                )
    return rows


def summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in report["methods"]:
        if item["status"] != "complete":
            rows.append(
                {
                    "method": item["method"],
                    "family": item["family"],
                    "bits": item["bits"],
                    "tensor": "",
                    "status": item["status"],
                    "reason": item["reason"],
                }
            )
            continue
        for tensor in ("k", "v"):
            rows.append(
                {
                    "method": item["method"],
                    "family": item["family"],
                    "bits": item["bits"],
                    "tensor": tensor.upper(),
                    "status": "complete",
                    "reason": "",
                    **item["summary"][tensor],
                }
            )
    return rows


def _panel(axis, report: dict[str, Any], tensor: str, *, cumulative: bool, letter: str) -> None:
    for index, item in enumerate(_complete(report)):
        source = item[tensor]["cumulative" if cumulative else "per_frame"]
        axis.plot(
            [row["frame"] for row in source],
            [row["nmse"] for row in source],
            label=item["label"],
            linewidth=1.6,
            dashes=_DASHES[index % len(_DASHES)],
            alpha=0.9,
        )
    axis.set_xlabel("latent frame index")
    axis.set_ylabel("normalized reconstruction MSE")
    kind = "cumulative" if cumulative else "per frame"
    axis.set_title(f"({letter}) {tensor.upper()}, {kind}")
    axis.set_yscale("log")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)


def render(
    report: dict[str, Any], output: Path, *, formats: Sequence[str] = ("pdf", "png")
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on the host env
        raise RuntimeError("plotting requires matplotlib") from error

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    _panel(axes[0][0], report, "k", cumulative=False, letter="a")
    _panel(axes[0][1], report, "v", cumulative=False, letter="b")
    _panel(axes[1][0], report, "k", cumulative=True, letter="c")
    _panel(axes[1][1], report, "v", cumulative=True, letter="d")
    title = (
        f"KV reconstruction error over {report['frames']} latent frames "
        f"({report['model']})"
    )
    skipped = [
        f"{item['label']} ({item['status']})"
        for item in report["methods"]
        if item["status"] != "complete"
    ]
    if skipped:
        # Name what is missing on the figure itself; an absent curve otherwise
        # reads as an untested method.
        title += "\nnot run: " + ", ".join(skipped)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in formats:
        path = output.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=220)
        written.append(path)
    plt.close(figure)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-name", default="reconstruction")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = _load(args.report)
    _write_csv(
        args.output_dir / "reconstruction_by_frame.csv",
        curve_rows(report),
        ["method", "family", "bits", "tensor", "frame", "nmse", "cumulative_nmse"],
    )
    _write_csv(
        args.output_dir / "reconstruction_summary.csv",
        summary_rows(report),
        [
            "method", "family", "bits", "tensor", "status", "reason",
            "first_frame_nmse", "last_frame_nmse", "max_frame_nmse",
            "head_mean_nmse", "tail_mean_nmse", "tail_over_head", "pooled_nmse",
        ],
    )
    try:
        written = render(report, args.output_dir / args.figure_name, formats=args.formats)
    except RuntimeError as error:
        print(f"[plot] {error}; CSVs were still written")
        return 0
    for path in written:
        print(f"[plot] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
