"""Small, method-agnostic helpers for append-only KV-cache states.

The quantizers in this repository use different metadata and quantization
axes, but they all need the same lifetime rules in the model runtime:

* quantized segments are immutable;
* a residual buffer may hold data which is not ready for quantization yet;
* the current diffusion block stays in a write buffer so repeated denoising
  calls can replace it without re-quantizing history.

This module deliberately stores quantized segments as a list.  Concatenating
packed payloads would require unpacking/repacking whenever a segment is not
byte aligned, which would violate append invariance.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Tuple

import torch


INCREMENTAL_STATE_FORMAT = "incremental_kv_v1"


def new_state(meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    meta = dict(meta or {})
    shape = tuple(int(dim) for dim in meta.get("shape", (0, 0, 0, 0)))
    return {
        "format": INCREMENTAL_STATE_FORMAT,
        "segments": [],
        # ``k``/``v`` are compatibility aliases for the first segment.  They
        # are intentionally not used for accounting because they alias the
        # objects held in ``segments``.
        "k": None,
        "v": None,
        "residual_k": None,
        "residual_v": None,
        "write_k": None,
        "write_v": None,
        "quantized_tokens": 0,
        "num_tokens": 0,
        "shape": shape,
        "tensor_dtype": meta.get("tensor_dtype"),
        "write_start": None,
        "write_end": None,
        "attention_space": bool(meta.get("attention_space", False)),
    }


def is_incremental_state(state: Any) -> bool:
    return isinstance(state, dict) and state.get("format") == INCREMENTAL_STATE_FORMAT


def _state_length(tensor_state: Dict[str, Any]) -> int:
    shape = tensor_state.get("orig_shape")
    if shape is None or len(shape) < 2:
        raise ValueError("Quantized tensor state must contain orig_shape with a sequence dimension")
    return int(shape[1])


def append_segment(
    state: Dict[str, Any],
    k_state: Dict[str, Any],
    v_state: Dict[str, Any],
    length: int | None = None,
) -> None:
    if length is None:
        length = _state_length(k_state)
    length = int(length)
    if length <= 0:
        return
    segment = {"k": k_state, "v": v_state, "length": length}
    state["segments"].append(segment)
    if state.get("k") is None:
        state["k"] = k_state
        state["v"] = v_state


def _refresh_aliases(state: Dict[str, Any]) -> None:
    if state["segments"]:
        state["k"] = state["segments"][0]["k"]
        state["v"] = state["segments"][0]["v"]
    else:
        state["k"] = None
        state["v"] = None


def recompute_counts(state: Dict[str, Any]) -> None:
    quantized_tokens = sum(int(segment["length"]) for segment in state["segments"])
    residual_k = state.get("residual_k")
    write_k = state.get("write_k")
    residual_tokens = int(residual_k.shape[1]) if isinstance(residual_k, torch.Tensor) else 0
    write_tokens = int(write_k.shape[1]) if isinstance(write_k, torch.Tensor) else 0
    state["quantized_tokens"] = quantized_tokens
    state["num_tokens"] = quantized_tokens + residual_tokens + write_tokens
    shape = state.get("shape", (0, 0, 0, 0))
    if len(shape) == 4 and isinstance(residual_k, torch.Tensor):
        state["shape"] = (
            int(residual_k.shape[0]),
            int(state["num_tokens"]),
            int(residual_k.shape[2]),
            int(residual_k.shape[3]),
        )
    elif len(shape) == 4 and isinstance(write_k, torch.Tensor):
        state["shape"] = (
            int(write_k.shape[0]),
            int(state["num_tokens"]),
            int(write_k.shape[2]),
            int(write_k.shape[3]),
        )


def ensure_state(state: Dict[str, Any], meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Convert a legacy one-shot state in place to the incremental schema."""
    if is_incremental_state(state):
        for key, value in new_state(meta).items():
            state.setdefault(key, value)
        return state

    old_k = state.get("k")
    old_v = state.get("v")
    if old_k is None or old_v is None:
        state.clear()
        state.update(new_state(meta))
        return state

    old_shape = tuple(int(dim) for dim in old_k.get("orig_shape", (0, 0, 0, 0)))
    inferred_meta = dict(meta or {})
    inferred_meta.setdefault("shape", old_shape)
    inferred_meta.setdefault("tensor_dtype", old_k.get("tensor_dtype"))
    converted = new_state(inferred_meta)
    converted["segments"].append({"k": old_k, "v": old_v, "length": _state_length(old_k)})
    converted["k"] = old_k
    converted["v"] = old_v
    converted["quantized_tokens"] = _state_length(old_k)
    converted["num_tokens"] = _state_length(old_k)
    state.clear()
    state.update(converted)
    return state


def _same_range(
    state: Dict[str, Any],
    start: Any,
    end: Any,
) -> bool:
    return (
        start is not None
        and end is not None
        and state.get("write_start") == int(start)
        and state.get("write_end") == int(end)
    )


def prepare_write(
    state: Dict[str, Any],
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    meta: Dict[str, Any] | None,
    commit: Callable[[Dict[str, Any], torch.Tensor, torch.Tensor, Dict[str, Any]], None],
) -> str:
    """Install a new mutable write, committing the previous one if needed.

    Returns ``"replace"`` when the current absolute range was overwritten and
    ``"append"`` for a new range.  The callback is responsible for moving the
    previous write into the quantized/residual part of the method-specific
    state.
    """
    meta = dict(meta or {})
    start = meta.get("absolute_start", meta.get("global_start"))
    end = meta.get("absolute_end", meta.get("global_end"))
    old_write_k = state.get("write_k")
    old_write_v = state.get("write_v")

    if isinstance(old_write_k, torch.Tensor):
        if _same_range(state, start, end):
            if old_write_k.shape != new_k.shape or old_write_v.shape != new_v.shape:
                raise ValueError("A repeated KV-cache write must keep the same shape")
            state["write_k"] = new_k
            state["write_v"] = new_v
            state["tensor_dtype"] = meta.get("tensor_dtype", new_k.dtype)
            if meta.get("attention_space") is not None:
                state["attention_space"] = bool(meta["attention_space"])
            recompute_counts(state)
            return "replace"

        previous_end = state.get("write_end")
        if start is not None and previous_end is not None and int(start) < int(previous_end):
            raise ValueError(
                "Cannot overwrite an already committed KV-cache range; "
                "only the current write range may be replaced"
            )
        commit(state, old_write_k, old_write_v, meta)
        state["write_k"] = None
        state["write_v"] = None

    state["write_k"] = new_k
    state["write_v"] = new_v
    state["tensor_dtype"] = meta.get("tensor_dtype", new_k.dtype)
    if start is not None:
        state["write_start"] = int(start)
    else:
        state["write_start"] = None
    if end is not None:
        state["write_end"] = int(end)
    else:
        state["write_end"] = None
    if meta.get("attention_space") is not None:
        state["attention_space"] = bool(meta["attention_space"])
    recompute_counts(state)
    return "append"


def iter_tensor_parts(state: Dict[str, Any], key: str) -> Iterable[torch.Tensor]:
    for segment in state["segments"]:
        yield segment[key]
    residual = state.get(f"residual_{key}")
    if isinstance(residual, torch.Tensor) and residual.shape[1] > 0:
        yield residual
    write = state.get(f"write_{key}")
    if isinstance(write, torch.Tensor) and write.shape[1] > 0:
        yield write


def _empty_from_state(
    state: Dict[str, Any],
    meta: Dict[str, Any] | None,
    dtype: torch.dtype | None,
    device: torch.device | str | None,
) -> torch.Tensor:
    meta = dict(meta or {})
    shape = tuple(int(dim) for dim in meta.get("shape", state.get("shape", (0, 0, 0, 0))))
    if len(shape) != 4:
        raise ValueError(f"Expected an empty KV shape [B, 0, H, D], got {shape}")
    shape = (shape[0], 0, shape[2], shape[3])
    if dtype is None:
        dtype = meta.get("tensor_dtype") or state.get("tensor_dtype") or torch.bfloat16
    if device is None:
        device = meta.get("device", "cpu")
    return torch.empty(shape, dtype=dtype, device=device)


def materialize(
    state: Dict[str, Any],
    decode_segment: Callable[[Dict[str, Any]], torch.Tensor],
    meta: Dict[str, Any] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k_parts = list(iter_tensor_parts(state, "k"))
    v_parts = list(iter_tensor_parts(state, "v"))
    if not k_parts:
        empty = _empty_from_state(state, meta, None, None)
        return empty, empty.clone()
    decoded_k = []
    decoded_v = []
    for segment in state["segments"]:
        decoded_k.append(decode_segment(segment["k"]))
        decoded_v.append(decode_segment(segment["v"]))
    residual_k = state.get("residual_k")
    residual_v = state.get("residual_v")
    if isinstance(residual_k, torch.Tensor) and residual_k.shape[1] > 0:
        decoded_k.append(residual_k)
        decoded_v.append(residual_v)
    write_k = state.get("write_k")
    write_v = state.get("write_v")
    if isinstance(write_k, torch.Tensor) and write_k.shape[1] > 0:
        decoded_k.append(write_k)
        decoded_v.append(write_v)
    return torch.cat(decoded_k, dim=1), torch.cat(decoded_v, dim=1)


def tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    return 0


def state_device(state: Dict[str, Any]) -> torch.device:
    for key in ("write_k", "residual_k"):
        value = state.get(key)
        if isinstance(value, torch.Tensor):
            return value.device
    for segment in state.get("segments", []):
        value = segment["k"].get("q")
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def evict_prefix(state: Dict[str, Any], requested_tokens: int) -> int:
    """Drop complete immutable segments from the logical prefix.

    A partial segment is deliberately left untouched.  The caller receives
    the actual number removed and can keep the remaining block as a small
    capacity slack.  No packed payload is unpacked or re-quantized.
    """
    remaining = max(int(requested_tokens), 0)
    removed = 0
    while state["segments"] and remaining >= int(state["segments"][0]["length"]):
        length = int(state["segments"][0]["length"])
        state["segments"].pop(0)
        remaining -= length
        removed += length
    if removed:
        _refresh_aliases(state)
        recompute_counts(state)
    return removed


def evict_range(state: Dict[str, Any], start_tokens: int, requested_tokens: int) -> int:
    """Drop a complete, immutable segment range without re-packing it.

    This is used for local attention with sink tokens.  A range which cuts a
    segment or reaches the residual/write area is left untouched and returns
    zero; callers can then retain a small capacity slack safely.
    """
    start = max(int(start_tokens), 0)
    requested = max(int(requested_tokens), 0)
    end = start + requested
    if requested == 0:
        return 0
    cursor = 0
    remove_indices = []
    for index, segment in enumerate(state["segments"]):
        segment_start = cursor
        segment_end = cursor + int(segment["length"])
        cursor = segment_end
        overlaps = segment_start < end and segment_end > start
        if not overlaps:
            continue
        if segment_start < start or segment_end > end:
            return 0
        remove_indices.append(index)
    if cursor < end or not remove_indices:
        return 0
    removed = sum(int(state["segments"][index]["length"]) for index in remove_indices)
    for index in reversed(remove_indices):
        state["segments"].pop(index)
    _refresh_aliases(state)
    recompute_counts(state)
    return removed


def state_memory_bytes(
    state: Dict[str, Any],
    segment_memory_bytes: Callable[[Dict[str, Any]], int],
) -> int:
    total = sum(
        int(segment_memory_bytes({"k": segment["k"], "v": segment["v"]}))
        for segment in state.get("segments", [])
    )
    total += tensor_bytes(state.get("residual_k"))
    total += tensor_bytes(state.get("residual_v"))
    total += tensor_bytes(state.get("write_k"))
    total += tensor_bytes(state.get("write_v"))
    return int(total)
