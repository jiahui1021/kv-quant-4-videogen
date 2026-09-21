"""Runtime helpers for the shared KV-cache quantizers in Causal-Forcing."""

from __future__ import annotations

import torch


def _cache_lists(pipeline) -> list[list[dict]]:
    if hasattr(pipeline, "kv_cache1"):
        return [pipeline.kv_cache1] if pipeline.kv_cache1 is not None else []
    lists = []
    for name in ("kv_cache_pos", "kv_cache_neg"):
        value = getattr(pipeline, name, None)
        if value is not None:
            lists.append(value)
    return lists


def _initialize_pipeline_caches(pipeline, dtype, device) -> None:
    if hasattr(pipeline, "kv_cache1"):
        if pipeline.kv_cache1 is None:
            pipeline._initialize_kv_cache(batch_size=1, dtype=dtype, device=device)
            pipeline._initialize_crossattn_cache(
                batch_size=1, dtype=dtype, device=device
            )
        return

    if getattr(pipeline, "kv_cache_pos", None) is None:
        pipeline._initialize_kv_cache(batch_size=1, dtype=dtype, device=device)
        pipeline._initialize_crossattn_cache(
            batch_size=1, dtype=dtype, device=device
        )


def _expand_unbounded_cache(block: dict, required_tokens: int) -> None:
    current_k = block["k"]
    current_size = int(block.get("kv_cache_size", current_k.shape[1]))
    if current_size >= required_tokens:
        return
    if current_k.numel() == 0:
        batch_size = int(block["batch_size"])
        num_heads = int(block["num_heads"])
        head_dim = int(block["head_dim"])
        dtype = block.get("dtype", torch.bfloat16)
        device = block.get("device", "cuda")
    else:
        batch_size, _, num_heads, head_dim = current_k.shape
        dtype, device = current_k.dtype, current_k.device
    new_k = torch.zeros(
        [batch_size, required_tokens, num_heads, head_dim], dtype=dtype, device=device
    )
    new_v = torch.zeros_like(new_k)
    if current_k.numel() > 0:
        copy_tokens = min(current_k.shape[1], required_tokens)
        new_k[:, :copy_tokens] = current_k[:, :copy_tokens]
        new_v[:, :copy_tokens] = block["v"][:, :copy_tokens]
    block["k"], block["v"] = new_k, new_v


def attach_quantizer_to_pipeline(
    pipeline,
    quantizer,
    num_output_frames: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Attach a shared quantizer to every causal self-attention cache."""
    _initialize_pipeline_caches(pipeline, dtype=dtype, device=device)
    required_tokens = int(num_output_frames) * int(pipeline.frame_seq_length)

    for cache_list in _cache_lists(pipeline):
        for layer_idx, block in enumerate(cache_list):
            k = block["k"]
            if pipeline.local_attn_size == -1 and k.shape[1] < required_tokens:
                block["batch_size"] = int(k.shape[0])
                block["num_heads"] = int(k.shape[2])
                block["head_dim"] = int(k.shape[3])
                block["dtype"] = dtype
                block["device"] = device
                _expand_unbounded_cache(block, required_tokens)
                k = block["k"]

            block["kv_cache_size"] = int(k.shape[1])
            block["batch_size"] = int(k.shape[0])
            block["num_heads"] = int(k.shape[2])
            block["head_dim"] = int(k.shape[3])
            block["frame_seq_length"] = int(pipeline.frame_seq_length)
            block["num_frame_per_block"] = int(pipeline.num_frame_per_block)
            block["layer_id"] = int(layer_idx)
            block["quantizer"] = quantizer
            block["quant_state"] = None
            block["k"] = torch.empty(0, dtype=dtype, device=device)
            block["v"] = torch.empty(0, dtype=dtype, device=device)


def reset_quantized_kv_cache(pipeline) -> None:
    """Reset indices and compressed state before processing a new prompt."""
    for cache_list in _cache_lists(pipeline):
        for block in cache_list:
            for name in ("global_end_index", "local_end_index"):
                value = block[name]
                if isinstance(value, torch.Tensor):
                    # Keep the legacy tensor object intact for callers that
                    # own it, while accepting the host-integer cursors used by
                    # the current Causal-Forcing pipeline.
                    value.fill_(0)
                else:
                    block[name] = 0
            block["quant_state"] = None
            block.pop("eviction_slack_tokens", None)
            for key in ("recent_k", "recent_v"):
                value = block.get(key)
                if isinstance(value, torch.Tensor) and value.ndim == 4:
                    block[key] = value[:, :0].clone()
            block["recent_start_index"] = 0
            block["recent_end_index"] = 0
            if isinstance(block.get("k"), torch.Tensor) and block["k"].numel():
                block["k"].zero_()
            if isinstance(block.get("v"), torch.Tensor) and block["v"].numel():
                block["v"].zero_()


def finalize_quantized_kv_cache(pipeline, quantizer=None) -> None:
    """Commit the final mutable write buffer after generation finishes."""
    if quantizer is None or not callable(getattr(quantizer, "finalize_state", None)):
        return
    for cache_list in _cache_lists(pipeline):
        for block in cache_list:
            state = block.get("quant_state")
            if state is None:
                continue
            write_k = state.get("write_k") if isinstance(state, dict) else None
            dtype = write_k.dtype if isinstance(write_k, torch.Tensor) else block.get("dtype", torch.bfloat16)
            quantizer.finalize_state(state, meta={"tensor_dtype": dtype})


def _resident_tokens(block: dict) -> int:
    """Cache positions the block holds, never the size it was allocated at."""
    state = block.get("quant_state")
    if isinstance(state, dict) and state.get("num_tokens") is not None:
        return int(state["num_tokens"])
    end_index = block.get("local_end_index")
    if isinstance(end_index, torch.Tensor):
        return int(end_index.item())
    if end_index is not None:
        return int(end_index)
    cache_k = block.get("k")
    if isinstance(cache_k, torch.Tensor) and cache_k.ndim == 4:
        return int(cache_k.shape[1])
    return 0


def _block_geometry(block: dict) -> tuple[int, int, int, int]:
    cache_k = block.get("k")
    if isinstance(cache_k, torch.Tensor) and cache_k.ndim == 4 and cache_k.shape[1] > 0:
        batch_size, _, num_heads, head_dim = cache_k.shape
        return int(batch_size), int(num_heads), int(head_dim), int(cache_k.element_size())
    dtype = block.get("dtype", torch.bfloat16)
    return (
        int(block.get("batch_size", 0)),
        int(block.get("num_heads", 0)),
        int(block.get("head_dim", 0)),
        int(torch.empty((), dtype=dtype).element_size()),
    )


def resident_kv_memory_bytes(pipeline, quantizer=None) -> tuple[int, int]:
    """BF16 equivalent and resident bytes under ``resident_analytic.v1``.

    The equivalent is the BF16 cost of the cache positions actually held, the
    same figure the QVG breakdown and the Self-Forcing record report, so every
    method and the BF16 row answer the same question.  Counting the
    preallocated capacity instead tied the ratio to the cache allocation: a
    run whose window differed from its video length reported a compression
    ratio that had nothing to do with the quantizer.
    """
    bf16_bytes = 0
    resident_bytes = 0
    for cache_list in _cache_lists(pipeline):
        for block in cache_list:
            tokens = _resident_tokens(block)
            batch_size, num_heads, head_dim, element_size = _block_geometry(block)
            block_bf16_bytes = (
                batch_size * tokens * num_heads * head_dim * element_size * 2
            )
            bf16_bytes += block_bf16_bytes
            state = block.get("quant_state")
            if quantizer is not None:
                if state is not None:
                    resident_bytes += int(quantizer.memory_bytes(state))
            else:
                resident_bytes += block_bf16_bytes
    return int(bf16_bytes), int(resident_bytes)


def active_kv_memory_bytes(pipeline, quantizer=None) -> tuple[int, int]:
    """Backward-compatible alias for resident-capacity accounting."""
    return resident_kv_memory_bytes(pipeline, quantizer)
