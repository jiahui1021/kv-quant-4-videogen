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
            block["global_end_index"].fill_(0)
            block["local_end_index"].fill_(0)
            block["quant_state"] = None
            if isinstance(block.get("k"), torch.Tensor) and block["k"].numel():
                block["k"].zero_()
            if isinstance(block.get("v"), torch.Tensor) and block["v"].numel():
                block["v"].zero_()


def active_kv_memory_bytes(pipeline, quantizer) -> tuple[int, int]:
    """Return active BF16-equivalent and compressed cache bytes."""
    bf16_bytes = 0
    compressed_bytes = 0
    for cache_list in _cache_lists(pipeline):
        for block in cache_list:
            active_tokens = int(block["local_end_index"].item())
            batch_size = int(block.get("batch_size", 0))
            num_heads = int(block.get("num_heads", 0))
            head_dim = int(block.get("head_dim", 0))
            bf16_bytes += (
                batch_size * active_tokens * num_heads * head_dim * 2 * 2
            )
            state = block.get("quant_state")
            if state is not None:
                compressed_bytes += int(quantizer.memory_bytes(state))
            else:
                compressed_bytes += (
                    batch_size * active_tokens * num_heads * head_dim * 2 * 2
                )
    return int(bf16_bytes), int(compressed_bytes)
