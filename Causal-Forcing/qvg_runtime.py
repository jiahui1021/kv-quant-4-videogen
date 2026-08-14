"""Causal-Forcing adapter for the vendored official Quant-VideoGen codec."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import torch


QVG_UPSTREAM_COMMIT = "0601468f2dbba6a17ac7086faec6d41527cad188"
SUPPORTED_QVG_BITS = (2, 4)
_VENDOR_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "Quant-VideoGen"


@dataclass
class QVGConfig:
    """Official QVG configuration for the supported INT2/INT4 codecs."""

    bits: int = 2
    quant_type: str | None = None
    num_k_centroids: int = 256
    num_v_centroids: int = 256
    kmeans_max_iters: int = 2
    quant_block_size: int = 64
    num_prq_stages: int = 1
    quant_factor: int = 8
    asymmetric: bool = False
    timing_enabled: bool = False
    # Debug-only switch used to validate the cache/RoPE integration before
    # enabling the real Triton codec.  The cache remains a QVG cache, but no
    # finalized span is compressed while this is false.
    compression_enabled: bool = True

    def __post_init__(self) -> None:
        expected_type = f"triton-nstages-kmeans-int{self.bits}"
        if self.bits not in SUPPORTED_QVG_BITS:
            raise ValueError(
                f"Unsupported QVG bit width {self.bits}; "
                f"supported widths are {SUPPORTED_QVG_BITS}"
            )
        if self.quant_type is None:
            self.quant_type = expected_type
        elif self.quant_type != expected_type:
            raise ValueError(
                f"bits={self.bits} requires quant_type={expected_type}, "
                f"got {self.quant_type}"
            )
        if self.quant_factor <= 0:
            raise ValueError("quant_factor must be positive")
        if self.quant_block_size <= 0:
            raise ValueError("quant_block_size must be positive")
        if self.num_k_centroids <= 0 or self.num_v_centroids <= 0:
            raise ValueError("QVG centroid counts must be positive")
        if self.kmeans_max_iters <= 0 or self.num_prq_stages <= 0:
            raise ValueError("QVG K-Means iterations and PRQ stages must be positive")

    @property
    def cache_num_k_centroids(self) -> int:
        return self.num_k_centroids

    @property
    def cache_num_v_centroids(self) -> int:
        return self.num_v_centroids

    @classmethod
    def from_method(cls, method: str, **kwargs) -> "QVGConfig":
        method = method.upper()
        if not method.startswith("QVG_INT"):
            raise ValueError(f"Unsupported QVG method: {method}")
        try:
            bits = int(method.rsplit("INT", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Unsupported QVG method: {method}") from exc
        if bits not in SUPPORTED_QVG_BITS:
            supported = ", ".join(f"QVG_INT{value}" for value in SUPPORTED_QVG_BITS)
            raise ValueError(
                f"Unsupported QVG method: {method}; supported methods: {supported}"
            )
        return cls(
            bits=bits,
            quant_type=f"triton-nstages-kmeans-int{bits}",
            **kwargs,
        )


@dataclass
class QVGStats:
    timing_enabled: bool = False
    quantize_time_s: float = 0.0
    dequantize_time_s: float = 0.0
    quantize_calls: int = 0
    dequantize_calls: int = 0
    quantized_spans: int = 0
    quantized_chunks: int = 0
    _quantize_events: list[tuple[Any, Any]] = field(
        default_factory=list, repr=False
    )
    _dequantize_events: list[tuple[Any, Any]] = field(
        default_factory=list, repr=False
    )

    def reset(self) -> None:
        self.quantize_time_s = 0.0
        self.dequantize_time_s = 0.0
        self.quantize_calls = 0
        self.dequantize_calls = 0
        self.quantized_spans = 0
        self.quantized_chunks = 0
        self._quantize_events.clear()
        self._dequantize_events.clear()

    def resolve_timing(self, synchronize: bool = True) -> None:
        if (
            synchronize
            and (self._quantize_events or self._dequantize_events)
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize()
        for start, end in self._quantize_events:
            self.quantize_time_s += float(start.elapsed_time(end)) / 1000.0
        for start, end in self._dequantize_events:
            self.dequantize_time_s += float(start.elapsed_time(end)) / 1000.0
        self._quantize_events.clear()
        self._dequantize_events.clear()


@dataclass(frozen=True)
class QVGMemory:
    bf16_equivalent_bytes: int
    physical_bytes: int
    physical_bf16_bytes: int
    physical_compressed_bytes: int
    bf16_chunks: int
    quantized_chunks: int

    @property
    def logical_values(self) -> int:
        return self.bf16_equivalent_bytes // 2


def _load_qvg():
    """Load the pinned vendored codec only when QVG is selected."""
    vendor = str(_VENDOR_ROOT)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    try:
        from quant_videogen.compress import compress_kv_cache, get_quantize_fn
        from quant_videogen.kv_cache import ChunkedKVCache
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "QVG requires the vendored Quant-VideoGen runtime dependencies, "
            "including Triton. Install them in the Causal-Forcing environment."
        ) from exc
    return ChunkedKVCache, compress_kv_cache, get_quantize_fn


def _qvg_debug_enabled() -> bool:
    # Read the environment at the point of use so a launcher can enable debug
    # logging without importing this module again.
    import os

    return os.environ.get("QVG_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _qvg_debug(message: str) -> None:
    if _qvg_debug_enabled():
        print(f"[QVG_DEBUG] {message}", flush=True)


def _cache_lists(pipeline) -> list[list[dict]]:
    if hasattr(pipeline, "kv_cache1"):
        value = getattr(pipeline, "kv_cache1", None)
        return [] if value is None else [value]
    values = []
    for name in ("kv_cache_pos", "kv_cache_neg"):
        value = getattr(pipeline, name, None)
        if value is not None:
            values.append(value)
    return values


def _initialize_pipeline_cache_shells(pipeline, dtype, device) -> None:
    """Create shape-only cache entries without allocating full BF16 K/V."""
    if getattr(pipeline, "kv_cache1", None) is not None:
        return
    try:
        blocks = pipeline.generator.model.blocks
        first_attention = blocks[0].self_attn
        num_layers = len(blocks)
        num_heads = int(first_attention.num_heads)
        head_dim = int(first_attention.head_dim)
    except (AttributeError, IndexError, TypeError):
        pipeline._initialize_kv_cache(
            batch_size=1, dtype=dtype, device=device
        )
        return

    pipeline.kv_cache1 = [
        {
            "k": torch.empty(
                1, 0, num_heads, head_dim, dtype=dtype, device=device
            ),
            "v": torch.empty(
                1, 0, num_heads, head_dim, dtype=dtype, device=device
            ),
            "global_end_index": torch.zeros(
                1, dtype=torch.long, device=device
            ),
            "local_end_index": torch.zeros(
                1, dtype=torch.long, device=device
            ),
        }
        for _ in range(num_layers)
    ]


def attach_qvg_to_pipeline(
    pipeline,
    config: QVGConfig,
    num_output_frames: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Replace causal self-attention caches with absolute-position QVG caches."""
    ChunkedKVCache, _, _ = _load_qvg()
    model = getattr(getattr(pipeline, "generator", None), "model", None)
    if any(
        int(getattr(block.self_attn, "sink_size", 0)) != 0
        for block in getattr(model, "blocks", ())
    ):
        raise NotImplementedError(
            f"QVG_INT{config.bits} currently requires sink_size=0"
        )
    local_attn_size = int(getattr(pipeline, "local_attn_size", -1))
    frames_per_block = int(getattr(pipeline, "num_frame_per_block", 1))
    minimum_local_frames = config.quant_factor * frames_per_block
    attention_window_frames = (
        32760 // int(pipeline.frame_seq_length)
        if local_attn_size == -1
        else local_attn_size
    )
    if attention_window_frames < minimum_local_frames:
        raise NotImplementedError(
            f"QVG_INT{config.bits} requires an attention window of at least "
            f"{minimum_local_frames} frames so the first span can be "
            "materialized before eviction"
        )
    _initialize_pipeline_cache_shells(pipeline, dtype=dtype, device=device)
    if getattr(pipeline, "crossattn_cache", None) is None:
        pipeline._initialize_crossattn_cache(
            batch_size=1, dtype=dtype, device=device
        )
    max_num_chunks = int(num_output_frames)
    if max_num_chunks <= 0:
        raise ValueError("num_output_frames must be positive")

    stats = QVGStats(timing_enabled=bool(config.timing_enabled))
    cache_lists = _cache_lists(pipeline)
    if not cache_lists:
        raise RuntimeError("Pipeline did not initialize a causal KV cache")

    for cache_list in cache_lists:
        for layer_idx, block in enumerate(cache_list):
            old_k = block.get("k")
            if not isinstance(old_k, torch.Tensor) or old_k.ndim != 4:
                raise RuntimeError(
                    f"Cannot infer QVG cache shape for layer {layer_idx}"
                )
            batch_size, _, num_heads, head_dim = old_k.shape
            block["k"] = ChunkedKVCache(
                int(batch_size),
                int(pipeline.frame_seq_length),
                int(num_heads),
                int(head_dim),
                max_num_chunks,
                dtype,
                device,
                layout="BSHD",
            )
            block["v"] = ChunkedKVCache(
                int(batch_size),
                int(pipeline.frame_seq_length),
                int(num_heads),
                int(head_dim),
                max_num_chunks,
                dtype,
                device,
                layout="BSHD",
            )
            block["cache_backend"] = "qvg"
            block["qvg_config"] = config
            block["qvg_stats"] = stats
            block["kv_cache_size"] = max_num_chunks * int(
                pipeline.frame_seq_length
            )
            block["global_end_index"].zero_()
            block["local_end_index"].zero_()
            block["qvg_evicted_until"] = 0
            block.pop("quantizer", None)
            block.pop("quant_state", None)

    pipeline.qvg_enabled = True
    pipeline.qvg_config = config
    pipeline.qvg_stats = stats
    pipeline.qvg_max_num_chunks = max_num_chunks
    pipeline.qvg_scheduled_spans = set()
    pipeline.qvg_compression_enabled = bool(config.compression_enabled)
    _qvg_debug(
        "attached "
        f"INT{config.bits}: capacity={max_num_chunks} frames, "
        f"frame_seq_length={pipeline.frame_seq_length}, "
        f"compression_enabled={config.compression_enabled}"
    )


def reset_qvg_cache(pipeline, reset_stats: bool = True) -> None:
    """Clear every QVG span and mutable BF16 chunk for a new prompt."""
    for cache_list in _cache_lists(pipeline):
        for block in cache_list:
            if block.get("cache_backend") != "qvg":
                continue
            block["k"].clear()
            block["v"].clear()
            block["global_end_index"].zero_()
            block["local_end_index"].zero_()
            block["qvg_evicted_until"] = 0
    stats = getattr(pipeline, "qvg_stats", None)
    if reset_stats and stats is not None:
        stats.reset()
    if hasattr(pipeline, "qvg_scheduled_spans"):
        pipeline.qvg_scheduled_spans.clear()
    _qvg_debug("cache reset")


def qvg_span_for_block(
    block_index: int,
    all_num_frames: Sequence[int],
    frame_seq_length: int,
    quant_factor: int = 8,
    generation_start_frame: int = 0,
) -> tuple[int, int] | None:
    """Return the finalized token span compressed before ``block_index``."""
    if block_index < quant_factor or block_index % quant_factor != 0:
        return None
    start_block = block_index - quant_factor
    start_frame = int(generation_start_frame) + sum(
        int(value) for value in all_num_frames[:start_block]
    )
    end_frame = int(generation_start_frame) + sum(
        int(value) for value in all_num_frames[:block_index]
    )
    return start_frame * int(frame_seq_length), end_frame * int(frame_seq_length)


def ensure_qvg_capacity(pipeline, required_frames: int) -> None:
    """Grow empty QVG caches before a prompt with more context frames."""
    if not getattr(pipeline, "qvg_enabled", False):
        return
    required_frames = int(required_frames)
    if required_frames <= int(getattr(pipeline, "qvg_max_num_chunks", 0)):
        return
    ChunkedKVCache, _, _ = _load_qvg()
    for cache_list in _cache_lists(pipeline):
        for block in cache_list:
            old_k = block["k"]
            old_v = block["v"]
            for name, old in (("k", old_k), ("v", old_v)):
                new = ChunkedKVCache(
                    int(old.batch_size),
                    int(old.frame_seq_length),
                    int(old.num_heads),
                    int(old.head_dim),
                    required_frames,
                    old.dtype,
                    old.device,
                    layout=old.layout,
                )
                block[name] = new
            block["kv_cache_size"] = required_frames * int(
                pipeline.frame_seq_length
            )
            block["global_end_index"].zero_()
            block["local_end_index"].zero_()
            block["qvg_evicted_until"] = 0
    pipeline.qvg_max_num_chunks = required_frames


def maybe_quantize_qvg_history(
    pipeline,
    block_index: int,
    all_num_frames: Sequence[int],
    generation_start_frame: int = 0,
) -> bool:
    """Apply the official once-per-eight-finalized-chunks QVG schedule."""
    if not getattr(pipeline, "qvg_enabled", False):
        return False
    config = pipeline.qvg_config
    if not bool(getattr(config, "compression_enabled", True)):
        return False
    span = qvg_span_for_block(
        block_index,
        all_num_frames,
        pipeline.frame_seq_length,
        quant_factor=config.quant_factor,
        generation_start_frame=generation_start_frame,
    )
    if span is None:
        return False
    scheduled = getattr(pipeline, "qvg_scheduled_spans", None)
    if scheduled is None:
        scheduled = set()
        pipeline.qvg_scheduled_spans = scheduled
    if span in scheduled:
        return False
    quantize_qvg_span(pipeline, span[0], span[1])
    scheduled.add(span)
    _qvg_debug(
        f"quantized span tokens=[{span[0]},{span[1]}) "
        f"at generation block {block_index}"
    )
    return True


def _range_is_already_quantized(cache, start_token: int, end_token: int) -> bool:
    frame_tokens = int(cache.frame_seq_length)
    start_chunk = start_token // frame_tokens
    end_chunk = end_token // frame_tokens
    for span in cache.quantized_spans:
        if not (
            int(span["end_chunk"]) <= start_chunk
            or int(span["start_chunk"]) >= end_chunk
        ):
            return True
    return False


def _record_cuda_event(stats: QVGStats, kind: str):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    target = (
        stats._quantize_events
        if kind == "quantize"
        else stats._dequantize_events
    )
    return start, end, target


def quantize_qvg_span(pipeline, start_token: int, end_token: int) -> None:
    """Compress one immutable finalized span with the official real codec."""
    if start_token >= end_token:
        raise ValueError("QVG span must be non-empty")
    _, compress_kv_cache, get_quantize_fn = _load_qvg()
    config: QVGConfig = pipeline.qvg_config
    if not bool(getattr(config, "compression_enabled", True)):
        raise RuntimeError("QVG compression is disabled in BF16 debug mode")
    stats: QVGStats = pipeline.qvg_stats
    quantize_fn = get_quantize_fn(config.quant_type, config)
    cache_lists = _cache_lists(pipeline)
    if not cache_lists:
        raise RuntimeError("No QVG caches are attached")

    # Preflight every layer before modifying any cache.  This prevents a
    # failed compressor on a later layer from leaving a partially quantized
    # span that cannot be retried safely.
    for cache_list in cache_lists:
        for layer_idx, layer in enumerate(cache_list):
            if _range_is_already_quantized(layer["k"], start_token, end_token):
                raise RuntimeError(
                    f"QVG span [{start_token}, {end_token}) overlaps an "
                    f"immutable K span in layer {layer_idx}"
                )
            if _range_is_already_quantized(layer["v"], start_token, end_token):
                raise RuntimeError(
                    f"QVG span [{start_token}, {end_token}) overlaps an "
                    f"immutable V span in layer {layer_idx}"
                )

    for cache_list in cache_lists:
        for layer_idx, layer in enumerate(cache_list):
            k_cache = layer["k"]
            v_cache = layer["v"]

            k = k_cache.read(start_token, end_token)
            v = v_cache.read(start_token, end_token)
            k_bhsd = k.permute(0, 2, 1, 3).contiguous()
            v_bhsd = v.permute(0, 2, 1, 3).contiguous()

            cuda_timing = bool(config.timing_enabled and k.is_cuda)
            if cuda_timing:
                start_event, end_event, target = _record_cuda_event(
                    stats, "quantize"
                )
            else:
                started = time.perf_counter()
            k_quant, v_quant = compress_kv_cache(
                k_bhsd,
                v_bhsd,
                config.quant_type,
                config,
                quantize_fn,
            )
            if cuda_timing:
                end_event.record()
                target.append((start_event, end_event))
            else:
                stats.quantize_time_s += time.perf_counter() - started
            stats.quantize_calls += 1

            if not isinstance(k_quant, dict) or not isinstance(v_quant, dict):
                raise RuntimeError("QVG real codec did not return packed states")
            k_quant["info"] = {
                "output_dtype": k.dtype,
                "quant_config": config,
            }
            v_quant["info"] = {
                "output_dtype": v.dtype,
                "quant_config": config,
            }
            k_cache.store_quantized(start_token, end_token, k_quant)
            v_cache.store_quantized(start_token, end_token, v_quant)

    stats.quantized_spans += 1
    stats.quantized_chunks += (end_token - start_token) // int(
        pipeline.frame_seq_length
    )
    _qvg_debug(
        f"stored span tokens=[{start_token},{end_token}), "
        f"frame_chunks={stats.quantized_chunks}"
    )
    if _qvg_debug_enabled():
        memory = qvg_memory_breakdown(pipeline)
        _qvg_debug(
            "memory "
            f"bf16_tail={memory.physical_bf16_bytes} bytes, "
            f"packed={memory.physical_compressed_bytes} bytes, "
            f"total={memory.physical_bytes} bytes"
        )


def _range_contains_quantized(cache, start_token: int, end_token: int) -> bool:
    frame_tokens = int(cache.frame_seq_length)
    start_chunk = start_token // frame_tokens
    end_chunk = end_token // frame_tokens
    return any(
        int(span["start_chunk"]) < end_chunk
        and int(span["end_chunk"]) > start_chunk
        for span in cache.quantized_spans
    )


def read_qvg_cache(cache, start_token: int, end_token: int, stats: QVGStats):
    """Read one QVG cache while measuring actual decompression work."""
    is_quantized = _range_contains_quantized(cache, start_token, end_token)
    if not is_quantized:
        return cache.read(start_token, end_token)

    timing_enabled = bool(
        getattr(cache, "device", torch.device("cpu")).type == "cuda"
        and stats.timing_enabled
    )
    if timing_enabled:
        start_event, end_event, target = _record_cuda_event(stats, "dequantize")
    else:
        started = time.perf_counter()
    value = cache.read(start_token, end_token)
    if timing_enabled:
        end_event.record()
        target.append((start_event, end_event))
    else:
        stats.dequantize_time_s += time.perf_counter() - started
    stats.dequantize_calls += 1
    return value


def _iter_tensors(value: Any, seen: set[int]) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        identity = id(value)
        if identity not in seen:
            seen.add(identity)
            yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_tensors(child, seen)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_tensors(child, seen)


def qvg_memory_breakdown(pipeline) -> QVGMemory:
    """Count physical tensors and their full-BF16 logical equivalent."""
    equivalent = 0
    physical_bf16 = 0
    physical_compressed = 0
    bf16_chunks = 0
    quantized_chunks = 0
    seen: set[int] = set()

    for cache_list in _cache_lists(pipeline):
        for layer in cache_list:
            if layer.get("cache_backend") != "qvg":
                continue
            for cache in (layer["k"], layer["v"]):
                chunk_values = int(cache.batch_size) * int(
                    cache.frame_seq_length
                ) * int(cache.num_heads) * int(cache.head_dim)
                element_size = torch.empty((), dtype=cache.dtype).element_size()
                for state, chunk in zip(cache.chunk_state, cache.chunks):
                    state_value = int(state)
                    if state_value == 1:
                        equivalent += chunk_values * element_size
                        bf16_chunks += 1
                        if chunk is not None and id(chunk) not in seen:
                            seen.add(id(chunk))
                            physical_bf16 += chunk.numel() * chunk.element_size()
                    elif state_value == 2:
                        equivalent += chunk_values * element_size
                        quantized_chunks += 1
                for span in cache.quantized_spans:
                    for tensor in _iter_tensors(span["quant_data"], seen):
                        physical_compressed += (
                            tensor.numel() * tensor.element_size()
                        )

    # Chunk counts above include K/V and all transformer layers. Report the
    # physical count while summary helpers expose a per-cache user-facing count.
    return QVGMemory(
        bf16_equivalent_bytes=int(equivalent),
        physical_bytes=int(physical_bf16 + physical_compressed),
        physical_bf16_bytes=int(physical_bf16),
        physical_compressed_bytes=int(physical_compressed),
        bf16_chunks=int(bf16_chunks),
        quantized_chunks=int(quantized_chunks),
    )


def qvg_resident_memory_bytes(pipeline) -> tuple[int, int]:
    memory = qvg_memory_breakdown(pipeline)
    return memory.bf16_equivalent_bytes, memory.physical_bytes


def qvg_cache_counts(pipeline) -> tuple[int, int]:
    """Return BF16 and quantized chunk counts for one representative K cache."""
    for cache_list in _cache_lists(pipeline):
        if not cache_list:
            continue
        cache = cache_list[0]["k"]
        return (
            sum(int(state) == 1 for state in cache.chunk_state),
            sum(int(state) == 2 for state in cache.chunk_state),
        )
    return 0, 0


def qvg_metrics(pipeline) -> dict[str, Any]:
    config: QVGConfig = pipeline.qvg_config
    stats: QVGStats = pipeline.qvg_stats
    stats.resolve_timing(synchronize=False)
    bf16_chunks, quantized_chunks = qvg_cache_counts(pipeline)
    memory = qvg_memory_breakdown(pipeline)
    return {
        "qvg_quant_type": config.quant_type,
        "qvg_quant_factor": int(config.quant_factor),
        "qvg_num_k_centroids": int(config.num_k_centroids),
        "qvg_num_v_centroids": int(config.num_v_centroids),
        "qvg_kmeans_max_iters": int(config.kmeans_max_iters),
        "qvg_quant_block_size": int(config.quant_block_size),
        "qvg_num_prq_stages": int(config.num_prq_stages),
        "qvg_compression_enabled": bool(
            getattr(config, "compression_enabled", True)
        ),
        # These timers cover the official codec calls only; cache reads,
        # layout conversion, storage and BF16 release are outside this scope.
        "qvg_codec_quantize_time_s": float(stats.quantize_time_s),
        "qvg_codec_dequantize_time_s": float(stats.dequantize_time_s),
        "qvg_quantized_spans": int(stats.quantized_spans),
        # This is the cumulative number of frame chunks compressed during the
        # run.  ``quantized_chunks`` below is the currently resident count in
        # one representative cache and is intentionally reported separately.
        "qvg_quantized_chunks": int(stats.quantized_chunks),
        "qvg_resident_quantized_chunks": int(quantized_chunks),
        "qvg_bf16_chunks": int(bf16_chunks),
        "qvg_physical_bf16_bytes": int(memory.physical_bf16_bytes),
        "qvg_physical_compressed_bytes": int(
            memory.physical_compressed_bytes
        ),
        "qvg_packed_bytes": int(memory.physical_compressed_bytes),
        "qvg_bf16_tail_bytes": int(memory.physical_bf16_bytes),
        "qvg_physical_bytes": int(memory.physical_bytes),
        "qvg_resident_total_kv_bytes": int(memory.physical_bytes),
        "qvg_uncompressed_reference_kv_bytes": int(
            memory.bf16_equivalent_bytes
        ),
        "qvg_effective_kv_bits_per_value": (
            memory.physical_bytes * 8 / max(memory.logical_values, 1)
        ),
        "qvg_compression_ratio": (
            memory.bf16_equivalent_bytes / max(memory.physical_bytes, 1)
        ),
        "upstream_commit": QVG_UPSTREAM_COMMIT,
    }
