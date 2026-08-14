"""
Uncompress and dequantize for triton-xxx quant types (triton-nstages-kmeans-int2/int4).
Mirrors the structure of compress.py: get_dequantize_fn + uncompress_kv_cache.
"""

import re
import torch

from .compress import get_quantize_type, QuantizeFunctions
from .sim.quant.quantize_config import QuantizeConfig
from .functions import triton_prq_dequantize_tensor


########################################################
# Entrypoints (mirror compress.py)
########################################################

def extract_num_bits(quant_config: QuantizeConfig):
    m = re.search(r'int(\d+)', quant_config.quant_type)
    if m is None:
        raise ValueError(f"Cannot identify num_bits from {quant_config.quant_type}")
    return int(m.group(1))

def uncompress_kv_cache(
    k_cache: torch.Tensor | dict,
    v_cache: torch.Tensor | dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Uncompress (and dequantize) one layer of KV cache, mirroring compress_kv_cache.

    Same I/O pattern as compress_kv_cache:
      - compress_kv_cache(k, v, ...) -> (k_quant, v_quant)   # one layer
      - uncompress_kv_cache(k_cached, v_cached, ...) -> (k, v)  # one layer

    Args:
        k: Cached K for this layer (tensor for non-triton, packed_state dict for triton-xxx).
        v: Cached V for this layer (tensor for non-triton, packed_state dict for triton-xxx).
        quant_type: Same quant_type used when compressing.
        quant_config: Same config used when compressing.
        dequantize_fn: From get_dequantize_fn(quant_type, quant_config).
        device: If set, move packed state to this device before dequantizing.
        output_dtype: Dtype for reconstructed K/V tensors.

    Returns:
        (k_tensor, v_tensor) for this layer, ready for attention.
    """

    return uncompress_single_kv_cache(k_cache), uncompress_single_kv_cache(v_cache)


def uncompress_single_kv_cache(cache: torch.Tensor | dict) -> torch.Tensor:
    """Decompress one packed K *or* V state.

    ``ChunkedKVCache`` owns K and V independently, so passing the same packed
    object twice to the pair API would perform the same decompression twice.
    This single-state entry point keeps that cache read path explicit and
    avoids doubling the Triton work.
    """
    if not isinstance(cache, dict):
        return cache

    quant_config = cache["info"]["quant_config"]
    output_dtype = cache["info"]["output_dtype"]
    quantize_type = get_quantize_type(quant_config.quant_type)
    num_bits = extract_num_bits(quant_config)

    if quantize_type in (QuantizeFunctions.TRITON_PRQ, QuantizeFunctions.TRITON_PRQ_CLIP):
        return triton_prq_dequantize_tensor(
            cache,
            quant_config.quant_block_size,
            num_bits,
            output_dtype=output_dtype,
        )
    raise ValueError(f"Unsupported QVG quantization type: {quant_config.quant_type}")
