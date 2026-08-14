"""
Simulated NVFP4 Quantization Kernels in Triton

NVFP4 (NVIDIA FP4) is a 4-bit floating point format with:
- E2M1: 2 exponent bits, 1 mantissa bit (sign bit is separate)

This module provides triton kernels for simulating FP4 quantization effects
while keeping the data in higher precision formats (bf16/fp32).
"""

import torch
import triton
import triton.language as tl
from termcolor import cprint
from triton.language.extra import libdevice
from typing import Optional, Tuple


# =============================================================================
# Python Helper Functions
# =============================================================================


def compute_percentile_by_sorting(x: torch.Tensor, percentile: float) -> float:
    """
    Compute the percentile value of a tensor using sorting.

    This is a custom implementation that sorts the tensor and picks the value
    at the percentile index, instead of using torch.quantile which can have
    numerical stability issues.

    Args:
        x: Input tensor (will be flattened)
        percentile: Percentile value (0-100)

    Returns:
        The value at the specified percentile
    """
    x_flat = x.flatten().float()
    n = x_flat.numel()

    if n == 0:
        return 0.0

    # Sort the tensor
    sorted_x, _ = torch.sort(x_flat)

    # Compute the index for the percentile
    # percentile=99 means we want the value at 99% of the way through
    idx = int((percentile / 100.0) * (n - 1))
    idx = max(0, min(idx, n - 1))  # Clamp to valid range

    return sorted_x[idx].item()


def percentile_clip_and_scale(
    x: torch.Tensor,
    *,
    percentile: float,
    target_max: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
    """
    Percentile-based clipping and scaling to handle extreme outliers.

    This function:
    - Finds the value at `percentile` of |x|
    - Stores outliers (values beyond that threshold) in `residual`
    - Zeros out those outliers in x (so they won't be quantized)
    - Scales the remaining values so the new max becomes `target_max`

    Returns:
        x_scaled: Tensor after removing outliers and scaling
        residual: Outlier values kept in full precision (or None if no clipping)
        scale_factor: Scale factor used (1.0 if no clipping)
    """
    x_abs = x.abs()
    percentile_value = compute_percentile_by_sorting(x_abs, percentile)
    if percentile_value <= 0:
        return x, None, 1.0

    # Clip values to the percentile threshold and store residual (outliers)
    x_clipped = torch.clamp(x, min=-percentile_value, max=percentile_value)

    clipped_mask = x_clipped != x
    residual = x * clipped_mask
    x_no_outliers = x - residual

    # Scale so that the new max value becomes target_max
    scale_factor = target_max / percentile_value
    x_scaled = x_no_outliers * scale_factor
    return x_scaled, residual, scale_factor


def percentile_unscale_and_add_residual(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    scale_factor: float,
) -> torch.Tensor:
    """
    Inverse of `percentile_clip_and_scale` post-processing:
    - unscale x by scale_factor
    - add residual (outliers) back
    """
    if residual is None:
        return x
    return x / scale_factor + residual


# =============================================================================
# Triton Helper Functions
# =============================================================================


@triton.jit
def _compute_max_representable_value(
    ebit,
    mbit,
    use_all_one_exponent,
    use_all_one_mantissa,
):
    """
    Compute the maximum representable value for FloatExMy format.

    Args:
        ebit: Number of exponent bits
        mbit: Number of mantissa bits
        use_all_one_exponent: If True, max exponent is 2^(e_bit-1), else 2^(e_bit-1)-1
        use_all_one_mantissa: If True, max mantissa is 2-2^(-m_bit), else 2-2^(1-m_bit)

    Returns:
        Maximum representable value as float32
    """
    if use_all_one_exponent:
        repre_max_expo = tl.exp2(tl.exp2((ebit - 1).to(tl.float32)))
    else:
        repre_max_expo = tl.exp2(tl.exp2((ebit - 1).to(tl.float32))) - 1
    if use_all_one_mantissa:
        repre_max_mant = 2.0 - tl.exp2((-mbit).to(tl.float32))
    else:
        repre_max_mant = 2.0 - tl.exp2((1 - mbit).to(tl.float32))
    return repre_max_expo * repre_max_mant


# =============================================================================
# Core Triton Kernels
# =============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}, num_stages=3, num_warps=4),
        # triton.Config({"BLOCK_SIZE": 512}, num_stages=3, num_warps=4),
        # triton.Config({"BLOCK_SIZE": 512}, num_stages=4, num_warps=8),
        # triton.Config({"BLOCK_SIZE": 1024}, num_stages=3, num_warps=8),
        # triton.Config({"BLOCK_SIZE": 1024}, num_stages=4, num_warps=8),
        # triton.Config({"BLOCK_SIZE": 2048}, num_stages=3, num_warps=8),
        # triton.Config({"BLOCK_SIZE": 2048}, num_stages=4, num_warps=8),
        # triton.Config({"BLOCK_SIZE": 4096}, num_stages=3, num_warps=8),
        # triton.Config({"BLOCK_SIZE": 4096}, num_stages=4, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _floatExMy_quantize_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    e_bit,
    m_bit,
    use_all_one_exponent,
    use_all_one_mantissa,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for FloatExMy quantization.

    Quantization to a floating point format with:
    - e_bit: number of exponent bits
    - m_bit: number of mantissa bits
    - use_all_one_exponent: if True, max exponent is 2^(e_bit-1), else 2^(e_bit-1)-1
    - use_all_one_mantissa: if True, max mantissa is 2-2^(-m_bit), else 2-2^(1-m_bit)
    """
    if isinstance(e_bit, tl.constexpr):
        ebit = e_bit.value
    else:
        ebit = e_bit

    if isinstance(m_bit, tl.constexpr):
        mbit = m_bit.value
    else:
        mbit = m_bit

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x = x.to(tl.float32)

    # Compute maximum representable value and clamp input
    repre_max = _compute_max_representable_value(
        ebit, mbit, use_all_one_exponent, use_all_one_mantissa
    )
    x = tl.clamp(x, min=-repre_max, max=repre_max)

    sign = 1 - 2 * libdevice.signbit(x)
    x_abs = tl.abs(x)

    # Compute exponent bounds
    Elow = -tl.exp2((ebit - 1).to(tl.float32)) + 2
    Ehigh = tl.exp2((ebit - 1).to(tl.float32))
    Mhigh = tl.exp2(mbit.to(tl.float32))

    # Extract and clamp exponent
    expo = tl.floor(tl.log2(x_abs))
    expo = tl.clamp(expo, min=Elow, max=Ehigh)

    # Extract and quantize mantissa
    mant = x_abs / tl.exp2(expo)
    mant_int = tl.floor(mant)
    mant_frac = mant - mant_int
    mant_frac = mant_frac * Mhigh
    mant_frac = libdevice.round(mant_frac)
    mant_q = mant_int + mant_frac / Mhigh

    # Reconstruct quantized value
    y = sign * tl.exp2(expo) * mant_q
    y = y.to(x_ptr.dtype.element_ty)

    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def _compute_block_scales_and_divide_kernel(
    x_ptr,
    scale_ptr,
    y_ptr,
    x_stride,
    scale_stride,
    y_stride,
    N: tl.constexpr,
    D: tl.constexpr,
    D2: tl.constexpr,
    num_of_block_per_dim: tl.constexpr,
    num_of_block_per_dim2: tl.constexpr,
    block_size: tl.constexpr,
    max_val,
    do_divide: tl.constexpr,
):
    """
    Compute per-block scaling factors for FP4 quantization.
    Scale = max(|x|) / max_representable_value
    """
    pid = tl.program_id(axis=0)

    offset = tl.arange(0, D2)
    mask = offset < D
    x = tl.load(x_ptr + pid * x_stride + offset, mask=mask)
    x = x.to(tl.float32)

    # Reshape to (num_of_block_per_dim2, block_size)
    x = tl.reshape(x, (num_of_block_per_dim2, block_size))

    x_abs = tl.abs(x)
    max_abs = tl.max(x_abs, axis=1)
    scale = max_abs / max_val
    scale = tl.maximum(scale, 1e-10)

    offset_scale = tl.arange(0, num_of_block_per_dim2)
    mask_scale = offset_scale < num_of_block_per_dim
    tl.store(
        scale_ptr + pid * scale_stride + offset_scale,
        scale.to(scale_ptr.dtype.element_ty),
        mask=mask_scale,
    )

    if do_divide:
        scale = tl.reshape(scale, (num_of_block_per_dim2, 1))
        y = x / scale
        y = tl.reshape(y, (D2,))
        tl.store(y_ptr + pid * y_stride + offset, y, mask=mask)


@triton.jit
def _compute_block_multiply_kernel(
    x_ptr,
    scale_ptr,
    y_ptr,
    x_stride,
    scale_stride,
    y_stride,
    N: tl.constexpr,
    D: tl.constexpr,
    D2: tl.constexpr,
    num_of_block_per_dim: tl.constexpr,
    num_of_block_per_dim2: tl.constexpr,
    block_size: tl.constexpr,
):
    """
    Compute per-block scaling factors for FP4 quantization.
    Scale = max(|x|) / max_representable_value
    """
    pid = tl.program_id(axis=0)

    offset = tl.arange(0, D2)
    mask = offset < D
    x = tl.load(x_ptr + pid * x_stride + offset, mask=mask)
    x = x.to(tl.float32)

    # Reshape to (num_of_block_per_dim2, block_size)
    x = tl.reshape(x, (num_of_block_per_dim2, block_size))

    offset_scale = tl.arange(0, num_of_block_per_dim2)
    mask_scale = offset_scale < num_of_block_per_dim

    scale = tl.load(scale_ptr + pid * scale_stride + offset_scale, mask=mask_scale)
    scale = scale.to(tl.float32)

    # Reshape to (num_of_block_per_dim2, 1)
    scale = tl.reshape(scale, (num_of_block_per_dim2, 1))

    # Multiply
    y = x * scale

    # Reshape to (D2,)
    y = tl.reshape(y, (D2,))

    tl.store(y_ptr + pid * y_stride + offset, y, mask=mask)


@triton.jit
def _compute_block_divide_kernel(
    x_ptr,
    scale_ptr,
    y_ptr,
    x_stride,
    scale_stride,
    y_stride,
    N: tl.constexpr,
    D: tl.constexpr,
    D2: tl.constexpr,
    num_of_block_per_dim: tl.constexpr,
    num_of_block_per_dim2: tl.constexpr,
    block_size: tl.constexpr,
):
    """
    Compute per-block scaling factors for FP4 quantization.
    Scale = max(|x|) / max_representable_value
    """
    pid = tl.program_id(axis=0)

    offset = tl.arange(0, D2)
    mask = offset < D
    x = tl.load(x_ptr + pid * x_stride + offset, mask=mask)
    x = x.to(tl.float32)

    # Reshape to (num_of_block_per_dim2, block_size)
    x = tl.reshape(x, (num_of_block_per_dim2, block_size))

    offset_scale = tl.arange(0, num_of_block_per_dim2)
    mask_scale = offset_scale < num_of_block_per_dim

    scale = tl.load(scale_ptr + pid * scale_stride + offset_scale, mask=mask_scale)
    scale = scale.to(tl.float32)

    # Reshape to (num_of_block_per_dim2, 1)
    scale = tl.reshape(scale, (num_of_block_per_dim2, 1))

    # Multiply
    y = x / scale

    # Reshape to (D2,)
    y = tl.reshape(y, (D2,))

    tl.store(y_ptr + pid * y_stride + offset, y, mask=mask)


# =============================================================================
# Python API Functions
# =============================================================================


def floatExMy_quantize_triton(
    x: torch.Tensor,
    e_bit: int,
    m_bit: int,
    use_all_one_exponent: bool = False,
    use_all_one_mantissa: bool = False,
) -> torch.Tensor:
    """
    Quantize tensor to FloatExMy format using Triton.

    Args:
        x: Input tensor (bf16 or fp32)
        e_bit: Number of exponent bits
        m_bit: Number of mantissa bits
        use_all_one_exponent: If True, max exponent is 2^(e_bit-1), else 2^(e_bit-1)-1
        use_all_one_mantissa: If True, max mantissa is 2-2^(-m_bit), else 2-2^(1-m_bit)

    Returns:
        Quantized tensor in same dtype as input
    """
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    y = torch.zeros_like(x)

    if x.dtype in [torch.bfloat16, torch.float32]:
        _floatExMy_quantize_kernel[grid](
            x, y, n_elements, e_bit, m_bit, use_all_one_exponent, use_all_one_mantissa
        )
    else:
        raise NotImplementedError(
            f"Unsupported dtype {x.dtype} for float quantization triton"
        )

    return y


def compute_block_scales_and_divide(
    x: torch.Tensor, block_size: int, max_val: float, do_divide: bool = True
) -> torch.Tensor:
    """
    Compute per-block scaling factors for scaled FP4 quantization.

    Args:
        x: Input tensor
        block_size: Number of elements per scaling block
        max_val: Maximum representable value in the target format
        do_divide: If True, divide the input by the scaling factors. If false, only compute the scaling factors.

    Returns:
        Scale tensor of shape (n_blocks,)
    """
    assert x.is_contiguous(), "x must be contiguous"

    input_shape_until_last_dim = x.shape[:-1]
    x_2d = x.view(-1, x.shape[-1])
    N, D = x_2d.shape

    num_of_block_per_dim = triton.cdiv(D, block_size)
    num_of_block_per_dim2 = triton.next_power_of_2(num_of_block_per_dim)
    D2 = triton.next_power_of_2(num_of_block_per_dim * block_size)

    y = torch.zeros_like(x_2d)
    scales = torch.zeros(N, num_of_block_per_dim, device=x.device, dtype=x.dtype)
    grid = lambda meta: (N,)
    _compute_block_scales_and_divide_kernel[grid](
        x_2d,
        scales,
        y,
        x_2d.stride(0),
        scales.stride(0),
        y.stride(0),
        N,
        D,
        D2,
        num_of_block_per_dim,
        num_of_block_per_dim2,
        block_size,
        max_val,
        do_divide,
    )

    # Reshape to the original shape
    output_scale_shape = input_shape_until_last_dim + (num_of_block_per_dim,)
    scales = scales.view(output_scale_shape)
    y = y.view(x.shape)

    if do_divide:
        return y, scales
    else:
        return scales


def compute_block_multiply(
    x: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    Compute per-block multiplication for FP4 quantization.
    """

    x_2d = x.view(-1, x.shape[-1])
    scales_2d = scales.view(-1, scales.shape[-1])
    N, D = x_2d.shape

    num_of_block_per_dim = triton.cdiv(D, block_size)
    num_of_block_per_dim2 = triton.next_power_of_2(num_of_block_per_dim)
    D2 = triton.next_power_of_2(num_of_block_per_dim * block_size)

    y = torch.zeros_like(x_2d)
    grid = lambda meta: (N,)
    _compute_block_multiply_kernel[grid](
        x_2d,
        scales_2d,
        y,
        x_2d.stride(0),
        scales_2d.stride(0),
        y.stride(0),
        N,
        D,
        D2,
        num_of_block_per_dim,
        num_of_block_per_dim2,
        block_size,
    )
    y = y.view(x.shape)
    return y


def compute_block_divide(
    x: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    Compute per-block division for FP4 quantization.
    """

    x_2d = x.view(-1, x.shape[-1])
    scales_2d = scales.view(-1, scales.shape[-1])
    N, D = x_2d.shape

    num_of_block_per_dim = triton.cdiv(D, block_size)
    num_of_block_per_dim2 = triton.next_power_of_2(num_of_block_per_dim)
    D2 = triton.next_power_of_2(num_of_block_per_dim * block_size)

    y = torch.zeros_like(x_2d)
    grid = lambda meta: (N,)
    _compute_block_divide_kernel[grid](
        x_2d,
        scales_2d,
        y,
        x_2d.stride(0),
        scales_2d.stride(0),
        y.stride(0),
        N,
        D,
        D2,
        num_of_block_per_dim,
        num_of_block_per_dim2,
        block_size,
    )

    y = y.view(x.shape)
    return y


def _nvfp4_e2m1_quantize_triton(
    x: torch.Tensor,
    block_size: int = 16,
    return_mid_result: bool = False,
    use_ipython: bool = False,
) -> torch.Tensor:
    """
    Quantize tensor to FP4 E2M1 format using Triton.

    Representable values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}

    Args:
        x: Input tensor (bf16 or fp32)

    Returns:
        Quantized tensor in same dtype as input
    """
    # Check max representable value
    e4m3_max = torch.finfo(torch.float8_e4m3fn).max
    e4m3_min = 1 / 512
    nvfp4_max = 6.0
    data_max = x.abs().max().item()
    if data_max > nvfp4_max * e4m3_max:
        cprint(
            f"Warning: Data max ({data_max:.4f}) exceeds max representable value "
            f"(NVFP4_max={nvfp4_max} * E4M3_max={e4m3_max} = {nvfp4_max * e4m3_max:.4f}). "
            f"Values will be clipped, causing precision loss.",
            "yellow",
        )

    scales = compute_block_scales_and_divide(x, block_size, nvfp4_max, do_divide=False)
    scales_e4m3 = floatExMy_quantize_triton(
        scales, e_bit=4, m_bit=3, use_all_one_exponent=True, use_all_one_mantissa=False
    )
    # Clip values to E4M3 min and max
    scales_e4m3 = torch.clamp(scales_e4m3, min=e4m3_min, max=e4m3_max)

    y = compute_block_divide(x, scales_e4m3, block_size)
    z = floatExMy_quantize_triton(
        y, e_bit=2, m_bit=1, use_all_one_exponent=True, use_all_one_mantissa=True
    )
    w = compute_block_multiply(z, scales_e4m3, block_size)

    if use_ipython:
        import IPython

        IPython.embed()

    if return_mid_result:
        return w, (z, y, scales, scales_e4m3)
    else:
        return w


def nvfp4_e2m1_quantize_triton(
    x: torch.Tensor,
    block_size: int = 16,
    return_mid_result: bool = False,
    use_ipython: bool = False,
    use_percentile_clipping: bool = False,
    percentile: float = 99.0,
) -> torch.Tensor:
    """
    Quantize tensor to NVFP4 E2M1 format using Triton.

    Args:
        x: Input tensor (bf16 or fp32)
        block_size: Number of elements per scaling block
        return_mid_result: If True, return intermediate results
        use_ipython: If True, embed IPython shell for debugging
        use_percentile_clipping: If True, clip values based on percentile threshold
            to avoid extreme outliers affecting quantization
        percentile: The percentile threshold for clipping (default: 99.0 for top 1%)

    Returns:
        Quantized tensor in same dtype as input
    """
    TARGET_MAX = 448 * 6
    
    # Percentile-based clipping and scaling to handle extreme outliers
    if use_percentile_clipping:
        x, residual, scale_factor = percentile_clip_and_scale(
            x,
            percentile=percentile,
            target_max=TARGET_MAX,
        )

    result = _nvfp4_e2m1_quantize_triton(
        x,
        block_size,
        return_mid_result,
        use_ipython,
    )

    # Add back the residual after quantization (scale back first)
    if use_percentile_clipping and residual is not None:
        if return_mid_result:
            quantized, mid_results = result
            # Scale back the quantized result and add residual
            quantized = percentile_unscale_and_add_residual(
                quantized, residual=residual, scale_factor=scale_factor
            )
            return quantized, mid_results
        else:
            # Scale back the quantized result and add residual
            result = percentile_unscale_and_add_residual(
                result, residual=residual, scale_factor=scale_factor
            )

    return result


def fp8_e4m3_quantize_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Quantize tensor to FP8 E4M3 format using Triton.
    """
    raise NotImplementedError("FP8 E4M3 quantization is not implemented")


# =============================================================================
# Block-wise INTx Quantization
# =============================================================================


def get_intx_max_value(num_bits: int) -> int:
    """Get the maximum representable value for INTx format."""
    return (1 << (num_bits - 1)) - 1  # 2^(n-1) - 1


def _blockwise_intx_quantize_triton(
    x: torch.Tensor,
    num_bits: int = 4,
    block_size: int = 16,
    return_mid_result: bool = False,
    use_ipython: bool = False,
) -> torch.Tensor:
    """
    Block-wise INTx quantization with E4M3 scale factors.
    """
    # Check max representable value
    int_max = get_intx_max_value(num_bits)
    e4m3_max = torch.finfo(torch.float8_e4m3fn).max
    e4m3_min = 1 / 512
    data_max = x.abs().max().item()
    if data_max > e4m3_max * int_max:
        cprint(
            f"Warning: Data max ({data_max:.4f}) exceeds max representable value "
            f"(E4M3_max={e4m3_max} * INT{num_bits}_max={int_max} = {e4m3_max * int_max:.4f}). "
            f"Values will be clipped, causing precision loss.",
            "yellow",
        )

    scales = compute_block_scales_and_divide(x, block_size, int_max, do_divide=False)
    scales_e4m3 = floatExMy_quantize_triton(
        scales, e_bit=4, m_bit=3, use_all_one_exponent=True, use_all_one_mantissa=False
    )
    # Clip values to E4M3 min and max
    scales_e4m3 = torch.clamp(scales_e4m3, min=e4m3_min, max=e4m3_max)

    # Divide input by E4M3 scales
    y = compute_block_divide(x, scales_e4m3, block_size)

    # Round to nearest integer and clamp to INT range
    z = torch.clamp(torch.round(y), min=-int_max, max=int_max)

    # Multiply back by scales to get dequantized values
    w = compute_block_multiply(z, scales_e4m3, block_size)

    if use_ipython:
        import IPython

        IPython.embed()

    if return_mid_result:
        return w, (z, y, scales, scales_e4m3)
    else:
        return w


def blockwise_intx_quantize_triton(
    x: torch.Tensor,
    num_bits: int = 4,
    block_size: int = 16,
    return_mid_result: bool = False,
    use_ipython: bool = False,
    use_percentile_clipping: bool = False,
    percentile: float = 99.0,
) -> torch.Tensor:
    """
    Block-wise INTx quantization with E4M3 scale factors and optional percentile clipping.

    Args:
        x: Input tensor (bf16 or fp32)
        num_bits: Number of bits for quantization (2, 3, 4, or 8)
        block_size: Number of elements per scaling block
        return_mid_result: If True, return intermediate results
        use_ipython: If True, embed IPython shell for debugging
        use_percentile_clipping: If True, clip values based on percentile threshold
            to avoid extreme outliers affecting quantization
        percentile: The percentile threshold for clipping (default: 99.0 for top 1%)

    Returns:
        Quantized tensor in same dtype as input
    """
    int_max = get_intx_max_value(num_bits)
    e4m3_max = torch.finfo(torch.float8_e4m3fn).max
    TARGET_MAX = e4m3_max * int_max

    # Percentile-based clipping and scaling to handle extreme outliers
    if use_percentile_clipping:
        x, residual, scale_factor = percentile_clip_and_scale(
            x,
            percentile=percentile,
            target_max=TARGET_MAX,
        )

    result = _blockwise_intx_quantize_triton(
        x,
        num_bits=num_bits,
        block_size=block_size,
        return_mid_result=return_mid_result,
        use_ipython=use_ipython,
    )

    # Add back the residual after quantization (scale back first)
    if use_percentile_clipping and residual is not None:
        if return_mid_result:
            quantized, mid_results = result
            # Scale back the quantized result and add residual
            quantized = percentile_unscale_and_add_residual(
                quantized, residual=residual, scale_factor=scale_factor
            )
            return quantized, mid_results
        else:
            # Scale back the quantized result and add residual
            result = percentile_unscale_and_add_residual(
                result, residual=residual, scale_factor=scale_factor
            )

    return result


def blockwise_int2_quantize_triton(
    x: torch.Tensor,
    block_size: int = 16,
    return_mid_result: bool = False,
    use_ipython: bool = False,
    use_percentile_clipping: bool = False,
    percentile: float = 99.0,
) -> torch.Tensor:
    """
    Block-wise INT2 quantization with E4M3 scale factors.
    """
    return blockwise_intx_quantize_triton(
        x,
        num_bits=2,
        block_size=block_size,
        return_mid_result=return_mid_result,
        use_ipython=use_ipython,
        use_percentile_clipping=use_percentile_clipping,
        percentile=percentile,
    )


def blockwise_int3_quantize_triton(
    x: torch.Tensor,
    block_size: int = 16,
    return_mid_result: bool = False,
    use_ipython: bool = False,
    use_percentile_clipping: bool = False,
    percentile: float = 99.0,
) -> torch.Tensor:
    """
    Block-wise INT3 quantization with E4M3 scale factors.
    """
    return blockwise_intx_quantize_triton(
        x,
        num_bits=3,
        block_size=block_size,
        return_mid_result=return_mid_result,
        use_ipython=use_ipython,
        use_percentile_clipping=use_percentile_clipping,
        percentile=percentile,
    )


def blockwise_int4_quantize_triton(
    x: torch.Tensor,
    block_size: int = 16,
    return_mid_result: bool = False,
    use_ipython: bool = False,
    use_percentile_clipping: bool = False,
    percentile: float = 99.0,
) -> torch.Tensor:
    """
    Block-wise INT4 quantization with E4M3 scale factors.
    """
    return blockwise_intx_quantize_triton(
        x,
        num_bits=4,
        block_size=block_size,
        return_mid_result=return_mid_result,
        use_ipython=use_ipython,
        use_percentile_clipping=use_percentile_clipping,
        percentile=percentile,
    )


def blockwise_int8_quantize_triton(
    x: torch.Tensor,
    block_size: int = 16,
    return_mid_result: bool = False,
    use_ipython: bool = False,
    use_percentile_clipping: bool = False,
    percentile: float = 99.0,
) -> torch.Tensor:
    """
    Block-wise INT8 quantization with E4M3 scale factors.
    """
    return blockwise_intx_quantize_triton(
        x,
        num_bits=8,
        block_size=block_size,
        return_mid_result=return_mid_result,
        use_ipython=use_ipython,
        use_percentile_clipping=use_percentile_clipping,
        percentile=percentile,
    )


# =============================================================================
# Reference Implementations (for testing)
# =============================================================================


def floatExMy_quantize_pytorch(
    x,
    e_bit,
    m_bit,
    use_all_one_exponent=False,
    use_all_one_mantissa=False,
    IPython=False,
):
    """
    PyTorch implementation of FloatExMy quantization.
    Args:
        x: Input tensor
        e_bit: Number of exponent bits
        m_bit: Number of mantissa bits
        use_all_one_exponent: If False, when exponent is all one, the value represents NaN (E5M2 or BFloat16). Otherwise, it represents the maximum representable value.
        use_all_one_mantissa: If False, when exponent and mantissa are all one, the value represents NaN (E4M3). Otherwise, it represents the maximum representable value.
        IPython: If True, embed the IPython shell

    Returns:
        Quantized tensor
    """
    # Clamp the input to the maximum representable value
    if use_all_one_exponent:
        repre_max_expo = 2 ** (2 ** (e_bit - 1))
    else:
        repre_max_expo = 2 ** (2 ** (e_bit - 1)) - 1
    if use_all_one_mantissa:
        repre_max_mant = 2 - 2 ** (-m_bit)
    else:
        repre_max_mant = 2 - 2 ** (1 - m_bit)
    repre_max = repre_max_expo * repre_max_mant
    x = torch.clamp(x, min=-repre_max, max=repre_max)

    sign, x_abs = x.sign(), x.abs()
    Elow, Ehigh, Mhigh = -(2 ** (e_bit - 1)) + 2, 2 ** (e_bit - 1), 2**m_bit
    expo = torch.floor(torch.log2(x_abs))
    expo = torch.clamp(expo, min=Elow, max=Ehigh)
    mant = x_abs / torch.exp2(expo)

    mant_int = torch.floor(mant)
    mant_frac = mant - mant_int
    mant_frac = mant_frac * Mhigh
    mant_frac = torch.round(mant_frac)
    mant_q = mant_int + mant_frac / Mhigh
    y = sign * (2**expo) * mant_q
    y = y.to(x)

    return y
