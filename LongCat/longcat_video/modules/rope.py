# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl

from ..common import (
    SCALE_MIN_THRES,
    convert_fp8_dtype_torch_to_triton,
    flatten_if_batched,
)
from ..quant.dequant import _fp8_dequant_pg_1d
from ..quant.quant import _fp8_quant_pg_1d
from ..quant.div import _fp8_div_1d

################################################################################
# Per-group quant Input, High-precision Output
################################################################################


@triton.jit
def _fp8_rope_pg2hp_forward_kernel(
    q_x_ptr,
    s_x_ptr,
    real_ptr,
    imag_ptr,
    y_ptr,
    SEQ_LEN: tl.constexpr,
    N: tl.constexpr,
    N2: tl.constexpr,
    SN: tl.constexpr,
    SN2: tl.constexpr,
    QB: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_HEADS2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_HALF: tl.constexpr,
    fp8_max,
    quant_type: tl.constexpr,
    q_x_stride_0,
    s_x_stride_0,
    real_stride_0,
    real_stride_1,
    imag_stride_0,
    imag_stride_1,
    y_stride_0,
    SCALE_MIN_THRES: tl.constexpr,
):
    row_id = tl.program_id(0)
    batch_id = row_id // SEQ_LEN
    seq_id = row_id % SEQ_LEN

    cols = tl.arange(0, N2)
    mask = cols < N
    scale_cols = tl.arange(0, SN2)
    scale_mask = scale_cols < SN
    head_half_cols = tl.arange(0, HEAD_DIM_HALF)
    head_half_mask = head_half_cols < HEAD_DIM

    # Shift the pointer to the current row
    q_x_ptr += row_id * q_x_stride_0
    s_x_ptr += row_id * s_x_stride_0
    real_ptr += seq_id * real_stride_0
    imag_ptr += seq_id * imag_stride_0
    y_ptr += row_id * y_stride_0

    # Load input
    q_x = tl.load(q_x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    s_x = tl.load(s_x_ptr + scale_cols, mask=scale_mask, other=0.0).to(tl.float32)
    real = tl.load(
        real_ptr + head_half_cols * real_stride_1, mask=head_half_mask, other=0.0
    ).to(tl.float32)
    imag = tl.load(
        imag_ptr + head_half_cols * imag_stride_1, mask=head_half_mask, other=0.0
    ).to(tl.float32)

    # Dequantize
    x = _fp8_dequant_pg_1d(q_x, s_x, quant_type, SCALE_MIN_THRES, N2, SN2, QB)
    x = tl.reshape(x, (N2))

    # Apply RoPE
    x = x.reshape(NUM_HEADS2, HEAD_DIM_HALF, 2)
    x_first, x_second = tl.split(x)

    # Forward: y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    y_first = x_first * real - x_second * imag
    y_second = x_second * real + x_first * imag

    y = tl.interleave(y_first, y_second)
    y = tl.reshape(y, (N2))

    tl.store(y_ptr + cols, y, mask=mask)


def fp8_rope_pg2hp_fwd(
    q_x, s_x, freqs_cis, num_heads, head_dim, rope_config, block_size=None
):
    """
    Forward pass of the RoPE.

    Args:
        q_x (torch.Tensor): Input tensor, FP8 groupwise-quantized. (B, N, H * D)
        s_x (torch.Tensor): Input scale tensor. (B, N, H * D // QB)
        freqs_cis (torch.Tensor): Frequency tensor. (N, D // 2)

    Returns:
        x_out (torch.Tensor): Output tensor. High-precision. (B, N, H * D)
    """
    float8_dtype = rope_config.float8_dtype
    quant_type = rope_config.quant_type
    QB = block_size if block_size is not None else rope_config.block_size

    # Change batched 3D input to 2D
    [q_x, s_x], batched, BS = flatten_if_batched(q_x, s_x)

    freqs_real, freqs_imag = freqs_cis.real, freqs_cis.imag
    assert (
        q_x.is_contiguous() and s_x.is_contiguous()
    ), "Input tensors must be contiguous"

    M, N = q_x.shape
    SEQ_LEN = M // BS
    SN = N // QB
    assert SN == s_x.shape[-1]

    N2 = triton.next_power_of_2(N)
    SN2 = N2 // QB

    NUM_HEADS = num_heads
    NUM_HEADS2 = triton.next_power_of_2(NUM_HEADS)

    HEAD_DIM = head_dim
    HEAD_DIM_HALF = HEAD_DIM // 2
    assert HEAD_DIM in [64, 128, 256], "Head dim should be 64, 128, or 256"

    # Allocate output
    y = torch.empty((M, N), dtype=torch.bfloat16, device=q_x.device)

    # heuristics for number of warps
    num_warps = 8
    fp8MaxValue = torch.finfo(float8_dtype).max

    # Launch kernel. Here it must be a 1D grid, other wise will raise: triton error [cuda]: invalid argument
    _fp8_rope_pg2hp_forward_kernel[(BS * SEQ_LEN,)](
        q_x,
        s_x,
        freqs_real,
        freqs_imag,
        y,
        SEQ_LEN,
        N,
        N2,
        SN,
        SN2,
        QB,
        NUM_HEADS,
        NUM_HEADS2,
        HEAD_DIM,
        HEAD_DIM_HALF,
        fp8MaxValue,
        quant_type,
        q_x.stride(0),
        s_x.stride(0),
        freqs_real.stride(0),
        freqs_real.stride(1),
        freqs_imag.stride(0),
        freqs_imag.stride(1),
        y.stride(0),
        SCALE_MIN_THRES,
        num_warps=num_warps,
    )

    if batched:
        y = y.reshape(BS, -1, y.shape[-1])

    return y


################################################################################
# High-precision Input, Per-group quant Output
################################################################################


@triton.jit
def _fp8_rope_hp2pg_forward_kernel(
    x_ptr,
    real_ptr,
    imag_ptr,
    q_y_ptr,
    s_y_ptr,
    SEQ_LEN: tl.constexpr,
    N: tl.constexpr,
    N2: tl.constexpr,
    SN: tl.constexpr,
    SN2: tl.constexpr,
    QB: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_HEADS2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_HALF: tl.constexpr,
    fp8_max,
    quant_type: tl.constexpr,
    x_stride_0,
    real_stride_0,
    real_stride_1,
    imag_stride_0,
    imag_stride_1,
    q_y_stride_0,
    s_y_stride_0,
    SCALE_MIN_THRES: tl.constexpr,
):
    row_id = tl.program_id(0)
    batch_id = row_id // SEQ_LEN
    seq_id = row_id % SEQ_LEN

    cols = tl.arange(0, N2)
    mask = cols < N
    scale_cols = tl.arange(0, SN2)
    scale_mask = scale_cols < SN
    head_half_cols = tl.arange(0, HEAD_DIM_HALF)
    head_half_mask = head_half_cols < HEAD_DIM

    # Shift the pointer to the current row
    x_ptr += row_id * x_stride_0
    real_ptr += seq_id * real_stride_0
    imag_ptr += seq_id * imag_stride_0
    q_y_ptr += row_id * q_y_stride_0
    s_y_ptr += row_id * s_y_stride_0

    # Load input
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    real = tl.load(
        real_ptr + head_half_cols * real_stride_1, mask=head_half_mask, other=0.0
    ).to(tl.float32)
    imag = tl.load(
        imag_ptr + head_half_cols * imag_stride_1, mask=head_half_mask, other=0.0
    ).to(tl.float32)

    # Apply RoPE
    x = x.reshape(NUM_HEADS2, HEAD_DIM_HALF, 2)
    x_first, x_second = tl.split(x)

    # Forward: y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    y_first = x_first * real - x_second * imag
    y_second = x_second * real + x_first * imag

    y = tl.interleave(y_first, y_second)
    y = tl.reshape(y, (N2))

    # Quantize
    y = tl.reshape(y, (SN2, QB))
    q_y, s_y = _fp8_quant_pg_1d(y, fp8_max, quant_type, SCALE_MIN_THRES, N2, SN2, QB)

    # Write output
    q_y = q_y.to(q_y.type.element_ty)
    s_y = s_y.to(s_y.type.element_ty)
    tl.store(q_y_ptr + cols, q_y, mask=mask)
    tl.store(s_y_ptr + scale_cols, s_y, mask=scale_mask)


def fp8_rope_hp2pg_fwd(x, freqs_cis, num_heads, head_dim, rope_config, block_size=None):
    """
    Forward pass of the RoPE.

    Args:
        x (torch.Tensor): Input tensor, high-precision. (B, N, H * D)
        freqs_cis (torch.Tensor): Frequency tensor. (N, D // 2)

    Returns:
        q_y (torch.Tensor): Output tensor. FP8 groupwise-quantized. (B, N, H * D). Usually the group-size is the same as the head-dim.
        s_y (torch.Tensor): Output scale tensor. (B, N, H * D // QB)
    """
    float8_dtype = rope_config.float8_dtype
    quant_type = rope_config.quant_type
    QB = block_size if block_size is not None else rope_config.block_size

    # Change batched 3D input to 2D
    [x], batched, BS = flatten_if_batched(x)

    freqs_real, freqs_imag = freqs_cis.real, freqs_cis.imag
    assert x.is_contiguous(), "Input tensors must be contiguous"

    M, N = x.shape
    SEQ_LEN = M // BS
    SN = N // QB

    N2 = triton.next_power_of_2(N)
    SN2 = N2 // QB

    NUM_HEADS = num_heads
    NUM_HEADS2 = triton.next_power_of_2(NUM_HEADS)

    HEAD_DIM = head_dim
    HEAD_DIM_HALF = HEAD_DIM // 2
    assert HEAD_DIM in [64, 128, 256], "Head dim should be 64, 128, or 256"

    # Allocate output
    q_y = torch.empty((M, N), dtype=float8_dtype, device=x.device)
    s_y = torch.empty((M, SN), dtype=torch.bfloat16, device=x.device)

    # heuristics for number of warps
    num_warps = 8
    fp8MaxValue = torch.finfo(float8_dtype).max

    # Launch kernel. Here it must be a 1D grid, other wise will raise: triton error [cuda]: invalid argument
    _fp8_rope_hp2pg_forward_kernel[(BS * SEQ_LEN,)](
        x,
        freqs_real,
        freqs_imag,
        q_y,
        s_y,
        SEQ_LEN,
        N,
        N2,
        SN,
        SN2,
        QB,
        NUM_HEADS,
        NUM_HEADS2,
        HEAD_DIM,
        HEAD_DIM_HALF,
        fp8MaxValue,
        quant_type,
        x.stride(0),
        freqs_real.stride(0),
        freqs_real.stride(1),
        freqs_imag.stride(0),
        freqs_imag.stride(1),
        q_y.stride(0),
        s_y.stride(0),
        SCALE_MIN_THRES,
        num_warps=num_warps,
    )

    if batched:
        q_y = q_y.reshape(BS, -1, q_y.shape[-1])
        s_y = s_y.reshape(BS, -1, s_y.shape[-1])

    return q_y, s_y


################################################################################
# High-precision Input, Per-tensor quant Output
################################################################################


@triton.jit
def _fp8_rope_hp2pt_forward_kernel(
    x_ptr,
    real_ptr,
    imag_ptr,
    q_y_ptr,
    s_y_ptr,
    SEQ_LEN: tl.constexpr,
    N: tl.constexpr,
    N2: tl.constexpr,
    SN: tl.constexpr,
    SN2: tl.constexpr,
    QB: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_HEADS2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_HALF: tl.constexpr,
    fp8_max,
    quant_type: tl.constexpr,
    x_stride_0,
    real_stride_0,
    real_stride_1,
    imag_stride_0,
    imag_stride_1,
    q_y_stride_0,
    SCALE_MIN_THRES: tl.constexpr,
):
    row_id = tl.program_id(0)
    batch_id = row_id // SEQ_LEN
    seq_id = row_id % SEQ_LEN

    cols = tl.arange(0, N2)
    mask = cols < N
    scale_cols = tl.arange(0, SN2)
    scale_mask = scale_cols < SN
    head_half_cols = tl.arange(0, HEAD_DIM_HALF)
    head_half_mask = head_half_cols < HEAD_DIM

    # Shift the pointer to the current row
    x_ptr += row_id * x_stride_0
    real_ptr += seq_id * real_stride_0
    imag_ptr += seq_id * imag_stride_0
    q_y_ptr += row_id * q_y_stride_0

    # Load input
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    real = tl.load(
        real_ptr + head_half_cols * real_stride_1, mask=head_half_mask, other=0.0
    ).to(tl.float32)
    imag = tl.load(
        imag_ptr + head_half_cols * imag_stride_1, mask=head_half_mask, other=0.0
    ).to(tl.float32)
    s_y = tl.load(s_y_ptr).to(tl.float32)

    # Apply RoPE
    x = x.reshape(NUM_HEADS2, HEAD_DIM_HALF, 2)
    x_first, x_second = tl.split(x)

    # Forward: y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    y_first = x_first * real - x_second * imag
    y_second = x_second * real + x_first * imag

    y = tl.interleave(y_first, y_second)
    y = tl.reshape(y, (N2))

    # Quantize
    q_y, s_y = _fp8_div_1d(y, s_y, quant_type, N2)

    # Write output
    q_y = q_y.to(q_y.type.element_ty)
    tl.store(q_y_ptr + cols, q_y, mask=mask)


def fp8_rope_hp2pt_fwd(
    x, s_y, freqs_cis, num_heads, head_dim, rope_config, block_size=None
):
    """
    Forward pass of the RoPE. Per-tensor quant. We shall compute the scale factor during prefilling and use this kernel during generation.

    Args:
        x (torch.Tensor): Input tensor, high-precision. (B, N, H * D)
        s_y Optional(torch.Tensor): Scale tensor. (1,)
        freqs_cis (torch.Tensor): Frequency tensor. (N, D // 2)

    Returns:
        q_y (torch.Tensor): Output tensor. FP8 per-tensor quantized. (B, N, H * D)
        s_y (torch.Tensor): Output scale tensor. (1,)
    """
    float8_dtype = rope_config.float8_dtype
    quant_type = rope_config.quant_type
    QB = block_size if block_size is not None else rope_config.block_size

    # Change batched 3D input to 2D
    [x], batched, BS = flatten_if_batched(x)

    freqs_real, freqs_imag = freqs_cis.real, freqs_cis.imag
    assert x.is_contiguous(), "Input tensors must be contiguous"

    M, N = x.shape
    SEQ_LEN = M // BS
    SN = N // QB

    N2 = triton.next_power_of_2(N)
    SN2 = N2 // QB

    NUM_HEADS = num_heads
    NUM_HEADS2 = triton.next_power_of_2(NUM_HEADS)

    HEAD_DIM = head_dim
    HEAD_DIM_HALF = HEAD_DIM // 2
    assert HEAD_DIM in [64, 128, 256], "Head dim should be 64, 128, or 256"

    # Allocate output
    q_y = torch.empty((M, N), dtype=float8_dtype, device=x.device)

    # heuristics for number of warps
    num_warps = 8
    fp8MaxValue = torch.finfo(float8_dtype).max

    # Launch kernel. Here it must be a 1D grid, other wise will raise: triton error [cuda]: invalid argument
    _fp8_rope_hp2pt_forward_kernel[(BS * SEQ_LEN,)](
        x,
        freqs_real,
        freqs_imag,
        q_y,
        s_y,
        SEQ_LEN,
        N,
        N2,
        SN,
        SN2,
        QB,
        NUM_HEADS,
        NUM_HEADS2,
        HEAD_DIM,
        HEAD_DIM_HALF,
        fp8MaxValue,
        quant_type,
        x.stride(0),
        freqs_real.stride(0),
        freqs_real.stride(1),
        freqs_imag.stride(0),
        freqs_imag.stride(1),
        q_y.stride(0),
        SCALE_MIN_THRES,
        num_warps=num_warps,
    )

    if batched:
        q_y = q_y.reshape(BS, -1, q_y.shape[-1])

    return q_y, s_y


################################################################################
# High-precision Input, High-precision Fake Quantized Output
################################################################################


@triton.jit
def _fp8_rope_hp2pg_fake_quant_forward_kernel(
    x_ptr,
    real_ptr,
    imag_ptr,
    y_ptr,
    SEQ_LEN: tl.constexpr,
    N: tl.constexpr,
    N2: tl.constexpr,
    SN: tl.constexpr,
    SN2: tl.constexpr,
    QB: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_HEADS2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_HALF: tl.constexpr,
    fp8_max,
    quant_type: tl.constexpr,
    triton_float8_dtype: tl.constexpr,
    x_stride_0,
    real_stride_0,
    real_stride_1,
    imag_stride_0,
    imag_stride_1,
    y_stride_0,
    SCALE_MIN_THRES: tl.constexpr,
):
    row_id = tl.program_id(0)
    batch_id = row_id // SEQ_LEN
    seq_id = row_id % SEQ_LEN

    cols = tl.arange(0, N2)
    mask = cols < N
    scale_cols = tl.arange(0, SN2)
    scale_mask = scale_cols < SN
    head_half_cols = tl.arange(0, HEAD_DIM_HALF)
    head_half_mask = head_half_cols < HEAD_DIM

    # Shift the pointer to the current row
    x_ptr += row_id * x_stride_0
    real_ptr += seq_id * real_stride_0
    imag_ptr += seq_id * imag_stride_0
    y_ptr += row_id * y_stride_0

    # Load input
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    real = tl.load(
        real_ptr + head_half_cols * real_stride_1, mask=head_half_mask, other=0.0
    ).to(tl.float32)
    imag = tl.load(
        imag_ptr + head_half_cols * imag_stride_1, mask=head_half_mask, other=0.0
    ).to(tl.float32)

    # Apply RoPE
    x = x.reshape(NUM_HEADS2, HEAD_DIM_HALF, 2)
    x_first, x_second = tl.split(x)

    # Forward: y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    y_first = x_first * real - x_second * imag
    y_second = x_second * real + x_first * imag

    y = tl.interleave(y_first, y_second)
    y = tl.reshape(y, (N2))

    # Quantize
    y = tl.reshape(y, (SN2, QB))
    q_y, s_y = _fp8_quant_pg_1d(y, fp8_max, quant_type, SCALE_MIN_THRES, N2, SN2, QB)

    # Cast to low-precision
    q_y = q_y.to(triton_float8_dtype)

    # Dequantize (Fake quant)
    y = _fp8_dequant_pg_1d(q_y, s_y, quant_type, SCALE_MIN_THRES, N2, SN2, QB)
    y = tl.reshape(y, (N2))

    # Write output
    tl.store(y_ptr + cols, y, mask=mask)


def fp8_rope_hp2pg_fake_quant_fwd(
    x, freqs_cis, num_heads, head_dim, rope_config, block_size=None
):
    """
    Forward pass of the RoPE.

    Args:
        x (torch.Tensor): Input tensor, high-precision. (B, N, H * D)
        freqs_cis (torch.Tensor): Frequency tensor. (N, D // 2)

    Returns:
        y (torch.Tensor): Output tensor. Per-group quantized but in high-precision. Fake-quantized. (B, N, H * D)
    """
    float8_dtype = rope_config.float8_dtype
    quant_type = rope_config.quant_type
    QB = block_size if block_size is not None else rope_config.block_size

    # Change batched 3D input to 2D
    [x], batched, BS = flatten_if_batched(x)

    freqs_real, freqs_imag = freqs_cis.real, freqs_cis.imag
    assert x.is_contiguous(), "Input tensors must be contiguous"

    M, N = x.shape
    SEQ_LEN = M // BS
    SN = N // QB

    N2 = triton.next_power_of_2(N)
    SN2 = N2 // QB

    NUM_HEADS = num_heads
    NUM_HEADS2 = triton.next_power_of_2(NUM_HEADS)

    HEAD_DIM = head_dim
    HEAD_DIM_HALF = HEAD_DIM // 2
    assert HEAD_DIM in [64, 128, 256], "Head dim should be 64, 128, or 256"

    # Allocate output
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # heuristics for number of warps
    num_warps = 8
    fp8MaxValue = torch.finfo(float8_dtype).max
    triton_float8_dtype = convert_fp8_dtype_torch_to_triton[float8_dtype]

    # Launch kernel. Here it must be a 1D grid, other wise will raise: triton error [cuda]: invalid argument
    _fp8_rope_hp2pg_fake_quant_forward_kernel[(BS * SEQ_LEN,)](
        x,
        freqs_real,
        freqs_imag,
        y,
        SEQ_LEN,
        N,
        N2,
        SN,
        SN2,
        QB,
        NUM_HEADS,
        NUM_HEADS2,
        HEAD_DIM,
        HEAD_DIM_HALF,
        fp8MaxValue,
        quant_type,
        triton_float8_dtype,
        x.stride(0),
        freqs_real.stride(0),
        freqs_real.stride(1),
        freqs_imag.stride(0),
        freqs_imag.stride(1),
        y.stride(0),
        SCALE_MIN_THRES,
        num_warps=num_warps,
    )

    if batched:
        y = y.reshape(BS, -1, y.shape[-1])

    return y
