import torch
import triton
import triton.language as tl
from einops import rearrange


# Only for debug
def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def get_configs():
    configs = []
    for BLOCK_M in [64, 128, 256, 512]:
        for num_warps in [4, 8]:
            for num_stages in [2, 3, 4, 5]:
                configs.append(
                    triton.Config(
                        {"BLOCK_M": BLOCK_M}, num_stages=num_stages, num_warps=num_warps
                    )
                )
    return configs


@triton.autotune(
    configs=get_configs(),
    key=["HEAD_DIM", "HEAD_DIM_HALF"],
)
@triton.jit
def _triton_rope(
    q_ptr,
    q_stride_0,
    q_stride_1,
    k_ptr,
    k_stride_0,
    k_stride_1,
    q_out_ptr,
    q_out_stride_0,
    q_out_stride_1,
    k_out_ptr,
    k_out_stride_0,
    k_out_stride_1,
    cos_ptr,
    cos_stride_0,
    sin_ptr,
    sin_stride_0,
    BLOCK_M: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_HALF: tl.constexpr,
):
    row_id = tl.program_id(0)
    seq_id = tl.program_id(1)

    # Shift to current head
    q_ptr += row_id * q_stride_0
    k_ptr += row_id * k_stride_0
    q_out_ptr += row_id * q_out_stride_0
    k_out_ptr += row_id * k_out_stride_0

    # Calculate offsets
    seq_offset = tl.arange(0, BLOCK_M)
    head_offset = tl.arange(0, HEAD_DIM)

    # Calculate offsets for q, k, cos, sin along sequence dimension
    offset_qm = seq_id * BLOCK_M + seq_offset
    offset_km = seq_id * BLOCK_M + seq_offset
    offset_cos_half_m = seq_id * BLOCK_M + seq_offset
    offset_sin_half_m = seq_id * BLOCK_M + seq_offset

    q_ptr = q_ptr + offset_qm[:, None] * q_stride_1 + head_offset[None, :]
    k_ptr = k_ptr + offset_km[:, None] * k_stride_1 + head_offset[None, :]
    cos_ptr = cos_ptr + offset_cos_half_m[:, None] * cos_stride_0 + head_offset[None, :]
    sin_ptr = sin_ptr + offset_sin_half_m[:, None] * sin_stride_0 + head_offset[None, :]
    out_q_ptr = q_out_ptr + offset_qm[:, None] * q_out_stride_1 + head_offset[None, :]
    out_k_ptr = k_out_ptr + offset_km[:, None] * k_out_stride_1 + head_offset[None, :]

    # Load q, k, cos, sin
    q = tl.load(q_ptr, mask=offset_qm[:, None] < SEQ_LEN, other=0.0).to(tl.float32)
    k = tl.load(k_ptr, mask=offset_km[:, None] < SEQ_LEN, other=0.0).to(tl.float32)
    cos = tl.load(cos_ptr, mask=offset_cos_half_m[:, None] < SEQ_LEN, other=0.0).to(
        tl.float32
    )
    sin = tl.load(sin_ptr, mask=offset_sin_half_m[:, None] < SEQ_LEN, other=0.0).to(
        tl.float32
    )

    # Get half of cos and sin
    cos = cos.reshape(BLOCK_M, HEAD_DIM_HALF, 2)
    sin = sin.reshape(BLOCK_M, HEAD_DIM_HALF, 2)
    cos_half, _ = tl.split(cos)
    sin_half, _ = tl.split(sin)

    # Apply RoPE
    q = q.reshape(BLOCK_M, HEAD_DIM_HALF, 2)
    k = k.reshape(BLOCK_M, HEAD_DIM_HALF, 2)
    q_first, q_second = tl.split(q)
    k_first, k_second = tl.split(k)

    # Forward: y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    out_q_first = q_first * cos_half - q_second * sin_half
    out_q_second = q_first * sin_half + q_second * cos_half
    out_k_first = k_first * cos_half - k_second * sin_half
    out_k_second = k_first * sin_half + k_second * cos_half

    # Interleave
    out_q = tl.interleave(out_q_first, out_q_second)
    out_k = tl.interleave(out_k_first, out_k_second)

    # Reshape
    out_q = tl.reshape(out_q, (BLOCK_M, HEAD_DIM))
    out_k = tl.reshape(out_k, (BLOCK_M, HEAD_DIM))

    out_q = out_q.to(q.type.element_ty)
    out_k = out_k.to(k.type.element_ty)

    # Store
    tl.store(out_q_ptr, out_q, mask=offset_qm[:, None] < SEQ_LEN)
    tl.store(out_k_ptr, out_k, mask=offset_km[:, None] < SEQ_LEN)


def rope_forward(q, k, cos, sin):
    """
    Args:
        q (torch.Tensor): (B, H, S, D)
        k (torch.Tensor): (B, H, S, D)
        cos (torch.Tensor): (1, 1, S, D) or (S, D)
        sin (torch.Tensor): (1, 1, S, D) or (S, D)

    Returns:
        q_out (torch.Tensor): (B, H, S, D)
        k_out (torch.Tensor): (B, H, S, D)
    """

    batch_size, num_heads, seq_len, head_dim = q.shape

    q = q.reshape(batch_size * num_heads, seq_len, head_dim).contiguous()
    k = k.reshape(batch_size * num_heads, seq_len, head_dim).contiguous()
    cos = cos.reshape(seq_len, head_dim).contiguous()
    sin = sin.reshape(seq_len, head_dim).contiguous()

    q_out, k_out = torch.empty_like(q), torch.empty_like(k)

    grid = lambda META: (
        batch_size * num_heads,
        triton.cdiv(seq_len, META["BLOCK_M"]),
    )
    _triton_rope[grid](
        q,
        q.stride(0),
        q.stride(1),
        k,
        k.stride(0),
        k.stride(1),
        q_out,
        q_out.stride(0),
        q_out.stride(1),
        k_out,
        k_out.stride(0),
        k_out.stride(1),
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        HEAD_DIM_HALF=head_dim // 2,
    )

    q_out = q_out.reshape(batch_size, num_heads, seq_len, head_dim)
    k_out = k_out.reshape(batch_size, num_heads, seq_len, head_dim)

    return q_out, k_out


def rope_forward_seq_first(q, k, freqs_cis):
    """
    Preprocess and apply RoPE to the input tensors.
    
    Args:
        q (torch.Tensor): (B, S, H, D)
        k (torch.Tensor): (B, S, H, D)
        freqs_cis (tuple of torch.Tensor): (S, D)

    Returns:
        q_out (torch.Tensor): (B, S, H, D)
        k_out (torch.Tensor): (B, S, H, D)
    """

    B, S, H, D = q.shape
    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()
    cos, sin = freqs_cis
    q_out, k_out = rope_forward(q, k, cos, sin)
    q_out = q_out.permute(0, 2, 1, 3)
    k_out = k_out.permute(0, 2, 1, 3)
    return q_out, k_out

################################################################################
# Unit Test
################################################################################


def test_rope_forward():
    """
    Unit test for rope_forward using saved debug data.

    The debug.pt file contains: [q, k, cos, sin, q_out, k_out]
    where q_out and k_out are the expected outputs from reference implementation.
    """
    import os

    debug_path = "/lustre/fsw/portfolios/nvr/users/hxi/workspace/VideoGeneration/LongCat-Video/debug.pt"

    # Load saved test data
    q, k, cos, sin, q_out_expected, k_out_expected = torch.load(debug_path)

    # # Get only the first several seq len and head dim
    # seq_keep, hid_keep = 2, 16
    # q = q[:, :1, :seq_keep, :hid_keep]
    # k = k[:, :1, :seq_keep, :hid_keep]
    # cos = cos[:, :, :seq_keep, :hid_keep]
    # sin = sin[:, :, :seq_keep, :hid_keep]
    # q_out_expected = q_out_expected[:, :1, :seq_keep, :hid_keep]
    # k_out_expected = k_out_expected[:, :1, :seq_keep, :hid_keep]

    # Print Shapes
    print(f"q shape: {q.shape}")  # [B, H, S, D]
    print(f"k shape: {k.shape}")  # [B, H, S, D]
    print(f"cos shape: {cos.shape}")  # [1, 1, S, D]
    print(f"sin shape: {sin.shape}")  # [1, 1, S, D]
    print(f"q_out_expected shape: {q_out_expected.shape}")  # [B, H, S, D]
    print(f"k_out_expected shape: {k_out_expected.shape}")  # [B, H, S, D]

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = q.to(device).contiguous()
    k = k.to(device).contiguous()
    cos = cos.to(device).contiguous()
    sin = sin.to(device).contiguous()
    q_out_expected = q_out_expected.to(device)
    k_out_expected = k_out_expected.to(device)

    # q, k shape: (B, H, S, D) - already correct for rope_forward
    # cos, sin shape: (1, 1, S, D) - need to convert to (1, S, D//2)
    B, H, S, D = q.shape

    print(f"Test configuration: num_heads={H}, head_dim={D}")
    print(f"q shape: {q.shape}, k shape: {k.shape}")
    print(f"cos shape: {cos.shape}, sin shape: {sin.shape}")

    # Run triton kernel
    q_out, k_out = rope_forward(q, k, cos, sin)

    # Compare results
    q_diff = (q_out - q_out_expected).abs().max()
    k_diff = (k_out - k_out_expected).abs().max()
    print(f"Max q difference: {q_diff}")
    print(f"Max k difference: {k_diff}")

    # print(f"q_out_expected: {q_out_expected}")
    # print(f"q_out: {q_out}")


if __name__ == "__main__":
    test_rope_forward()
