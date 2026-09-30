# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for mhc_post (immutable spec — do not modify).

X'_j = post_j * F + sum_i comb[i][j] * X_i   (comb applied TRANSPOSED)

F (..., T, C), X (..., T, n*C) with stream i in columns [i*C, (i+1)*C),
post (..., T, n) float32, comb (..., T, n*n) float32 with comb[..., i*n + j] = comb[i][j].
Output: X's shape and dtype, TILE, DRAM interleaved.
"""

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import UnsupportedAxisValue
from ttnn.bringup.mhc_post import mhc_post, default_compute_kernel_config

N = 4

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}


def reference_mhc_post(f, x, post, comb):
    n = post.shape[-1]
    lead, C = tuple(f.shape[:-1]), f.shape[-1]
    ff = f.to(torch.float32).reshape(-1, 1, C)
    xx = x.to(torch.float32).reshape(-1, n, C)
    pp = post.to(torch.float32).reshape(-1, n, 1)
    mm = comb.to(torch.float32).reshape(-1, n, n)
    out = pp * ff + torch.einsum("tij,tic->tjc", mm, xx)
    return out.reshape(*lead, n * C)


def _sinkhorn(logits, iters=20, eps=1e-6):
    m = torch.softmax(logits, dim=-1) + eps
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


def make_inputs(x_shape, *, dtype, sublayer_dtype, comb_mode="sinkhorn"):
    torch.manual_seed(42)
    lead, nc = tuple(x_shape[:-1]), x_shape[-1]
    C = nc // N
    f = torch.randn(lead + (C,)).to(TORCH_DTYPE[sublayer_dtype])
    x = torch.randn(x_shape).to(TORCH_DTYPE[dtype])
    post = (2.0 * torch.sigmoid(torch.randn(lead + (N,)))).float()
    if comb_mode == "sinkhorn":
        comb = _sinkhorn(torch.randn(*lead, N, N))
    elif comb_mode == "cyclic":
        # comb[i][(i+1) % n] = 1  ->  X'_j carries X_{j-1}: pins the transposed orientation.
        comb = torch.roll(torch.eye(N), shifts=1, dims=1).expand(*lead, N, N)
    else:
        raise ValueError(comb_mode)
    comb = comb.reshape(lead + (N * N,)).float().contiguous()
    return f, x, post, comb


def to_device(t, device, dtype):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def run(device, x_shape, *, dtype, sublayer_dtype, comb_mode="sinkhorn", compute_kernel_config="default"):
    f, x, post, comb = make_inputs(x_shape, dtype=dtype, sublayer_dtype=sublayer_dtype, comb_mode=comb_mode)
    args = (
        to_device(f, device, sublayer_dtype),
        to_device(x, device, dtype),
        to_device(post, device, ttnn.float32),
        to_device(comb, device, ttnn.float32),
    )
    if compute_kernel_config == "default":
        out = mhc_post(*args)
    else:
        out = mhc_post(*args, compute_kernel_config=compute_kernel_config)
    return out, reference_mhc_post(f, x, post, comb), args


def pcc(a, b):
    a = a.to(torch.float64).flatten()
    b = b.to(torch.float64).flatten()
    if torch.equal(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def check(out, expected, x_shape, dtype):
    assert list(out.shape) == list(x_shape)
    assert out.dtype == dtype
    assert out.layout == ttnn.TILE_LAYOUT
    got = ttnn.to_torch(out).to(torch.float32)
    assert got.shape == expected.shape
    assert torch.isfinite(got).all()
    p = pcc(got, expected)
    assert p >= PCC[dtype], f"PCC {p} < {PCC[dtype]}"
    return got


# X shapes (last dim n*C). F/post/comb shapes are implied.
SHAPES = [
    pytest.param((32, 4 * 32), id="single_tile_C32_rank2"),
    pytest.param((64, 4 * 256), id="multi_tile_C256_rank2"),
    pytest.param((1, 128, 4 * 1024), id="non_square_C1024_rank3"),
    pytest.param((1, 2, 256, 4 * 2048), id="multi_batch_C2048_rank4"),
    pytest.param((1, 1, 640, 4 * 1792), id="dsv4_tp4_T640_C1792"),
    pytest.param((17, 4 * 128), id="T17_non_aligned"),
    pytest.param((1, 100, 4 * 1024), id="T100_non_aligned_rank3"),
    pytest.param((2, 33, 4 * 64), id="T33_non_aligned_batch2"),
    pytest.param((1, 4 * 7168), id="T1_decode_C7168"),
]

# (X/X' dtype, F dtype): Phase 0 fp32/fp32 + Refinement 1 bf16 combos.
DTYPES = [
    pytest.param((ttnn.float32, ttnn.float32), id="X_fp32-F_fp32"),
    pytest.param((ttnn.bfloat16, ttnn.bfloat16), id="X_bf16-F_bf16"),
    pytest.param((ttnn.float32, ttnn.bfloat16), id="X_fp32-F_bf16"),
    pytest.param((ttnn.bfloat16, ttnn.float32), id="X_bf16-F_fp32"),
]


@pytest.mark.parametrize("dtypes", DTYPES)
@pytest.mark.parametrize("x_shape", SHAPES)
def test_mhc_post(device, x_shape, dtypes):
    dtype, sublayer_dtype = dtypes
    out, expected, _ = run(device, x_shape, dtype=dtype, sublayer_dtype=sublayer_dtype)
    check(out, expected, x_shape, dtype)


@pytest.mark.parametrize(
    "x_shape", [pytest.param((64, 4 * 256), id="C256"), pytest.param((1, 40, 4 * 512), id="T40_non_aligned")]
)
def test_mhc_post_comb_orientation(device, x_shape):
    """cyclic comb: X'_j = post_j * F + X_{j-1} — a non-symmetric comb pins comb^T."""
    out, expected, _ = run(device, x_shape, dtype=ttnn.float32, sublayer_dtype=ttnn.float32, comb_mode="cyclic")
    check(out, expected, x_shape, ttnn.float32)


def test_mhc_post_explicit_config(device):
    x_shape = (1, 64, 4 * 512)
    cfg = default_compute_kernel_config()
    assert cfg.fp32_dest_acc_en
    out, expected, _ = run(device, x_shape, dtype=ttnn.float32, sublayer_dtype=ttnn.float32, compute_kernel_config=cfg)
    check(out, expected, x_shape, ttnn.float32)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
def test_mhc_post_deterministic(device, dtype):
    x_shape = (1, 1, 96, 4 * 768)
    out1, expected, args = run(device, x_shape, dtype=dtype, sublayer_dtype=dtype)
    out2 = mhc_post(*args)
    a = ttnn.to_torch(out1)
    b = ttnn.to_torch(out2)
    assert torch.equal(a, b), "two calls on the same inputs must be bitwise identical"
    check(out1, expected, x_shape, dtype)


def test_mhc_post_refuses_fp16_dest(device, expect_error):
    """fp32_dest_acc_en=False is outside TARGET: validate() refuses it (message names the axis)."""
    x_shape = (32, 4 * 32)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False
    )
    with expect_error(UnsupportedAxisValue, "fp32_dest_acc_en"):
        run(device, x_shape, dtype=ttnn.float32, sublayer_dtype=ttnn.float32, compute_kernel_config=cfg)
