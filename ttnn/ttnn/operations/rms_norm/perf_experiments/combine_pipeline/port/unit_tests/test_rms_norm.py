# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for ttnn.operations.rms_norm (immutable spec — do not modify).

RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma

Covers: Layout::TILE and Layout::ROW_MAJOR inputs, bfloat16 and float32, ranks 2-4,
tile-aligned and non-tile-aligned H / W, gamma absent / ROW_MAJOR / TILE / mixed dtype,
custom epsilon, the default (None) compute config, input validation, and one shape pinned
to each blocking regime of op_design.md (row-parallel R1, W-split R2 by occupancy, R2 by
L1 fit, R2 with >= 2 blocks per row-group).
"""

import pytest
import ttnn

from ttnn.operations.rms_norm import rms_norm

PCC_THRESHOLD = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


def torch_rms_norm(x, gamma=None, epsilon=1e-6):
    import torch  # function-local: the ttnn tree forbids global torch imports

    xf = x.to(torch.float32)
    y = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + epsilon)
    if gamma is not None:
        y = y * gamma.to(torch.float32).reshape(-1)
    return y


def compute_pcc(actual, expected):
    import torch  # function-local: the ttnn tree forbids global torch imports

    a = actual.flatten().to(torch.float64)
    e = expected.flatten().to(torch.float64)
    if torch.equal(a, e):
        return 1.0
    return torch.corrcoef(torch.stack([a, e]))[0, 1].item()


def max_precision_config():
    config = ttnn.ComputeConfigDescriptor()
    config.math_fidelity = ttnn.MathFidelity.HiFi4
    config.fp32_dest_acc_en = True
    config.math_approx_mode = False
    return config


def run_and_check(
    device,
    shape,
    dtype,
    layout,
    *,
    gamma_mode=None,  # None | "row_major" | "tile"
    gamma_dtype=None,
    epsilon=1e-6,
    input_scale=1.0,
    pass_config=True,
):
    import torch  # function-local: the ttnn tree forbids global torch imports

    torch.manual_seed(42)
    torch_input = (torch.randn(shape, dtype=torch.float32) * input_scale).to(TORCH_DTYPE[dtype])

    W = shape[-1]
    torch_gamma = None
    tt_gamma = None
    if gamma_mode is not None:
        g_dtype = gamma_dtype if gamma_dtype is not None else dtype
        g_layout = ttnn.TILE_LAYOUT if gamma_mode == "tile" else ttnn.ROW_MAJOR_LAYOUT
        torch_gamma = torch.randn(W, dtype=torch.float32).to(TORCH_DTYPE[g_dtype])
        tt_gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W),
            dtype=g_dtype,
            layout=g_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    kwargs = {"gamma": tt_gamma, "epsilon": epsilon}
    if pass_config:
        kwargs["compute_kernel_config"] = max_precision_config()
    tt_output = rms_norm(tt_input, **kwargs)

    assert tt_output.layout == layout, f"output layout {tt_output.layout} != input layout {layout}"
    assert tt_output.dtype == dtype, f"output dtype {tt_output.dtype} != input dtype {dtype}"
    assert list(tt_output.shape) == list(shape), f"output shape {list(tt_output.shape)} != {list(shape)}"

    actual = ttnn.to_torch(tt_output).to(torch.float32)
    expected = torch_rms_norm(torch_input, torch_gamma, epsilon)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all(), "output contains NaN/Inf"

    pcc = compute_pcc(actual, expected)
    assert pcc >= PCC_THRESHOLD[dtype], f"PCC {pcc:.6f} < {PCC_THRESHOLD[dtype]} for shape={shape} dtype={dtype}"


# Single-tile, multi-tile, non-square, multi-batch, ranks 2/3/4, aligned and non-aligned.
SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile
    (2, 3, 64, 96),  # multi-batch, non-square
    (1, 1, 17, 50),  # H and W non-aligned
    (2, 1, 100, 47),  # multi-batch, both non-aligned
    (3, 2, 45, 96),  # H non-aligned per image (tile-row padding inside each image)
    (32, 64),  # rank 2
    (17, 64),  # rank 2, H non-aligned
    (4, 128, 47),  # rank 3, W non-aligned
    (1, 32, 4096),  # rank 3, wide
]

LAYOUTS = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]
DTYPES = [ttnn.bfloat16, ttnn.float32]


@pytest.mark.parametrize("gamma_mode", [None, "row_major"], ids=["no_gamma", "gamma_rm"])
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "row_major"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_rms_norm(device, shape, layout, dtype, gamma_mode):
    run_and_check(device, shape, dtype, layout, gamma_mode=gamma_mode)


@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "row_major"])
@pytest.mark.parametrize("shape", [(1, 1, 64, 128), (1, 1, 17, 50)], ids=["aligned", "non_aligned"])
def test_rms_norm_tile_gamma(device, shape, layout):
    run_and_check(device, shape, ttnn.bfloat16, layout, gamma_mode="tile")


@pytest.mark.parametrize(
    "dtype, gamma_dtype",
    [(ttnn.bfloat16, ttnn.float32), (ttnn.float32, ttnn.bfloat16)],
    ids=["bf16_in_fp32_gamma", "fp32_in_bf16_gamma"],
)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "row_major"])
def test_rms_norm_mixed_gamma_dtype(device, layout, dtype, gamma_dtype):
    run_and_check(device, (2, 1, 64, 160), dtype, layout, gamma_mode="row_major", gamma_dtype=gamma_dtype)


@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "row_major"])
@pytest.mark.parametrize("epsilon", [1e-5, 1e-2])
def test_rms_norm_epsilon(device, layout, epsilon):
    # Small-magnitude input so mean(x^2) ~ 2.5e-3 and epsilon materially changes the result.
    run_and_check(
        device, (1, 1, 64, 96), ttnn.bfloat16, layout, gamma_mode="row_major", epsilon=epsilon, input_scale=0.05
    )


@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "row_major"])
def test_rms_norm_default_config(device, layout):
    # compute_kernel_config=None resolves through default_compute_kernel_config().
    run_and_check(device, (1, 1, 64, 128), ttnn.bfloat16, layout, gamma_mode="row_major", pass_config=False)


# Regime-pinned shapes (op_design.md -> Regime selection). The predicates depend only on
# grid_x >= 2, so each shape lands in its regime on Wormhole and Blackhole grids alike.
REGIME_SHAPES = [
    pytest.param((2, 1, 1024, 256), id="R1_row_parallel"),  # 64 tile-rows: k = 1
    pytest.param((1, 1, 32, 4096), id="R2_occupancy"),  # 1 tile-row, 128 W tiles: k > 1 by occupancy
    pytest.param((1, 1, 64, 1000), id="R2_occupancy_w_non_aligned"),  # W tail on the last member only
    pytest.param((1, 1, 32, 16384), id="R2_fit"),  # 512 W tiles: does not fit one Tensix core
    pytest.param(
        (1, 1, 6144, 12288), id="R2_fit_multi_block"
    ),  # k = 2 by fit, >= 3 tile-rows per group: depth 2 + lookahead
]


@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "row_major"])
@pytest.mark.parametrize("shape", REGIME_SHAPES)
def test_rms_norm_regimes(device, shape, layout):
    run_and_check(device, shape, ttnn.bfloat16, layout, gamma_mode="row_major")


@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "row_major"])
def test_rms_norm_regime_fit_fp32(device, layout):
    run_and_check(device, (1, 1, 32, 8192), ttnn.float32, layout, gamma_mode="row_major")


def test_rms_norm_rejects_rank_1(device, expect_error):
    import torch  # function-local: the ttnn tree forbids global torch imports

    tt_input = ttnn.from_torch(
        torch.randn(64, dtype=torch.float32).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    with expect_error((ValueError, RuntimeError), r".*"):
        rms_norm(tt_input)


def test_rms_norm_rejects_gamma_width_mismatch(device, expect_error):
    import torch  # function-local: the ttnn tree forbids global torch imports

    tt_input = ttnn.from_torch(
        torch.randn(1, 1, 32, 64, dtype=torch.float32).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_gamma = ttnn.from_torch(
        torch.randn(1, 1, 1, 32, dtype=torch.float32).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    with expect_error((ValueError, RuntimeError), r".*"):
        rms_norm(tt_input, gamma=tt_gamma)


def test_rms_norm_rejects_fp32_without_fp32_dest_acc(device, expect_error):
    import torch  # function-local: the ttnn tree forbids global torch imports

    tt_input = ttnn.from_torch(
        torch.randn(1, 1, 32, 64, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    config = ttnn.ComputeConfigDescriptor()
    config.math_fidelity = ttnn.MathFidelity.HiFi4
    config.fp32_dest_acc_en = False
    with expect_error((NotImplementedError, ValueError, RuntimeError), r".*"):
        rms_norm(tt_input, compute_kernel_config=config)
