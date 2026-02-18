# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Row Centralize - Tests

Run from repo root:
    pytest ttnn/ttnn/operations/row_centralize/test_row_centralize.py -v

Tests are colocated with operation code for experimental convenience.
"""

import pytest
import ttnn

from .row_centralize import row_centralize


def pytorch_reference(x, epsilon: float = 1e-5):
    """
    PyTorch reference for row_centralize.

    Implements: y = (x - mean(x, -1)) / sqrt(var(x, -1, correction=0) + epsilon)

    Uses correction=0 (population variance) because the kernel divides by W, not W-1.
    """
    import torch

    mu = x.mean(dim=-1, keepdim=True)
    c = x - mu
    var = c.pow(2).mean(dim=-1, keepdim=True)
    s = torch.rsqrt(var + epsilon)
    return c * s


# ============================================================
# Infrastructure Tests (stub kernels)
# These verify the Python-side setup without checking numerics.
# ============================================================


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 32), id="single_tile_32x32"),
        pytest.param((64, 64), id="two_tile_rows_64x64"),
        pytest.param((32, 128), id="wide_32x128"),
        pytest.param((1, 32, 64), id="batch1_32x64"),
        pytest.param((2, 64, 64), id="batch2_64x64"),
    ],
)
def test_row_centralize_shape_preserved(device, shape):
    """Verify that the operation runs and output has the correct shape and dtype/layout."""
    import torch

    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = row_centralize(ttnn_input, epsilon=1e-5)

    # Shape must be preserved
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {list(ttnn_output.shape)} vs {list(shape)}"

    # Output must be bfloat16
    assert ttnn_output.dtype == ttnn.bfloat16, f"Expected bfloat16 output, got {ttnn_output.dtype}"

    # Output must be ROW_MAJOR
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT, f"Expected ROW_MAJOR_LAYOUT output, got {ttnn_output.layout}"

    # Output memory layout must be interleaved
    assert (
        ttnn_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    ), f"Expected INTERLEAVED output, got {ttnn_output.memory_config().memory_layout}"


# ============================================================
# Validation Tests
# ============================================================


@pytest.mark.parametrize(
    "bad_rank",
    [
        pytest.param((32,), id="rank1"),
    ],
)
def test_row_centralize_rejects_low_rank(device, bad_rank):
    """Input with rank < 2 should raise ValueError."""
    import torch

    torch_input = torch.randn(bad_rank, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(ValueError, match="rank >= 2"):
        row_centralize(ttnn_input)


@pytest.mark.parametrize(
    "bad_layout_shape",
    [
        pytest.param((32, 32), id="tile_layout_input"),
    ],
)
def test_row_centralize_rejects_tile_layout(device, bad_layout_shape):
    """Input in TILE_LAYOUT should raise ValueError mentioning ROW_MAJOR."""
    import torch

    torch_input = torch.randn(bad_layout_shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,  # wrong layout
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(ValueError, match="ROW_MAJOR"):
        row_centralize(ttnn_input)


@pytest.mark.parametrize(
    "bad_last_dim",
    [
        pytest.param((32, 48), id="last_dim_48_not_div32"),
    ],
)
def test_row_centralize_rejects_non_divisible_last_dim(device, bad_last_dim):
    """Input whose last dimension is not divisible by 32 should raise ValueError."""
    import torch

    torch_input = torch.randn(bad_last_dim, dtype=torch.bfloat16)
    # For ROW_MAJOR the dims don't need to be tile-aligned for ttnn.from_torch.
    # The validation happens inside row_centralize, not in from_torch.
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(ValueError, match="divisible by 32"):
        row_centralize(ttnn_input)


# ============================================================
# Functional Tests (will be meaningful after kernel implementation)
# ============================================================


@pytest.mark.parametrize(
    "shape,epsilon",
    [
        pytest.param((32, 32), 1e-5, id="single_tile"),
        pytest.param((32, 128), 1e-5, id="wide_single_row"),
        pytest.param((64, 64), 1e-5, id="multi_row_multi_col"),
        pytest.param((2, 32, 64), 1e-5, id="batch_3d"),
    ],
)
def test_row_centralize_accuracy(device, shape, epsilon):
    """
    Compare row_centralize output against PyTorch reference.

    NOTE: With stub kernels, this test WILL FAIL numerically.
    After kernel implementation, uncomment the allclose assertion and verify
    rtol=0.02, atol=0.02 (bf16 precision).
    """
    import torch

    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = row_centralize(ttnn_input, epsilon=epsilon)

    # TODO: After kernel implementation, assert this passes:
    # torch_output = ttnn.to_torch(ttnn_output)
    # torch_expected = pytorch_reference(torch_input, epsilon=epsilon)
    # assert torch.allclose(
    #     torch_output.float(),
    #     torch_expected.float(),
    #     rtol=0.02,
    #     atol=0.02,
    # ), f"Max diff: {(torch_output.float() - torch_expected.float()).abs().max()}"

    # For now, only verify shape (stub kernels produce garbage output)
    assert list(ttnn_output.shape) == list(shape)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 32), id="single_tile"),
    ],
)
def test_row_centralize_constant_row_output_near_zero(device, shape):
    """
    A row where all values are constant should produce output near zero
    (after kernel implementation).
    """
    import torch

    torch_input = torch.ones(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = row_centralize(ttnn_input, epsilon=1e-5)

    # After kernel implementation, verify output is near zero:
    # torch_output = ttnn.to_torch(ttnn_output)
    # assert torch.allclose(torch_output, torch.zeros_like(torch_output), atol=0.01)

    # For stub kernels, just check shape
    assert list(ttnn_output.shape) == list(shape)
