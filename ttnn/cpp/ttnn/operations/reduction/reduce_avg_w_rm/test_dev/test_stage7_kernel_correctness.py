# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Stage 7: Kernel Correctness Tests
Owned by: ttnn-kernel-writer agent

These tests verify that the kernel implementation produces correct results,
matching PyTorch reference values.
"""

import pytest
import torch
import ttnn


@pytest.fixture
def device():
    """
    Device fixture with proper management.

    Note: Before running tests:
    1. Run 'tt-smi -ls' to verify device 0 is available
    2. Run 'tt-smi -r 0' to reset if needed (see CLAUDE.md)
    """
    with ttnn.manage_device(device_id=0) as dev:
        yield dev


def compute_reference(input_torch: torch.Tensor) -> torch.Tensor:
    """
    Compute the expected output using PyTorch.
    reduce_avg_w_rm computes the mean across the width dimension (dim=-1).

    Input: [N, C, H, W]
    Output: [N, C, H, 1] but with physical width=32 for tiling
    """
    # Mean along width dimension, keepdim for shape preservation
    return input_torch.mean(dim=-1, keepdim=True)


def test_basic_correctness(device):
    """Test basic functional correctness with a small tensor."""
    torch.manual_seed(42)

    # Simple case: 1x1x32x64 (single tile row, Wt=2)
    torch_input = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    expected = compute_reference(torch_input)

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output_tensor = ttnn.reduce_avg_w_rm(input_tensor)
    output_torch = ttnn.to_torch(output_tensor)

    # Output has shape [N, C, H, 32] due to tile constraints
    # We only compare the first element of width (index 0) since that's where the result is
    # Actually, the reduced value should be broadcasted across all 32 positions in pack_untilize
    # For width reduction, each row becomes a single value, and untilize writes it to first column

    # Note: The output shape is [1, 1, 32, 32] but the meaningful data is in the first column
    # after untilize, or it may be broadcasted. Let's check the first column.
    output_first_col = output_torch[:, :, :, 0:1]

    torch.testing.assert_close(
        output_first_col.float(), expected.float(), rtol=0.02, atol=0.01  # Relax tolerance for bfloat16
    )


def test_multi_batch_correctness(device):
    """Test correctness with multiple batches and channels."""
    torch.manual_seed(123)

    # Multiple batches and channels: 2x3x32x96 (Wt=3)
    torch_input = torch.randn(2, 3, 32, 96, dtype=torch.bfloat16)
    expected = compute_reference(torch_input)

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output_tensor = ttnn.reduce_avg_w_rm(input_tensor)
    output_torch = ttnn.to_torch(output_tensor)

    output_first_col = output_torch[:, :, :, 0:1]

    torch.testing.assert_close(output_first_col.float(), expected.float(), rtol=0.02, atol=0.01)


def test_multi_tile_row_correctness(device):
    """Test correctness with multiple tile rows (H > 32)."""
    torch.manual_seed(456)

    # Multiple tile rows: 1x1x64x128 (Ht=2, Wt=4)
    torch_input = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
    expected = compute_reference(torch_input)

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output_tensor = ttnn.reduce_avg_w_rm(input_tensor)
    output_torch = ttnn.to_torch(output_tensor)

    output_first_col = output_torch[:, :, :, 0:1]

    torch.testing.assert_close(output_first_col.float(), expected.float(), rtol=0.02, atol=0.01)


def test_large_width_correctness(device):
    """Test correctness with larger width (more tiles to reduce)."""
    torch.manual_seed(789)

    # Larger width: 1x1x32x256 (Wt=8)
    torch_input = torch.randn(1, 1, 32, 256, dtype=torch.bfloat16)
    expected = compute_reference(torch_input)

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output_tensor = ttnn.reduce_avg_w_rm(input_tensor)
    output_torch = ttnn.to_torch(output_tensor)

    output_first_col = output_torch[:, :, :, 0:1]

    torch.testing.assert_close(
        output_first_col.float(),
        expected.float(),
        rtol=0.03,  # Slightly more tolerance for larger reductions
        atol=0.02,
    )


def test_uniform_values(device):
    """Test with uniform input values - average should equal the input value."""
    # All 1.0 values: average should be 1.0
    torch_input = torch.ones(1, 1, 32, 64, dtype=torch.bfloat16)
    expected = compute_reference(torch_input)  # Should be all 1.0

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output_tensor = ttnn.reduce_avg_w_rm(input_tensor)
    output_torch = ttnn.to_torch(output_tensor)

    output_first_col = output_torch[:, :, :, 0:1]

    torch.testing.assert_close(output_first_col.float(), expected.float(), rtol=0.01, atol=0.001)


def test_zero_values(device):
    """Test with zero input values - average should be zero."""
    torch_input = torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16)
    expected = compute_reference(torch_input)  # Should be all 0.0

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output_tensor = ttnn.reduce_avg_w_rm(input_tensor)
    output_torch = ttnn.to_torch(output_tensor)

    output_first_col = output_torch[:, :, :, 0:1]

    torch.testing.assert_close(output_first_col.float(), expected.float(), rtol=0.01, atol=0.001)


def test_alternating_values(device):
    """Test with alternating positive and negative values."""
    torch.manual_seed(999)

    # Create alternating pattern
    torch_input = torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16)
    torch_input[:, :, :, 0::2] = 1.0  # Even columns = 1
    torch_input[:, :, :, 1::2] = -1.0  # Odd columns = -1
    # Average should be 0 for each row

    expected = compute_reference(torch_input)

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output_tensor = ttnn.reduce_avg_w_rm(input_tensor)
    output_torch = ttnn.to_torch(output_tensor)

    output_first_col = output_torch[:, :, :, 0:1]

    torch.testing.assert_close(output_first_col.float(), expected.float(), rtol=0.01, atol=0.001)


def test_comprehensive_correctness(device):
    """Comprehensive test with realistic dimensions and random values."""
    torch.manual_seed(42424)

    # Realistic shape: 4x8x128x256 (batch=4, channels=8, Ht=4, Wt=8)
    torch_input = torch.randn(4, 8, 128, 256, dtype=torch.bfloat16)
    expected = compute_reference(torch_input)

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output_tensor = ttnn.reduce_avg_w_rm(input_tensor)
    output_torch = ttnn.to_torch(output_tensor)

    output_first_col = output_torch[:, :, :, 0:1]

    torch.testing.assert_close(
        output_first_col.float(), expected.float(), rtol=0.05, atol=0.03  # More tolerance for larger reductions
    )
