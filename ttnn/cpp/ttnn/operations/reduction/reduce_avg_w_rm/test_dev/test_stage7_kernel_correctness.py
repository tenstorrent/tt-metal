# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Stage 7: Kernel Correctness Tests
Owned by: ttnn-kernel-writer agent

These tests verify that the reduce_avg_w_rm operation produces numerically correct
results compared to PyTorch reference implementation.
"""

import pytest
import torch
import ttnn


@pytest.fixture
def device():
    # Note: Run 'tt-smi -ls' first to verify device 0 is available
    # and 'tt-smi -r 0' to reset if needed (see CLAUDE.md)
    with ttnn.manage_device(device_id=0) as dev:
        yield dev


def compute_reference(input_torch):
    """
    Compute the reference output using PyTorch.

    reduce_avg_w_rm computes: output[n, c, h, 0] = mean(input[n, c, h, :])

    The output has width=32 for tile alignment, but only the first column
    contains valid data.
    """
    # Compute mean along width dimension (dim=-1)
    mean_values = torch.mean(input_torch.float(), dim=-1, keepdim=True)  # [N, C, H, 1]

    # Pad to width 32 for tile alignment (only first column has valid data)
    N, C, H, _ = input_torch.shape
    output = torch.zeros(N, C, H, 32, dtype=torch.bfloat16)
    output[:, :, :, 0:1] = mean_values.to(torch.bfloat16)

    return output


def test_single_tile_correctness(device):
    """Verify single tile (32x32) produces correct average."""
    torch.manual_seed(42)

    # Single tile input
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)

    # Convert to TTNN tensor
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run operation
    output_tensor = ttnn.reduce_avg_w_rm(input_tensor)

    # Convert back to torch
    output_torch = ttnn.to_torch(output_tensor)

    # Compute reference
    expected = compute_reference(torch_input)

    # Verify first column (the valid data)
    # bfloat16 has limited precision, so use relaxed tolerance
    torch.testing.assert_close(
        output_torch[:, :, :, 0:1], expected[:, :, :, 0:1], rtol=1e-2, atol=1e-2, msg="Single tile average mismatch"
    )


def test_multi_tile_width_correctness(device):
    """Verify multi-tile width produces correct average."""
    torch.manual_seed(123)

    # Multiple tiles in width (64 = 2 tiles)
    torch_input = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)

    # Convert to TTNN tensor
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run operation
    output_tensor = ttnn.reduce_avg_w_rm(input_tensor)

    # Convert back to torch
    output_torch = ttnn.to_torch(output_tensor)

    # Compute reference
    expected = compute_reference(torch_input)

    # Verify first column
    torch.testing.assert_close(
        output_torch[:, :, :, 0:1],
        expected[:, :, :, 0:1],
        rtol=1e-2,
        atol=1e-2,
        msg="Multi-tile width average mismatch",
    )


def test_multi_tile_height_correctness(device):
    """Verify multi-tile height produces correct average per row."""
    torch.manual_seed(456)

    # Multiple tiles in height (64 = 2 tiles)
    torch_input = torch.randn(1, 1, 64, 32, dtype=torch.bfloat16)

    # Convert to TTNN tensor
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run operation
    output_tensor = ttnn.reduce_avg_w_rm(input_tensor)

    # Convert back to torch
    output_torch = ttnn.to_torch(output_tensor)

    # Compute reference
    expected = compute_reference(torch_input)

    # Verify first column
    torch.testing.assert_close(
        output_torch[:, :, :, 0:1],
        expected[:, :, :, 0:1],
        rtol=1e-2,
        atol=1e-2,
        msg="Multi-tile height average mismatch",
    )


def test_batch_correctness(device):
    """Verify batched inputs produce correct averages."""
    torch.manual_seed(789)

    # Multiple batches
    torch_input = torch.randn(2, 2, 32, 32, dtype=torch.bfloat16)

    # Convert to TTNN tensor
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run operation
    output_tensor = ttnn.reduce_avg_w_rm(input_tensor)

    # Convert back to torch
    output_torch = ttnn.to_torch(output_tensor)

    # Compute reference
    expected = compute_reference(torch_input)

    # Verify first column for all batches
    torch.testing.assert_close(
        output_torch[:, :, :, 0:1], expected[:, :, :, 0:1], rtol=1e-2, atol=1e-2, msg="Batch average mismatch"
    )


def test_large_tensor_correctness(device):
    """Verify larger tensor produces correct averages."""
    torch.manual_seed(1024)

    # Larger tensor with multiple tiles in all dimensions
    torch_input = torch.randn(1, 4, 64, 128, dtype=torch.bfloat16)

    # Convert to TTNN tensor
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run operation
    output_tensor = ttnn.reduce_avg_w_rm(input_tensor)

    # Convert back to torch
    output_torch = ttnn.to_torch(output_tensor)

    # Compute reference
    expected = compute_reference(torch_input)

    # Verify first column
    torch.testing.assert_close(
        output_torch[:, :, :, 0:1], expected[:, :, :, 0:1], rtol=1e-2, atol=1e-2, msg="Large tensor average mismatch"
    )


def test_uniform_input_correctness(device):
    """Verify uniform input produces correct (trivial) average."""
    # All ones - average should be 1.0
    torch_input = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)

    # Convert to TTNN tensor
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run operation
    output_tensor = ttnn.reduce_avg_w_rm(input_tensor)

    # Convert back to torch
    output_torch = ttnn.to_torch(output_tensor)

    # Expected: all 1.0 in first column
    expected_value = 1.0
    actual_values = output_torch[0, 0, :, 0]

    # All values should be close to 1.0
    torch.testing.assert_close(
        actual_values,
        torch.full_like(actual_values, expected_value),
        rtol=1e-2,
        atol=1e-2,
        msg="Uniform input average should be 1.0",
    )


def test_zero_input_correctness(device):
    """Verify zero input produces zero average."""
    # All zeros - average should be 0.0
    torch_input = torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16)

    # Convert to TTNN tensor
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run operation
    output_tensor = ttnn.reduce_avg_w_rm(input_tensor)

    # Convert back to torch
    output_torch = ttnn.to_torch(output_tensor)

    # Expected: all 0.0 in first column
    actual_values = output_torch[0, 0, :, 0]

    # All values should be close to 0.0
    torch.testing.assert_close(
        actual_values, torch.zeros_like(actual_values), rtol=1e-2, atol=1e-2, msg="Zero input average should be 0.0"
    )


def test_known_value_correctness(device):
    """Verify known value pattern produces expected average."""
    # Create input where each row has values 0, 1, 2, ..., 31
    # Average should be (0+1+2+...+31)/32 = 15.5
    torch_input = torch.arange(32, dtype=torch.bfloat16).unsqueeze(0).repeat(32, 1)
    torch_input = torch_input.unsqueeze(0).unsqueeze(0)  # [1, 1, 32, 32]

    # Convert to TTNN tensor
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run operation
    output_tensor = ttnn.reduce_avg_w_rm(input_tensor)

    # Convert back to torch
    output_torch = ttnn.to_torch(output_tensor)

    # Expected: 15.5 in first column for all rows
    expected_value = 15.5
    actual_values = output_torch[0, 0, :, 0]

    # All values should be close to 15.5
    torch.testing.assert_close(
        actual_values,
        torch.full_like(actual_values, expected_value),
        rtol=1e-2,
        atol=0.5,  # Allow 0.5 tolerance for integer average
        msg=f"Expected average of 15.5, got {actual_values[0].item()}",
    )
