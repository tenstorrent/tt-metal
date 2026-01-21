# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Stage 7: Kernel Correctness Tests
Owned by: ttnn-kernel-writer agent

These tests verify that the kernels produce numerically correct results
compared to PyTorch reference implementations.
"""

import pytest
import torch
import ttnn


def compute_reference(input_torch: torch.Tensor) -> torch.Tensor:
    """
    Compute reference output using PyTorch.
    reduce_mean_w_rm: mean reduction along width dimension (last dimension).
    Input: [N, C, H, W] row-major
    Output: [N, C, H, 1] row-major (padded to width 32)
    """
    return input_torch.mean(dim=-1, keepdim=True)


# NOTE: Use the built-in `device` fixture from conftest.py - do NOT define your own.
# Before running: 'tt-smi -ls' to verify device, 'tt-smi -r 0' to reset (see CLAUDE.md)


class TestReduceMeanWRmCorrectness:
    """Test correctness of reduce_mean_w_rm operation."""

    def test_basic_correctness_32x64(self, device):
        """Basic test: 32x64 input (1 tile height, 2 tile widths)"""
        torch.manual_seed(42)
        input_torch = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
        expected = compute_reference(input_torch)

        input_tensor = ttnn.from_torch(
            input_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        output_tensor = ttnn.reduce_mean_w_rm(input_tensor)
        output_torch = ttnn.to_torch(output_tensor)

        # Output shape should be [1, 1, 32, 1]
        assert output_torch.shape == expected.shape, f"Shape mismatch: {output_torch.shape} vs {expected.shape}"

        # Numerical comparison with tolerance for bfloat16
        torch.testing.assert_close(output_torch, expected, rtol=1e-2, atol=1e-2)

    def test_multi_tile_height_64x64(self, device):
        """Multi-tile height: 64x64 (2 tile heights, 2 tile widths)"""
        torch.manual_seed(42)
        input_torch = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
        expected = compute_reference(input_torch)

        input_tensor = ttnn.from_torch(
            input_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        output_tensor = ttnn.reduce_mean_w_rm(input_tensor)
        output_torch = ttnn.to_torch(output_tensor)

        assert output_torch.shape == expected.shape, f"Shape mismatch: {output_torch.shape} vs {expected.shape}"
        torch.testing.assert_close(output_torch, expected, rtol=1e-2, atol=1e-2)

    def test_larger_width_32x128(self, device):
        """Larger width: 32x128 (1 tile height, 4 tile widths)"""
        torch.manual_seed(42)
        input_torch = torch.randn(1, 1, 32, 128, dtype=torch.bfloat16)
        expected = compute_reference(input_torch)

        input_tensor = ttnn.from_torch(
            input_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        output_tensor = ttnn.reduce_mean_w_rm(input_tensor)
        output_torch = ttnn.to_torch(output_tensor)

        assert output_torch.shape == expected.shape, f"Shape mismatch: {output_torch.shape} vs {expected.shape}"
        torch.testing.assert_close(output_torch, expected, rtol=1e-2, atol=1e-2)

    def test_square_64x64(self, device):
        """Square tensor: 64x64"""
        torch.manual_seed(42)
        input_torch = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
        expected = compute_reference(input_torch)

        input_tensor = ttnn.from_torch(
            input_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        output_tensor = ttnn.reduce_mean_w_rm(input_tensor)
        output_torch = ttnn.to_torch(output_tensor)

        assert output_torch.shape == expected.shape, f"Shape mismatch: {output_torch.shape} vs {expected.shape}"
        torch.testing.assert_close(output_torch, expected, rtol=1e-2, atol=1e-2)

    def test_uniform_values(self, device):
        """Uniform values: all elements same, mean should equal that value"""
        # Use a value that's representable in bfloat16
        value = 0.5
        input_torch = torch.full((1, 1, 32, 64), value, dtype=torch.bfloat16)
        expected = compute_reference(input_torch)  # Should be 0.5

        input_tensor = ttnn.from_torch(
            input_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        output_tensor = ttnn.reduce_mean_w_rm(input_tensor)
        output_torch = ttnn.to_torch(output_tensor)

        assert output_torch.shape == expected.shape, f"Shape mismatch: {output_torch.shape} vs {expected.shape}"
        torch.testing.assert_close(output_torch, expected, rtol=1e-2, atol=1e-2)

    def test_zeros(self, device):
        """All zeros: mean should be zero"""
        input_torch = torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16)
        expected = compute_reference(input_torch)

        input_tensor = ttnn.from_torch(
            input_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        output_tensor = ttnn.reduce_mean_w_rm(input_tensor)
        output_torch = ttnn.to_torch(output_tensor)

        assert output_torch.shape == expected.shape, f"Shape mismatch: {output_torch.shape} vs {expected.shape}"
        torch.testing.assert_close(output_torch, expected, rtol=1e-2, atol=1e-2)
