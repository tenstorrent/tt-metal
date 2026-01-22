# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Stage 7: Kernel Correctness Tests
Owned by: ttnn-kernel-writer agent

Tests verify that the layer_norm_w_rm operation produces correct results
compared to PyTorch reference (F.layer_norm).
"""

import pytest
import torch
import ttnn


def compute_reference_layer_norm(input_torch, gamma_torch, beta_torch, epsilon=1e-5):
    """Compute layer normalization reference using explicit formula.

    Formula: output = ((input - mean) / sqrt(variance + epsilon)) * gamma + beta

    Where:
    - mean is computed across the last dimension (W)
    - variance is computed across the last dimension (W)
    - gamma and beta are applied element-wise (broadcast across height)
    """
    # Ensure input is float32 for precision
    input_f32 = input_torch.float()
    gamma_f32 = gamma_torch.float()
    beta_f32 = beta_torch.float()

    # Compute mean across last dimension
    mean = input_f32.mean(dim=-1, keepdim=True)

    # Compute variance across last dimension (using biased variance like layer_norm)
    variance = ((input_f32 - mean) ** 2).mean(dim=-1, keepdim=True)

    # Standardize
    standardized = (input_f32 - mean) / torch.sqrt(variance + epsilon)

    # Apply affine transform
    # gamma and beta need to be broadcast to match input shape
    output = standardized * gamma_f32 + beta_f32

    return output.bfloat16()


# NOTE: Use the built-in `device` fixture from conftest.py - do NOT define your own.


class TestLayerNormWRmCorrectness:
    """Test functional correctness of layer_norm_w_rm operation."""

    @pytest.mark.parametrize(
        "H,W",
        [
            (32, 32),  # Single tile
            (32, 64),  # 1 tile height, 2 tiles width
            (64, 32),  # 2 tiles height, 1 tile width
            (64, 64),  # 2x2 tiles
            (32, 128),  # 1 tile height, 4 tiles width
            (128, 64),  # 4 tiles height, 2 tiles width
        ],
    )
    def test_correctness_various_sizes(self, device, H, W):
        """Test layer norm correctness for various tensor sizes."""
        torch.manual_seed(42)

        # Create input tensor
        input_torch = torch.randn(H, W, dtype=torch.bfloat16)

        # Create gamma and beta with shape [W] (will be broadcast across H)
        gamma_torch = torch.randn(W, dtype=torch.bfloat16) * 0.5 + 1.0  # Around 1.0
        beta_torch = torch.randn(W, dtype=torch.bfloat16) * 0.1  # Small values

        # Compute reference
        expected = compute_reference_layer_norm(input_torch, gamma_torch, beta_torch, epsilon=1e-5)

        # Convert to TTNN tensors
        input_tensor = ttnn.from_torch(
            input_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        gamma_tensor = ttnn.from_torch(
            gamma_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        beta_tensor = ttnn.from_torch(
            beta_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        # Run layer_norm_w_rm
        output_tensor = ttnn.layer_norm_w_rm(input_tensor, gamma_tensor, beta_tensor, epsilon=1e-5)
        output_torch = ttnn.to_torch(output_tensor)

        # Compare results
        torch.testing.assert_close(
            output_torch.float(),
            expected.float(),
            rtol=5e-2,
            atol=5e-2,
            msg=f"Layer norm mismatch for shape [{H}, {W}]",
        )

    def test_correctness_gamma_ones_beta_zeros(self, device):
        """Test that gamma=1, beta=0 produces standardized output (same as standardize_w_rm)."""
        torch.manual_seed(42)

        H, W = 64, 64
        input_torch = torch.randn(H, W, dtype=torch.bfloat16)

        # With gamma=1 and beta=0, layer_norm should produce standardized values
        gamma_torch = torch.ones(W, dtype=torch.bfloat16)
        beta_torch = torch.zeros(W, dtype=torch.bfloat16)

        # Reference: just standardization
        input_f32 = input_torch.float()
        mean = input_f32.mean(dim=-1, keepdim=True)
        variance = ((input_f32 - mean) ** 2).mean(dim=-1, keepdim=True)
        expected = ((input_f32 - mean) / torch.sqrt(variance + 1e-5)).bfloat16()

        # Convert to TTNN
        input_tensor = ttnn.from_torch(
            input_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        gamma_tensor = ttnn.from_torch(
            gamma_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        beta_tensor = ttnn.from_torch(
            beta_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        output_tensor = ttnn.layer_norm_w_rm(input_tensor, gamma_tensor, beta_tensor, epsilon=1e-5)
        output_torch = ttnn.to_torch(output_tensor)

        torch.testing.assert_close(
            output_torch.float(),
            expected.float(),
            rtol=5e-2,
            atol=5e-2,
            msg="gamma=1, beta=0 should produce standardized output",
        )

    def test_correctness_constant_gamma_beta(self, device):
        """Test with constant gamma and beta values."""
        torch.manual_seed(42)

        H, W = 64, 64
        input_torch = torch.randn(H, W, dtype=torch.bfloat16)

        # Constant gamma and beta
        gamma_torch = torch.full((W,), 2.0, dtype=torch.bfloat16)
        beta_torch = torch.full((W,), 0.5, dtype=torch.bfloat16)

        expected = compute_reference_layer_norm(input_torch, gamma_torch, beta_torch, epsilon=1e-5)

        input_tensor = ttnn.from_torch(
            input_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        gamma_tensor = ttnn.from_torch(
            gamma_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        beta_tensor = ttnn.from_torch(
            beta_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        output_tensor = ttnn.layer_norm_w_rm(input_tensor, gamma_tensor, beta_tensor, epsilon=1e-5)
        output_torch = ttnn.to_torch(output_tensor)

        torch.testing.assert_close(
            output_torch.float(),
            expected.float(),
            rtol=5e-2,
            atol=5e-2,
            msg="Constant gamma/beta test failed",
        )

    def test_correctness_all_zeros_input(self, device):
        """Test with all-zeros input (edge case for zero variance)."""
        H, W = 32, 32

        # All zeros input
        input_torch = torch.zeros(H, W, dtype=torch.bfloat16)
        gamma_torch = torch.ones(W, dtype=torch.bfloat16)
        beta_torch = torch.full((W,), 0.5, dtype=torch.bfloat16)

        # With all zeros, mean=0, variance=0
        # standardized = 0 / sqrt(0 + epsilon) = 0
        # output = 0 * gamma + beta = beta
        expected = beta_torch.expand(H, W).clone()

        input_tensor = ttnn.from_torch(
            input_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        gamma_tensor = ttnn.from_torch(
            gamma_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        beta_tensor = ttnn.from_torch(
            beta_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        output_tensor = ttnn.layer_norm_w_rm(input_tensor, gamma_tensor, beta_tensor, epsilon=1e-5)
        output_torch = ttnn.to_torch(output_tensor)

        torch.testing.assert_close(
            output_torch.float(),
            expected.float(),
            rtol=5e-2,
            atol=5e-2,
            msg="All-zeros input should produce output equal to beta",
        )

    def test_correctness_different_epsilon(self, device):
        """Test with different epsilon values."""
        torch.manual_seed(42)

        H, W = 32, 64
        input_torch = torch.randn(H, W, dtype=torch.bfloat16)
        gamma_torch = torch.randn(W, dtype=torch.bfloat16) * 0.5 + 1.0
        beta_torch = torch.randn(W, dtype=torch.bfloat16) * 0.1

        for epsilon in [1e-5, 1e-3, 1e-6]:
            expected = compute_reference_layer_norm(input_torch, gamma_torch, beta_torch, epsilon=epsilon)

            input_tensor = ttnn.from_torch(
                input_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            gamma_tensor = ttnn.from_torch(
                gamma_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            beta_tensor = ttnn.from_torch(
                beta_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )

            output_tensor = ttnn.layer_norm_w_rm(input_tensor, gamma_tensor, beta_tensor, epsilon=epsilon)
            output_torch = ttnn.to_torch(output_tensor)

            torch.testing.assert_close(
                output_torch.float(),
                expected.float(),
                rtol=5e-2,
                atol=5e-2,
                msg=f"Epsilon={epsilon} test failed",
            )

    def test_output_shape_matches_input(self, device):
        """Test that output shape always matches input shape."""
        torch.manual_seed(42)

        for H, W in [(32, 32), (64, 128), (128, 64)]:
            input_torch = torch.randn(H, W, dtype=torch.bfloat16)
            gamma_torch = torch.ones(W, dtype=torch.bfloat16)
            beta_torch = torch.zeros(W, dtype=torch.bfloat16)

            input_tensor = ttnn.from_torch(
                input_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            gamma_tensor = ttnn.from_torch(
                gamma_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            beta_tensor = ttnn.from_torch(
                beta_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )

            output_tensor = ttnn.layer_norm_w_rm(input_tensor, gamma_tensor, beta_tensor, epsilon=1e-5)

            assert output_tensor.shape == input_tensor.shape, f"Output shape mismatch for input [{H}, {W}]"
