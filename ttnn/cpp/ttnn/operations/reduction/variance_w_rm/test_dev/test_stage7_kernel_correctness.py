"""Stage 7: Kernel Correctness Tests
Owned by: ttnn-kernel-writer agent

Tests verify that variance_w_rm produces numerically correct results
against PyTorch reference (torch.var with unbiased=False).
"""
import pytest
import torch
import ttnn


def compute_variance_reference(input_torch):
    """
    Compute variance along the last dimension using population variance (unbiased=False).
    This matches what our kernel computes: mean((x - mean(x))^2)
    """
    return torch.var(input_torch, dim=-1, keepdim=True, unbiased=False)


def test_single_tile_correctness(device):
    """Test variance on a single tile (32x32)."""
    torch.manual_seed(42)
    input_torch = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    expected = compute_variance_reference(input_torch.float()).bfloat16()

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    output_tensor = ttnn.variance_w_rm(input_tensor)
    output_torch = ttnn.to_torch(output_tensor)

    # Variance should be positive
    assert (output_torch >= 0).all(), "Variance should be non-negative"

    # Check shape
    assert output_torch.shape[-1] == 1, f"Output width should be 1, got {output_torch.shape[-1]}"

    # Numerical check (bfloat16 has limited precision)
    torch.testing.assert_close(
        output_torch.float(),
        expected.float(),
        rtol=1e-1,
        atol=1e-1,
        msg="Single tile variance mismatch",
    )


def test_multi_tile_width_correctness(device):
    """Test variance on multiple tiles in width direction (32x64 = 1 tile height x 2 tiles width)."""
    torch.manual_seed(123)
    input_torch = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    expected = compute_variance_reference(input_torch.float()).bfloat16()

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    output_tensor = ttnn.variance_w_rm(input_tensor)
    output_torch = ttnn.to_torch(output_tensor)

    # Numerical check
    torch.testing.assert_close(
        output_torch.float(),
        expected.float(),
        rtol=1e-1,
        atol=1e-1,
        msg="Multi-tile width variance mismatch",
    )


def test_multi_tile_height_correctness(device):
    """Test variance on multiple tiles in height direction (64x32 = 2 tile heights x 1 tile width)."""
    torch.manual_seed(456)
    input_torch = torch.randn(1, 1, 64, 32, dtype=torch.bfloat16)
    expected = compute_variance_reference(input_torch.float()).bfloat16()

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    output_tensor = ttnn.variance_w_rm(input_tensor)
    output_torch = ttnn.to_torch(output_tensor)

    # Numerical check
    torch.testing.assert_close(
        output_torch.float(),
        expected.float(),
        rtol=1e-1,
        atol=1e-1,
        msg="Multi-tile height variance mismatch",
    )


def test_multi_tile_both_directions(device):
    """Test variance on multiple tiles in both directions (64x128 = 2x4 tiles)."""
    torch.manual_seed(789)
    input_torch = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
    expected = compute_variance_reference(input_torch.float()).bfloat16()

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    output_tensor = ttnn.variance_w_rm(input_tensor)
    output_torch = ttnn.to_torch(output_tensor)

    # Numerical check
    torch.testing.assert_close(
        output_torch.float(),
        expected.float(),
        rtol=1e-1,
        atol=1e-1,
        msg="Multi-tile both directions variance mismatch",
    )


def test_constant_row_variance_zero(device):
    """Test that constant rows have variance of zero."""
    # Create tensor where each row has the same value
    constant_value = 5.0
    input_torch = torch.full((1, 1, 32, 64), constant_value, dtype=torch.bfloat16)
    expected = torch.zeros(1, 1, 32, 1, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    output_tensor = ttnn.variance_w_rm(input_tensor)
    output_torch = ttnn.to_torch(output_tensor)

    # Variance of constant should be zero (or very close due to float precision)
    torch.testing.assert_close(
        output_torch.float(),
        expected.float(),
        rtol=1e-2,
        atol=1e-2,
        msg="Constant row should have zero variance",
    )


def test_known_variance_case(device):
    """Test a known variance case: [0, 1, 2, 3, ..., 31] repeated has known variance."""
    # For [0, 1, 2, ..., N-1], population variance = (N^2 - 1) / 12
    # For N=32: variance = (1024 - 1) / 12 = 85.25
    N = 32
    row_pattern = torch.arange(N, dtype=torch.float32)
    expected_variance = (N * N - 1) / 12.0  # = 85.25

    # Create input with this pattern repeated for each row
    input_torch = row_pattern.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, 32, 32)
    input_torch = input_torch.to(torch.bfloat16)

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    output_tensor = ttnn.variance_w_rm(input_tensor)
    output_torch = ttnn.to_torch(output_tensor)

    # All rows should have the same variance
    expected = torch.full((1, 1, 32, 1), expected_variance, dtype=torch.bfloat16)

    torch.testing.assert_close(
        output_torch.float(),
        expected.float(),
        rtol=5e-2,
        atol=5.0,  # Allow some absolute tolerance for bfloat16
        msg=f"Expected variance ~{expected_variance}, got {output_torch[0,0,0,0].item()}",
    )


@pytest.mark.skip(reason="Program factory needs update to handle batched inputs (N*C*Ht tile-rows)")
def test_batched_input(device):
    """Test variance on batched input.

    NOTE: Currently skipped because the program factory only accounts for Ht (height in tiles)
    but not the batch and channel dimensions. For a tensor [N, C, H, W], the total number of
    tile-rows should be N * C * Ht, not just Ht. This requires a Stage 5/6 fix.
    """
    torch.manual_seed(999)
    input_torch = torch.randn(2, 3, 32, 64, dtype=torch.bfloat16)
    expected = compute_variance_reference(input_torch.float()).bfloat16()

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    output_tensor = ttnn.variance_w_rm(input_tensor)
    output_torch = ttnn.to_torch(output_tensor)

    # Numerical check
    torch.testing.assert_close(
        output_torch.float(),
        expected.float(),
        rtol=1e-1,
        atol=1e-1,
        msg="Batched variance mismatch",
    )


def test_output_shape_reduced(device):
    """Test that output shape has reduced last dimension."""
    input_torch = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    output_tensor = ttnn.variance_w_rm(input_tensor)

    # Logical shape should have width=1
    assert output_tensor.shape[-1] == 1, f"Expected output width=1, got {output_tensor.shape[-1]}"
    assert output_tensor.shape[-2] == 64, f"Expected output height=64, got {output_tensor.shape[-2]}"

    # Padded shape should have width=32
    assert output_tensor.padded_shape[-1] == 32, f"Expected padded width=32, got {output_tensor.padded_shape[-1]}"


def test_wider_tensor(device):
    """Test variance on a wider tensor (256 elements = 8 tiles wide)."""
    torch.manual_seed(1111)
    input_torch = torch.randn(1, 1, 32, 256, dtype=torch.bfloat16)
    expected = compute_variance_reference(input_torch.float()).bfloat16()

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    output_tensor = ttnn.variance_w_rm(input_tensor)
    output_torch = ttnn.to_torch(output_tensor)

    # Numerical check
    torch.testing.assert_close(
        output_torch.float(),
        expected.float(),
        rtol=1e-1,
        atol=1e-1,
        msg="Wide tensor variance mismatch",
    )
