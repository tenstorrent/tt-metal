"""Stage 7: Kernel Correctness Tests for row_mean_sub_square_reduce
Owned by: ttnn-kernel-writer agent

These tests verify that the kernels produce numerically correct results.
Tests are run AFTER Stage 6 (kernel compilation) tests pass.

The operation computes population variance along the width (W) dimension:
  variance[n,c,h] = (1/W) * sum((input[n,c,h,w] - mean[n,c,h])^2 for w in 0..W-1)

PyTorch equivalent: torch.var(input, dim=-1, keepdim=True, unbiased=False)
"""
import pytest
import torch
import ttnn


def reference_variance_w(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the reference variance along width dimension.

    Args:
        input_tensor: [N, C, H, W] tensor

    Returns:
        [N, C, H, 32] tensor with variance in first position, padded to TILE_WIDTH=32
    """
    # Population variance (unbiased=False, divide by N not N-1)
    variance = torch.var(input_tensor.float(), dim=-1, keepdim=True, unbiased=False)
    # Pad to TILE_WIDTH=32
    output = torch.nn.functional.pad(variance, (0, 31))
    return output.to(input_tensor.dtype)


@pytest.fixture
def device():
    """Setup device for testing. Run 'tt-smi -r 0' before if test hangs."""
    with ttnn.manage_device(device_id=0) as dev:
        yield dev


# =============================================================================
# Basic Correctness Tests
# =============================================================================


def test_basic_correctness_single_tile(device):
    """
    Test correctness with single tile (32x32).
    This is the simplest case with Wt=1.
    """
    torch.manual_seed(42)
    input_torch = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    expected = reference_variance_w(input_torch)

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.row_mean_sub_square_reduce(input_tensor)
    result_torch = ttnn.to_torch(result)

    # Extract the first column (variance values)
    variance_result = result_torch[:, :, :, 0:1]
    variance_expected = expected[:, :, :, 0:1]

    # Use tolerances appropriate for bfloat16 computations
    # Variance computation involves multiple operations (mean, sub, square, sum)
    # which accumulate numerical errors in bfloat16
    torch.testing.assert_close(
        variance_result.float(),
        variance_expected.float(),
        rtol=0.1,  # 10% relative tolerance for bfloat16 accumulated errors
        atol=0.05,  # Absolute tolerance for small variances
        msg="Single tile variance mismatch",
    )


def test_basic_correctness_multi_tile_width(device):
    """
    Test correctness with multi-tile width (Wt=2).
    Tests reduction across multiple tiles in W dimension.
    """
    torch.manual_seed(43)
    input_torch = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)  # Wt=2
    expected = reference_variance_w(input_torch)

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.row_mean_sub_square_reduce(input_tensor)
    result_torch = ttnn.to_torch(result)

    variance_result = result_torch[:, :, :, 0:1]
    variance_expected = expected[:, :, :, 0:1]

    torch.testing.assert_close(
        variance_result.float(),
        variance_expected.float(),
        rtol=0.1,
        atol=0.05,
        msg="Multi-tile width variance mismatch",
    )


def test_basic_correctness_multi_tile_row(device):
    """
    Test correctness with multiple tile rows (Ht=4).
    Tests multi-core work distribution.
    """
    torch.manual_seed(44)
    input_torch = torch.randn(1, 1, 128, 64, dtype=torch.bfloat16)  # Ht=4, Wt=2
    expected = reference_variance_w(input_torch)

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.row_mean_sub_square_reduce(input_tensor)
    result_torch = ttnn.to_torch(result)

    variance_result = result_torch[:, :, :, 0:1]
    variance_expected = expected[:, :, :, 0:1]

    torch.testing.assert_close(
        variance_result.float(), variance_expected.float(), rtol=0.1, atol=0.05, msg="Multi-tile row variance mismatch"
    )


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_uniform_values_zero_variance(device):
    """
    Test with uniform values (variance should be 0).
    This is a critical edge case for numerical stability.
    """
    # All values are the same -> variance = 0
    input_torch = torch.ones(1, 1, 32, 64, dtype=torch.bfloat16) * 5.0
    expected = reference_variance_w(input_torch)

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.row_mean_sub_square_reduce(input_tensor)
    result_torch = ttnn.to_torch(result)

    variance_result = result_torch[:, :, :, 0:1]
    variance_expected = expected[:, :, :, 0:1]

    # Should be very close to 0
    torch.testing.assert_close(
        variance_result.float(),
        variance_expected.float(),
        rtol=0.1,
        atol=0.05,
        msg="Uniform values should produce zero variance",
    )


def test_known_variance(device):
    """
    Test with known input that produces a predictable variance.
    Using [0, 1] alternating pattern which has variance = 0.25
    """
    # Create pattern: 0, 1, 0, 1, ... -> mean=0.5, diff=[-0.5, 0.5, ...]
    # variance = mean((x - mean)^2) = mean([0.25, 0.25, ...]) = 0.25
    input_torch = torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16)
    input_torch[:, :, :, 1::2] = 1.0  # Alternating 0s and 1s
    expected = reference_variance_w(input_torch)

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.row_mean_sub_square_reduce(input_tensor)
    result_torch = ttnn.to_torch(result)

    variance_result = result_torch[:, :, :, 0:1]
    variance_expected = expected[:, :, :, 0:1]

    # Variance should be 0.25
    torch.testing.assert_close(
        variance_result.float(), variance_expected.float(), rtol=0.1, atol=0.05, msg="Known variance pattern mismatch"
    )

    # Additionally, verify it's close to 0.25
    assert torch.allclose(
        variance_result.float(), torch.full_like(variance_result.float(), 0.25), rtol=1e-2, atol=1e-3
    ), "Expected variance of 0.25 for alternating 0/1 pattern"


# =============================================================================
# Batch and Channel Tests
# =============================================================================


def test_batched_input(device):
    """
    Test with batched input (N > 1).
    """
    torch.manual_seed(45)
    input_torch = torch.randn(2, 1, 32, 64, dtype=torch.bfloat16)
    expected = reference_variance_w(input_torch)

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.row_mean_sub_square_reduce(input_tensor)
    result_torch = ttnn.to_torch(result)

    variance_result = result_torch[:, :, :, 0:1]
    variance_expected = expected[:, :, :, 0:1]

    torch.testing.assert_close(
        variance_result.float(), variance_expected.float(), rtol=0.1, atol=0.05, msg="Batched input variance mismatch"
    )


def test_multi_channel_input(device):
    """
    Test with multiple channels (C > 1).
    """
    torch.manual_seed(46)
    input_torch = torch.randn(1, 2, 32, 64, dtype=torch.bfloat16)
    expected = reference_variance_w(input_torch)

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.row_mean_sub_square_reduce(input_tensor)
    result_torch = ttnn.to_torch(result)

    variance_result = result_torch[:, :, :, 0:1]
    variance_expected = expected[:, :, :, 0:1]

    torch.testing.assert_close(
        variance_result.float(),
        variance_expected.float(),
        rtol=0.1,
        atol=0.05,
        msg="Multi-channel input variance mismatch",
    )


def test_full_batch_channel(device):
    """
    Test with full batch and channel dimensions.
    """
    torch.manual_seed(47)
    input_torch = torch.randn(2, 3, 64, 128, dtype=torch.bfloat16)
    expected = reference_variance_w(input_torch)

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.row_mean_sub_square_reduce(input_tensor)
    result_torch = ttnn.to_torch(result)

    variance_result = result_torch[:, :, :, 0:1]
    variance_expected = expected[:, :, :, 0:1]

    torch.testing.assert_close(
        variance_result.float(),
        variance_expected.float(),
        rtol=0.1,
        atol=0.05,
        msg="Full batch/channel variance mismatch",
    )


# =============================================================================
# Large Tensor Tests
# =============================================================================


def test_large_width(device):
    """
    Test with large width (W=256, Wt=8).
    """
    torch.manual_seed(48)
    input_torch = torch.randn(1, 1, 32, 256, dtype=torch.bfloat16)
    expected = reference_variance_w(input_torch)

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.row_mean_sub_square_reduce(input_tensor)
    result_torch = ttnn.to_torch(result)

    variance_result = result_torch[:, :, :, 0:1]
    variance_expected = expected[:, :, :, 0:1]

    torch.testing.assert_close(
        variance_result.float(), variance_expected.float(), rtol=0.1, atol=0.05, msg="Large width variance mismatch"
    )


def test_large_height(device):
    """
    Test with large height (H=256, Ht=8).
    Tests multi-core distribution with many tile rows.
    """
    torch.manual_seed(49)
    input_torch = torch.randn(1, 1, 256, 64, dtype=torch.bfloat16)
    expected = reference_variance_w(input_torch)

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.row_mean_sub_square_reduce(input_tensor)
    result_torch = ttnn.to_torch(result)

    variance_result = result_torch[:, :, :, 0:1]
    variance_expected = expected[:, :, :, 0:1]

    torch.testing.assert_close(
        variance_result.float(), variance_expected.float(), rtol=0.1, atol=0.05, msg="Large height variance mismatch"
    )


# =============================================================================
# Numerical Precision Tests
# =============================================================================


def test_numerical_stability_large_values(device):
    """
    Test numerical stability with large values.
    """
    torch.manual_seed(50)
    # Use values in range [-100, 100]
    input_torch = (torch.randn(1, 1, 32, 64, dtype=torch.bfloat16) * 100.0).clamp(-100, 100)
    expected = reference_variance_w(input_torch)

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.row_mean_sub_square_reduce(input_tensor)
    result_torch = ttnn.to_torch(result)

    variance_result = result_torch[:, :, :, 0:1]
    variance_expected = expected[:, :, :, 0:1]

    # Slightly relaxed tolerance for large values
    torch.testing.assert_close(
        variance_result.float(),
        variance_expected.float(),
        rtol=0.15,
        atol=0.1,
        msg="Large values numerical stability test failed",
    )


def test_numerical_stability_small_values(device):
    """
    Test numerical stability with small values.
    """
    torch.manual_seed(51)
    # Use small values in range [-0.01, 0.01]
    input_torch = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16) * 0.01
    expected = reference_variance_w(input_torch)

    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.row_mean_sub_square_reduce(input_tensor)
    result_torch = ttnn.to_torch(result)

    variance_result = result_torch[:, :, :, 0:1]
    variance_expected = expected[:, :, :, 0:1]

    torch.testing.assert_close(
        variance_result.float(),
        variance_expected.float(),
        rtol=0.15,
        atol=1e-4,
        msg="Small values numerical stability test failed",
    )
