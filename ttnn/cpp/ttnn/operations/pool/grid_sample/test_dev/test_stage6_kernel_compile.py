import ttnn
import torch
import pytest


def test_nearest_kernels_compile():
    """Test that kernels compile and execute without errors"""
    device = ttnn.open_device(device_id=0)

    # Minimal test - just needs to compile and run
    input_t = ttnn.ones((1, 32, 32, 32), device=device, dtype=ttnn.bfloat16)
    grid_t = ttnn.zeros((1, 16, 16, 2), device=device, dtype=ttnn.bfloat16)

    # Should now complete successfully
    result = ttnn.grid_sample(input_t, grid_t, mode="nearest")

    # Basic sanity checks
    assert result is not None
    # TTNN uses [N, H, W, C] format, output grid is 16x16
    assert result.shape == (1, 16, 16, 32), f"Expected shape (1, 16, 16, 32), got {result.shape}"

    # Convert to torch to check values
    result_torch = ttnn.to_torch(result)
    assert not torch.isnan(result_torch).any(), "Output contains NaN values"
    assert not torch.isinf(result_torch).any(), "Output contains Inf values"

    print(f"✓ Kernels compiled and executed successfully!")
    print(f"  Output shape: {result.shape}")
    print(f"  Output dtype: {result.dtype}")
    print(f"  Sample output values: {result_torch[0, 0, 0, :5]}")  # First 5 channels at (0,0)

    ttnn.close_device(device)
