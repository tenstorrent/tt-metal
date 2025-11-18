import ttnn
import torch
import torch.nn.functional as F
import pytest


def test_nearest_correctness_zeros_grid():
    """Test nearest neighbor with grid=zeros (should sample center of input)"""
    device = ttnn.open_device(device_id=0)

    # Create input with known pattern (32 channels for TILE_WIDTH alignment)
    input_torch = torch.ones((1, 32, 8, 8), dtype=torch.float32)
    # Set center pixel to 2.0 for identification
    input_torch[0, :, 4, 4] = 2.0

    # Grid of zeros maps to center of input (0, 0) -> (H/2, W/2) in pixel space
    grid_torch = torch.zeros((1, 4, 4, 2), dtype=torch.float32)

    # PyTorch reference (expects NCHW input, HWC grid with values in [-1, 1])
    expected_torch = F.grid_sample(input_torch, grid_torch, mode="nearest", padding_mode="zeros", align_corners=False)

    # TTNN (expects NHWC format)
    input_nhwc = input_torch.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC
    input_ttnn = ttnn.from_torch(input_nhwc, device=device, dtype=ttnn.bfloat16)
    grid_ttnn = ttnn.from_torch(grid_torch, device=device, dtype=ttnn.bfloat16)

    result_ttnn = ttnn.grid_sample(input_ttnn, grid_ttnn, mode="nearest")
    result_torch = ttnn.to_torch(result_ttnn)

    # Convert back to NCHW for comparison
    result_nchw = result_torch.permute(0, 3, 1, 2).contiguous()
    expected_float = expected_torch.float()
    result_float = result_nchw.float()

    # Check shapes match
    assert result_nchw.shape == expected_torch.shape, f"Shape mismatch: {result_nchw.shape} vs {expected_torch.shape}"

    # Check values are close (allowing for bfloat16 precision)
    max_diff = torch.abs(result_float - expected_float).max()
    print(f"  Max difference: {max_diff}")
    print(f"  Expected center value: {expected_float[0, 0, 0, 0]}")
    print(f"  Got center value: {result_float[0, 0, 0, 0]}")

    assert max_diff < 0.1, f"Values differ too much: max_diff={max_diff}"
    assert torch.allclose(result_float, expected_float, atol=0.05), "Output doesn't match PyTorch reference"

    print("✓ Correctness test with zeros grid passed!")

    ttnn.close_device(device)


def test_nearest_correctness_varied_grid():
    """Test nearest neighbor with varied grid coordinates"""
    device = ttnn.open_device(device_id=0)

    # Create input with spatial pattern (each position has unique value) - 32 channels for alignment
    input_torch = torch.zeros((1, 32, 8, 8), dtype=torch.float32)
    for h in range(8):
        for w in range(8):
            input_torch[0, :, h, w] = h * 10 + w  # Unique value per position

    # Grid with varied coordinates
    grid_torch = torch.tensor(
        [
            [
                [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]],  # Top row
                [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],  # Middle row
                [[-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
            ]  # Bottom row
        ],
        dtype=torch.float32,
    )

    # PyTorch reference
    expected_torch = F.grid_sample(input_torch, grid_torch, mode="nearest", padding_mode="zeros", align_corners=False)

    # TTNN
    input_nhwc = input_torch.permute(0, 2, 3, 1).contiguous()
    input_ttnn = ttnn.from_torch(input_nhwc, device=device, dtype=ttnn.bfloat16)
    grid_ttnn = ttnn.from_torch(grid_torch, device=device, dtype=ttnn.bfloat16)

    result_ttnn = ttnn.grid_sample(input_ttnn, grid_ttnn, mode="nearest")
    result_torch = ttnn.to_torch(result_ttnn)

    # Convert back to NCHW
    result_nchw = result_torch.permute(0, 3, 1, 2).contiguous()
    expected_float = expected_torch.float()
    result_float = result_nchw.float()

    # Check shapes
    assert result_nchw.shape == expected_torch.shape

    # Check values
    max_diff = torch.abs(result_float - expected_float).max()
    print(f"  Max difference: {max_diff}")
    print(f"  Expected sample at (1,1): {expected_float[0, 0, 1, 1]}")
    print(f"  Got sample at (1,1): {result_float[0, 0, 1, 1]}")

    assert max_diff < 0.1, f"Values differ too much: max_diff={max_diff}"
    assert torch.allclose(result_float, expected_float, atol=0.05), "Output doesn't match PyTorch reference"

    print("✓ Correctness test with varied grid passed!")

    ttnn.close_device(device)


def test_nearest_out_of_bounds():
    """Test nearest neighbor with out-of-bounds grid coordinates (should return zeros)"""
    device = ttnn.open_device(device_id=0)

    # Create input with all ones - 32 channels for alignment
    input_torch = torch.ones((1, 32, 8, 8), dtype=torch.float32)

    # Grid with out-of-bounds coordinates (outside [-1, 1] range)
    grid_torch = torch.tensor(
        [[[[-2.0, -2.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 0.0]]]],  # Out of bounds  # In bounds (center)
        dtype=torch.float32,
    )

    # PyTorch reference
    expected_torch = F.grid_sample(input_torch, grid_torch, mode="nearest", padding_mode="zeros", align_corners=False)

    # TTNN
    input_nhwc = input_torch.permute(0, 2, 3, 1).contiguous()
    input_ttnn = ttnn.from_torch(input_nhwc, device=device, dtype=ttnn.bfloat16)
    grid_ttnn = ttnn.from_torch(grid_torch, device=device, dtype=ttnn.bfloat16)

    result_ttnn = ttnn.grid_sample(input_ttnn, grid_ttnn, mode="nearest")
    result_torch = ttnn.to_torch(result_ttnn)

    # Convert back to NCHW
    result_nchw = result_torch.permute(0, 3, 1, 2).contiguous()
    expected_float = expected_torch.float()
    result_float = result_nchw.float()

    # Check out-of-bounds samples are zero
    assert result_float[0, 0, 0, 0] < 0.1, "Out-of-bounds sample should be zero"
    assert result_float[0, 0, 0, 1] < 0.1, "Out-of-bounds sample should be zero"

    # Check in-bounds samples are one
    assert abs(result_float[0, 0, 1, 0] - 1.0) < 0.1, "In-bounds sample should be one"
    assert abs(result_float[0, 0, 1, 1] - 1.0) < 0.1, "In-bounds sample should be one"

    max_diff = torch.abs(result_float - expected_float).max()
    print(f"  Max difference: {max_diff}")
    assert max_diff < 0.1

    print("✓ Out-of-bounds test passed!")

    ttnn.close_device(device)


if __name__ == "__main__":
    test_nearest_correctness_zeros_grid()
    test_nearest_correctness_varied_grid()
    test_nearest_out_of_bounds()
    print("\n✅ All correctness tests passed!")
