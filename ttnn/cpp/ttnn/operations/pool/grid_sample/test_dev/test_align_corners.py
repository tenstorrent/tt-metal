# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for align_corners parameter in grid_sample operation (nearest mode).
Tests both align_corners=False and align_corners=True modes against PyTorch reference.
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn


def create_identity_grid(N, H_out, W_out):
    """Create an identity sampling grid (samples at uniform grid points)."""
    # Create normalized coordinates in [-1, 1] range
    y = torch.linspace(-1, 1, H_out)
    x = torch.linspace(-1, 1, W_out)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H_out, W_out, 2)
    grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)  # (N, H_out, W_out, 2)
    return grid


def create_random_grid(N, H_out, W_out, scale=0.8):
    """Create random sampling grid within [-scale, scale] range."""
    grid = torch.randn(N, H_out, W_out, 2) * scale
    grid = torch.clamp(grid, -1.0, 1.0)
    return grid


def create_boundary_grid(N):
    """Create grid with specific boundary coordinates to test align_corners."""
    # Test exact boundary coordinates: -1, 0, 1
    coords = torch.tensor(
        [
            [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]],  # corners and center
            [[-1.0, 1.0], [1.0, -1.0], [0.5, 0.5]],  # more test points
        ]
    )
    grid = coords.unsqueeze(0).repeat(N, 1, 1, 1)  # (N, 2, 3, 2)
    return grid


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("input_shape", [(1, 4, 4, 3), (1, 8, 8, 16), (2, 16, 16, 8)])
@pytest.mark.parametrize("output_size", [(4, 4), (8, 8)])
def test_grid_sample_nearest_align_corners_identity(device, align_corners, input_shape, output_size):
    """Test grid_sample nearest mode with identity grid for both align_corners settings."""
    N, H_in, W_in, C = input_shape
    H_out, W_out = output_size

    # Create input tensor with known pattern
    input_torch = torch.randn(N, C, H_in, W_in, dtype=torch.float32)

    # Create identity grid
    grid_torch = create_identity_grid(N, H_out, W_out)

    # PyTorch reference (NCHW format)
    expected = F.grid_sample(input_torch, grid_torch, mode="nearest", padding_mode="zeros", align_corners=align_corners)

    # Convert to TTNN format (NHWC)
    input_nhwc = input_torch.permute(0, 2, 3, 1).contiguous()
    input_ttnn = ttnn.from_torch(input_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    grid_ttnn = ttnn.from_torch(grid_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # TTNN operation
    output_ttnn = ttnn.grid_sample(
        input_ttnn, grid_ttnn, mode="nearest", padding_mode="zeros", align_corners=align_corners
    )

    # Convert back to torch for comparison (NCHW)
    output_torch = ttnn.to_torch(output_ttnn).permute(0, 3, 1, 2).contiguous()

    # Compare with PyTorch reference
    # Use rtol and atol appropriate for bfloat16 precision
    assert torch.allclose(
        output_torch, expected, rtol=1e-2, atol=1e-2
    ), f"Output mismatch for align_corners={align_corners}, shape={input_shape}, output_size={output_size}"


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("input_shape", [(1, 8, 8, 8), (2, 16, 16, 16)])
def test_grid_sample_nearest_align_corners_random(device, align_corners, input_shape):
    """Test grid_sample nearest mode with random grids."""
    N, H_in, W_in, C = input_shape
    H_out, W_out = 8, 8

    # Create input tensor
    input_torch = torch.randn(N, C, H_in, W_in, dtype=torch.float32)

    # Create random grid
    grid_torch = create_random_grid(N, H_out, W_out)

    # PyTorch reference
    expected = F.grid_sample(input_torch, grid_torch, mode="nearest", padding_mode="zeros", align_corners=align_corners)

    # Convert to TTNN format
    input_nhwc = input_torch.permute(0, 2, 3, 1).contiguous()
    input_ttnn = ttnn.from_torch(input_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    grid_ttnn = ttnn.from_torch(grid_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # TTNN operation
    output_ttnn = ttnn.grid_sample(
        input_ttnn, grid_ttnn, mode="nearest", padding_mode="zeros", align_corners=align_corners
    )

    # Convert back and compare
    output_torch = ttnn.to_torch(output_ttnn).permute(0, 3, 1, 2).contiguous()
    assert torch.allclose(
        output_torch, expected, rtol=1e-2, atol=1e-2
    ), f"Random grid test failed for align_corners={align_corners}, shape={input_shape}"


@pytest.mark.parametrize("align_corners", [False, True])
def test_grid_sample_nearest_align_corners_boundary(device, align_corners):
    """Test specific boundary coordinates where align_corners makes a difference."""
    # Create a 4x4 input with unique values at each position
    input_torch = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

    # Grid with exact boundary coordinates
    grid_torch = create_boundary_grid(1)

    # PyTorch reference
    expected = F.grid_sample(input_torch, grid_torch, mode="nearest", padding_mode="zeros", align_corners=align_corners)

    # Convert to TTNN format
    input_nhwc = input_torch.permute(0, 2, 3, 1).contiguous()
    input_ttnn = ttnn.from_torch(input_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    grid_ttnn = ttnn.from_torch(grid_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # TTNN operation
    output_ttnn = ttnn.grid_sample(
        input_ttnn, grid_ttnn, mode="nearest", padding_mode="zeros", align_corners=align_corners
    )

    # Convert back and compare
    output_torch = ttnn.to_torch(output_ttnn).permute(0, 3, 1, 2).contiguous()

    # For boundary tests, we expect exact matches (within bfloat16 precision)
    assert torch.allclose(
        output_torch, expected, rtol=1e-2, atol=1e-2
    ), f"Boundary test failed for align_corners={align_corners}"

    print(f"\nalign_corners={align_corners}:")
    print(f"Expected: {expected.flatten().numpy()}")
    print(f"Got:      {output_torch.flatten().numpy()}")


def test_grid_sample_nearest_align_corners_difference(device):
    """Verify that align_corners=True and align_corners=False produce different results."""
    # Use a case where we know the results should differ
    N, H_in, W_in, C = 1, 8, 8, 4
    H_out, W_out = 4, 4

    input_torch = torch.randn(N, C, H_in, W_in, dtype=torch.float32)

    # Grid with coordinates near boundaries where difference is pronounced
    grid_torch = torch.tensor([[[[-0.9, -0.9], [0.9, -0.9]], [[-0.9, 0.9], [0.9, 0.9]]]], dtype=torch.float32)

    # Convert to TTNN format
    input_nhwc = input_torch.permute(0, 2, 3, 1).contiguous()
    input_ttnn = ttnn.from_torch(input_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    grid_ttnn = ttnn.from_torch(grid_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Test with align_corners=False
    output_false_ttnn = ttnn.grid_sample(
        input_ttnn, grid_ttnn, mode="nearest", padding_mode="zeros", align_corners=False
    )

    # Test with align_corners=True
    output_true_ttnn = ttnn.grid_sample(input_ttnn, grid_ttnn, mode="nearest", padding_mode="zeros", align_corners=True)

    # Convert to torch
    output_false = ttnn.to_torch(output_false_ttnn)
    output_true = ttnn.to_torch(output_true_ttnn)

    # Verify both match PyTorch reference
    expected_false = F.grid_sample(input_torch, grid_torch, mode="nearest", padding_mode="zeros", align_corners=False)
    expected_true = F.grid_sample(input_torch, grid_torch, mode="nearest", padding_mode="zeros", align_corners=True)

    output_false_nchw = output_false.permute(0, 3, 1, 2).contiguous()
    output_true_nchw = output_true.permute(0, 3, 1, 2).contiguous()

    assert torch.allclose(output_false_nchw, expected_false, rtol=1e-2, atol=1e-2)
    assert torch.allclose(output_true_nchw, expected_true, rtol=1e-2, atol=1e-2)

    print(f"\nVerified that align_corners=False and align_corners=True produce results matching PyTorch")


def test_grid_sample_nearest_single_pixel(device):
    """For 1x1 input, both align_corners modes should behave identically."""
    input_torch = torch.tensor([[[[5.0]]]], dtype=torch.float32)  # (1, 1, 1, 1)
    grid_torch = torch.tensor([[[[0.0, 0.0]]]], dtype=torch.float32)  # (1, 1, 1, 2)

    # PyTorch reference for both modes
    output_false = F.grid_sample(input_torch, grid_torch, mode="nearest", align_corners=False)
    output_true = F.grid_sample(input_torch, grid_torch, mode="nearest", align_corners=True)

    # Both should be the same for 1x1 input
    assert torch.allclose(output_false, output_true)

    # Test with TTNN
    input_nhwc = input_torch.permute(0, 2, 3, 1).contiguous()
    input_ttnn = ttnn.from_torch(input_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    grid_ttnn = ttnn.from_torch(grid_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output_false_ttnn = ttnn.grid_sample(input_ttnn, grid_ttnn, mode="nearest", align_corners=False)
    output_true_ttnn = ttnn.grid_sample(input_ttnn, grid_ttnn, mode="nearest", align_corners=True)

    output_false_torch = ttnn.to_torch(output_false_ttnn).permute(0, 3, 1, 2).contiguous()
    output_true_torch = ttnn.to_torch(output_true_ttnn).permute(0, 3, 1, 2).contiguous()

    # All should match
    assert torch.allclose(output_false_torch, output_false, rtol=1e-2, atol=1e-2)
    assert torch.allclose(output_true_torch, output_true, rtol=1e-2, atol=1e-2)
    assert torch.allclose(output_false_torch, output_true_torch, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("align_corners", [False, True])
def test_grid_sample_nearest_two_pixels(device, align_corners):
    """Test 2x2 input where align_corners difference becomes visible."""
    # 2x2 input with distinct values
    input_torch = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]], dtype=torch.float32)  # (1, 1, 2, 2)

    # Sample at exact corners
    grid_torch = torch.tensor([[[[-1.0, -1.0], [1.0, 1.0]]]], dtype=torch.float32)  # (1, 1, 2, 2)

    # PyTorch reference
    expected = F.grid_sample(input_torch, grid_torch, mode="nearest", padding_mode="zeros", align_corners=align_corners)

    # TTNN
    input_nhwc = input_torch.permute(0, 2, 3, 1).contiguous()
    input_ttnn = ttnn.from_torch(input_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    grid_ttnn = ttnn.from_torch(grid_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output_ttnn = ttnn.grid_sample(input_ttnn, grid_ttnn, mode="nearest", align_corners=align_corners)
    output_torch = ttnn.to_torch(output_ttnn).permute(0, 3, 1, 2).contiguous()

    assert torch.allclose(
        output_torch, expected, rtol=1e-2, atol=1e-2
    ), f"2x2 test failed for align_corners={align_corners}"

    print(f"\n2x2 test with align_corners={align_corners}:")
    print(f"Input: {input_torch.squeeze().numpy()}")
    print(f"Expected: {expected.squeeze().numpy()}")
    print(f"Got: {output_torch.squeeze().numpy()}")


@pytest.mark.parametrize("align_corners", [False, True])
def test_grid_sample_nearest_align_corners_float32_grid(device, align_corners):
    """Test with FLOAT32 grid for higher precision."""
    N, H_in, W_in, C = 1, 8, 8, 8
    H_out, W_out = 4, 4

    input_torch = torch.randn(N, C, H_in, W_in, dtype=torch.float32)
    grid_torch = create_identity_grid(N, H_out, W_out)

    # PyTorch reference
    expected = F.grid_sample(input_torch, grid_torch, mode="nearest", padding_mode="zeros", align_corners=align_corners)

    # TTNN with FLOAT32 grid
    input_nhwc = input_torch.permute(0, 2, 3, 1).contiguous()
    input_ttnn = ttnn.from_torch(input_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    grid_ttnn = ttnn.from_torch(grid_torch, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output_ttnn = ttnn.grid_sample(
        input_ttnn, grid_ttnn, mode="nearest", padding_mode="zeros", align_corners=align_corners
    )

    output_torch = ttnn.to_torch(output_ttnn).permute(0, 3, 1, 2).contiguous()
    assert torch.allclose(
        output_torch, expected, rtol=1e-2, atol=1e-2
    ), f"FLOAT32 grid test failed for align_corners={align_corners}"


@pytest.mark.parametrize("align_corners", [False, True])
def test_grid_sample_nearest_align_corners_backward_compatibility(device, align_corners):
    """Ensure the implementation matches the plan's mathematical formulas."""
    # Test case from the align_corners_plan.md
    input_shape = (1, 4, 4, 1)
    input_torch = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

    # Test specific coordinates that highlight the formula difference
    # For H=4, W=4:
    # align_corners=False: height_scale=2.0, height_offset=1.5
    #   coord=-1.0 → pixel = -1.0 * 2.0 + 1.5 = -0.5 (rounds to 0)
    #   coord=1.0  → pixel = 1.0 * 2.0 + 1.5 = 3.5 (rounds to 4, out of bounds)
    # align_corners=True:  height_scale=1.5, height_offset=1.5
    #   coord=-1.0 → pixel = -1.0 * 1.5 + 1.5 = 0.0 (pixel 0)
    #   coord=1.0  → pixel = 1.0 * 1.5 + 1.5 = 3.0 (pixel 3)

    grid_torch = torch.tensor([[[[0.0, -1.0]], [[0.0, 1.0]]]], dtype=torch.float32)

    # PyTorch reference
    expected = F.grid_sample(input_torch, grid_torch, mode="nearest", padding_mode="zeros", align_corners=align_corners)

    # TTNN
    input_nhwc = input_torch.permute(0, 2, 3, 1).contiguous()
    input_ttnn = ttnn.from_torch(input_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    grid_ttnn = ttnn.from_torch(grid_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output_ttnn = ttnn.grid_sample(
        input_ttnn, grid_ttnn, mode="nearest", padding_mode="zeros", align_corners=align_corners
    )

    output_torch = ttnn.to_torch(output_ttnn).permute(0, 3, 1, 2).contiguous()

    assert torch.allclose(
        output_torch, expected, rtol=1e-2, atol=1e-2
    ), f"Formula verification failed for align_corners={align_corners}"

    print(f"\nFormula verification for align_corners={align_corners}:")
    print(f"Expected: {expected.flatten().numpy()}")
    print(f"Got:      {output_torch.flatten().numpy()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
