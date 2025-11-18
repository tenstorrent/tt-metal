import ttnn
import torch
import pytest


def test_nearest_mode_validation():
    """Test parameter validation for nearest mode"""
    device = ttnn.open_device(device_id=0)

    # Test 1: Invalid tensor rank
    input_3d = torch.ones((1, 1, 2), dtype=torch.bfloat16)
    grid = torch.ones((1, 2, 2, 2), dtype=torch.bfloat16)

    input_t = ttnn.from_torch(input_3d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    grid_t = ttnn.from_torch(grid, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.grid_sample(input_t, grid_t, mode="nearest")
    assert "must be 4D" in str(exc_info.value)

    # Test 2: Invalid grid shape
    input_4d = torch.ones((1, 1, 2, 2), dtype=torch.bfloat16)
    grid_wrong = torch.ones((1, 2, 2, 3), dtype=torch.bfloat16)  # Wrong last dim

    input_t = ttnn.from_torch(input_4d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    grid_t = ttnn.from_torch(grid_wrong, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.grid_sample(input_t, grid_t, mode="nearest")
    assert "last dimension must" in str(exc_info.value).lower() or "multiple of" in str(exc_info.value).lower()

    # Test 3: Valid inputs should fail at device op level (still not implemented)
    input_valid = torch.ones((1, 1, 4, 4), dtype=torch.bfloat16)
    grid_valid = torch.ones((1, 2, 2, 2), dtype=torch.bfloat16)

    input_t = ttnn.from_torch(input_valid, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    grid_t = ttnn.from_torch(grid_valid, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.grid_sample(input_t, grid_t, mode="nearest")
    # Should still fail with "nearest mode not yet implemented" since we haven't fully implemented it
    assert "nearest" in str(exc_info.value).lower()

    ttnn.close_device(device)
