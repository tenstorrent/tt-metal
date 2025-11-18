import ttnn
import torch
import pytest


def test_nearest_mode_exists():
    """Test that grid_sample accepts mode='nearest'"""
    device = ttnn.open_device(device_id=0)

    # Create minimal tensors
    input_tensor = torch.ones((1, 1, 2, 2), dtype=torch.bfloat16)
    grid = torch.tensor([[[[-1, -1], [1, 1]]]], dtype=torch.bfloat16)

    input_t = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    grid_t = ttnn.from_torch(grid, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # This should not raise AttributeError but should fail with RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        result = ttnn.grid_sample(input_t, grid_t, mode="nearest")

    assert "nearest mode not yet implemented" in str(exc_info.value).lower()

    ttnn.close_device(device)
