import ttnn
import torch
import pytest


def test_nearest_device_operation():
    """Test device operation validates and computes output shape"""
    device = ttnn.open_device(device_id=0)

    # Test 1: Validate memory config and proper routing
    input_t = ttnn.ones((1, 8, 8, 32), device=device, dtype=ttnn.bfloat16)
    grid_t = ttnn.ones((1, 4, 4, 2), device=device, dtype=ttnn.bfloat16)

    # Should compute output shape correctly: [1, 2, 4, 4]
    with pytest.raises(RuntimeError) as exc_info:
        result = ttnn.grid_sample(input_t, grid_t, mode="nearest")

    # Should fail at program factory creation, not device op
    error_msg = str(exc_info.value).lower()
    assert "program" in error_msg or "factory" in error_msg or "create_program" in error_msg or "kernel" in error_msg
    # Should NOT fail at validation
    assert "validate" not in error_msg
    assert "must be 4d" not in error_msg

    # Test 2: Batch size mismatch should be caught
    input_batch2 = ttnn.ones((2, 8, 8, 32), device=device, dtype=ttnn.bfloat16)
    grid_batch1 = ttnn.ones((1, 4, 4, 2), device=device, dtype=ttnn.bfloat16)

    with pytest.raises(RuntimeError) as exc_info:
        result = ttnn.grid_sample(input_batch2, grid_batch1, mode="nearest")
    assert "batch size" in str(exc_info.value).lower()

    ttnn.close_device(device)
