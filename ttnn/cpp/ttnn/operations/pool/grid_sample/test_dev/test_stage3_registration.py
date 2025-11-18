import ttnn
import torch
import pytest


def test_nearest_mode_registration():
    """Test that nearest mode is properly registered with TTNN"""

    # Test: Mode parameter accepts 'nearest' and fails at the right level
    device = ttnn.open_device(device_id=0)

    input_t = ttnn.ones((1, 1, 4, 4), device=device, dtype=ttnn.bfloat16)
    grid_t = ttnn.ones((1, 2, 2, 2), device=device, dtype=ttnn.bfloat16)

    # Should fail at device operation level, not registration
    with pytest.raises(RuntimeError) as exc_info:
        result = ttnn.grid_sample(input_t, grid_t, mode="nearest")

    # Error should mention nearest or program, not registration
    error_msg = str(exc_info.value).lower()
    assert "nearest" in error_msg or "program" in error_msg or "device" in error_msg
    assert "not registered" not in error_msg
    assert "unknown" not in error_msg  # Would indicate parameter not recognized

    ttnn.close_device(device)
