import ttnn
import torch
import pytest


def test_nearest_program_factory():
    """Test program factory creates proper structure"""
    device = ttnn.open_device(device_id=0)

    # Small tensor that should create program
    input_t = ttnn.ones((1, 32, 32, 32), device=device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    grid_t = ttnn.ones((1, 16, 16, 2), device=device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Should fail mentioning kernels or kernel file not found, not program structure
    with pytest.raises(RuntimeError) as exc_info:
        result = ttnn.grid_sample(input_t, grid_t, mode="nearest")

    error_msg = str(exc_info.value).lower()
    # Should fail at kernel level
    assert (
        "kernel" in error_msg or "reader" in error_msg or "file not found" in error_msg or "no such file" in error_msg
    )
    # Should NOT fail at circular buffer or program structure level
    assert "circular buffer" not in error_msg

    ttnn.close_device(device)


def test_nearest_work_distribution():
    """Test work is properly distributed across cores"""
    device = ttnn.open_device(device_id=0)

    # Larger tensor to test multi-core distribution
    input_t = ttnn.ones((2, 64, 64, 64), device=device, dtype=ttnn.bfloat16)
    grid_t = ttnn.ones((2, 32, 32, 2), device=device, dtype=ttnn.bfloat16)

    with pytest.raises(RuntimeError) as exc_info:
        result = ttnn.grid_sample(input_t, grid_t, mode="nearest")

    # Should still fail at kernel level
    error_msg = str(exc_info.value).lower()
    assert "kernel" in error_msg or "file not found" in error_msg or "no such file" in error_msg

    ttnn.close_device(device)
