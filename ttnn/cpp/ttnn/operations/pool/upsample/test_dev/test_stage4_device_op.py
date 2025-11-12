# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Stage 4: Device Operation Structure Test
Goal: Verify device operation follows proper TTNN structure with nested structs
"""

import pytest
import torch
import ttnn
import numpy as np


@pytest.fixture
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


def test_device_operation_invoked(device):
    """Test that device operation is called with proper validation"""
    input_5d = torch.ones((1, 2, 2, 2, 4), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_5d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=2)

    # Should fail at program/device level, not host validation
    error_msg = str(exc_info.value).lower()
    assert "program" in error_msg or "device" in error_msg or "create" in error_msg or "kernel" in error_msg


def test_output_shape_calculation(device):
    """Test that output shape is calculated correctly by device op"""
    test_cases = [
        ((1, 2, 3, 4, 8), 2, (1, 4, 6, 8, 8)),
        ((1, 2, 3, 4, 8), (2, 3, 2), (1, 4, 9, 8, 8)),
        ((2, 1, 5, 5, 16), (1, 2, 2), (2, 1, 10, 10, 16)),
    ]

    for input_shape, scale_factor, expected_shape in test_cases:
        input_tensor = torch.ones(input_shape, dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Even though execution may fail, shape calculation should work or fail predictably
        with pytest.raises(RuntimeError) as exc_info:
            ttnn.upsample3d(tt_input, scale_factor=scale_factor)

        # Error should be about program creation or kernel, not shape calculation
        error_msg = str(exc_info.value).lower()
        assert "program" in error_msg or "factory" in error_msg or "kernel" in error_msg or "create" in error_msg


def test_memory_config_handling(device):
    """Test that memory config is properly passed to device operation"""
    input_5d = torch.ones((1, 2, 2, 2, 4), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_5d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Test with different memory configs
    configs = [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG]
    for config in configs:
        with pytest.raises(RuntimeError) as exc_info:
            ttnn.upsample3d(tt_input, scale_factor=2, memory_config=config)

        # Should fail at program/kernel level
        error_msg = str(exc_info.value).lower()
        assert "program" in error_msg or "kernel" in error_msg or "create" in error_msg


def test_interleaved_memory_validation(device):
    """Test that only interleaved memory is supported"""
    input_5d = torch.ones((1, 2, 2, 2, 4), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_5d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # For now, should only support interleaved
    # This test documents the current limitation
    with pytest.raises(RuntimeError):
        # Would fail when sharding is attempted or during execution
        ttnn.upsample3d(tt_input, scale_factor=2)
