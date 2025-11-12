# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Stage 5: Program Factory API Test
Goal: Verify program factory is called and creates basic program structure
"""

import pytest
import torch
import ttnn


@pytest.fixture
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


def test_program_creation_attempted(device):
    """Test that program factory is invoked"""
    input_tensor = torch.ones((1, 1, 2, 2, 4), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=1)  # Identity operation

    # Should mention kernel (compilation or missing kernel file)
    error_msg = str(exc_info.value).lower()
    assert "kernel" in error_msg or "buffer" in error_msg or "compile" in error_msg


def test_single_core_program(device):
    """Test minimal single-core configuration"""
    # Very small tensor for single core
    input_tensor = torch.ones((1, 1, 1, 1, 32), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=1)

    # Should get to kernel creation
    assert "kernel" in str(exc_info.value).lower()


def test_multi_core_work_distribution(device):
    """Test that work distribution is calculated"""
    # Larger tensor that should use multiple cores
    input_tensor = torch.ones((1, 4, 8, 8, 32), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=2)

    # Should attempt kernel creation
    error_msg = str(exc_info.value).lower()
    assert "kernel" in error_msg or "reader" in error_msg or "writer" in error_msg


def test_circular_buffer_creation(device):
    """Test that circular buffers are set up"""
    input_tensor = torch.ones((1, 2, 2, 2, 64), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=2)

    # Should mention kernels (CBs are created before kernels)
    assert "kernel" in str(exc_info.value).lower()
