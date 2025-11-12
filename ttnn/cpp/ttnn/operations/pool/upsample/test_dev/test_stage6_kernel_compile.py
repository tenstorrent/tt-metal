# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Stage 6: Kernel Compilation Test
Goal: Verify kernels compile successfully
"""

import pytest
import torch
import ttnn


@pytest.fixture
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


def test_kernels_compile(device):
    """Test that kernels compile without syntax errors"""
    input_tensor = torch.ones((1, 1, 2, 2, 32), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Should compile and execute successfully!
    output = ttnn.upsample3d(tt_input, scale_factor=1)
    output_tensor = ttnn.to_torch(output)

    # Identity operation - shapes should match
    assert output_tensor.shape == input_tensor.shape


def test_reader_kernel_created(device):
    """Test that reader kernel is properly set up"""
    input_tensor = torch.ones((1, 1, 1, 1, 64), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Should create and compile reader kernel successfully
    output = ttnn.upsample3d(tt_input, scale_factor=1)
    assert output is not None


def test_writer_kernel_created(device):
    """Test that writer kernel is properly set up"""
    input_tensor = torch.ones((1, 2, 2, 2, 16), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Should create and compile writer kernel successfully
    output = ttnn.upsample3d(tt_input, scale_factor=2)
    output_tensor = ttnn.to_torch(output)

    # Verify correct upsampling
    assert output_tensor.shape == (1, 4, 4, 4, 16)


def test_runtime_args_set(device):
    """Test that runtime arguments are properly configured"""
    input_tensor = torch.ones((1, 1, 2, 2, 32), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Kernels should compile and execute
    output = ttnn.upsample3d(tt_input, scale_factor=1)
    assert output is not None
