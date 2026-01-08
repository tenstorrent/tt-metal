# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.fixture
def device():
    # Note: Run 'tt-smi -ls' first to verify device 0 is available
    # and 'tt-smi -r 0' to reset if needed (see CLAUDE.md)
    with ttnn.manage_device(device_id=0) as dev:
        yield dev


def test_device_op_called(device):
    """Operation should reach device and execute successfully"""
    # Create ROW_MAJOR input tensor
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Operation should succeed
    output = ttnn.reduce_avg_w_rm(input_tensor)
    assert output is not None, "Operation should return a tensor"


def test_program_factory_selected(device):
    """select_program_factory should return valid factory type and execute"""
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Operation should not fail at factory selection
    output = ttnn.reduce_avg_w_rm(input_tensor)

    # Verify output has expected shape (height unchanged, width=32)
    expected_shape = (1, 1, 32, 32)
    assert list(output.shape) == list(expected_shape), f"Expected shape {expected_shape}, got {output.shape}"
