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


def test_program_factory_creates_cbs(device):
    """Program factory should create CBs and execute successfully"""
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


def test_work_distribution(device):
    """Should handle various input sizes"""
    # Small input (1 row of tiles)
    torch_small = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    small_input = ttnn.from_torch(
        torch_small,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Large input (many rows of tiles)
    torch_large = torch.randn(1, 32, 64, 64, dtype=torch.bfloat16)
    large_input = ttnn.from_torch(
        torch_large,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for inp, name in [(small_input, "small"), (large_input, "large")]:
        output = ttnn.reduce_avg_w_rm(inp)
        assert output is not None, f"Operation should succeed for {name} input"
