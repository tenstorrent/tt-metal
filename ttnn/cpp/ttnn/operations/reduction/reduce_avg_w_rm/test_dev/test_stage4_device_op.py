# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Stage 4: Device Operation Tests

Test that the operation reaches the program factory without failing at validation.
"""

import pytest
import torch
import ttnn


@pytest.fixture
def device():
    """
    Device fixture with proper management.

    Note: Before running tests:
    1. Run 'tt-smi -ls' to verify device 0 is available
    2. Run 'tt-smi -r 0' to reset if needed (see CLAUDE.md)
    """
    with ttnn.manage_device(device_id=0) as dev:
        yield dev


def test_device_op_called(device):
    """Operation should reach program factory, not fail at validation"""
    # Create row-major input tensor
    torch_input = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Operation should fail at program factory (not validation)
    with pytest.raises(RuntimeError) as exc:
        ttnn.reduce_avg_w_rm(input_tensor)

    # Error should be about program/kernel/factory, not validation
    error_msg = str(exc.value).lower()
    assert any(
        keyword in error_msg for keyword in ["kernel", "program", "factory", "not yet implemented"]
    ), f"Expected program/kernel/factory error, got: {exc.value}"


def test_program_factory_selected(device):
    """select_program_factory should return valid factory type"""
    # Create valid input tensor
    torch_input = torch.randn(1, 2, 64, 96, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Operation should not fail at factory selection
    with pytest.raises(RuntimeError) as exc:
        ttnn.reduce_avg_w_rm(input_tensor)

    # Should not mention "select" or "factory selection"
    error_msg = str(exc.value).lower()
    assert "select" not in error_msg, f"Factory selection should succeed, got: {exc.value}"


def test_validation_passes_for_valid_input(device):
    """Valid input should pass validation and reach factory"""
    # Create valid input: 4D, ROW_MAJOR, tile-aligned
    torch_input = torch.randn(2, 3, 64, 128, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Should fail at program factory implementation, not validation
    with pytest.raises(RuntimeError) as exc:
        ttnn.reduce_avg_w_rm(input_tensor)

    error_msg = str(exc.value).lower()
    # Should not be a validation error
    assert not any(
        word in error_msg for word in ["must be 4d", "row_major", "interleaved", "multiple of 32"]
    ), f"Validation should pass for valid input, got: {exc.value}"
