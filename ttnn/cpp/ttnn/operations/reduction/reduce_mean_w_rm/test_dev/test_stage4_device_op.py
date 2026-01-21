# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

# NOTE: Use the built-in `device` fixture from conftest.py - do NOT define your own.
# Before running: 'tt-smi -ls' to verify device, 'tt-smi -r 0' to reset (see CLAUDE.md)


def test_device_op_called(device):
    """Operation should reach program factory, not fail at validation"""
    input_tensor = ttnn.from_torch(
        torch.randn(1, 1, 32, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    with pytest.raises(RuntimeError) as exc:
        ttnn.reduce_mean_w_rm(input_tensor)

    # Error should be about program/kernel/factory, not validation
    error_msg = str(exc.value).lower()
    assert (
        "kernel" in error_msg or "program" in error_msg or "factory" in error_msg or "not yet implemented" in error_msg
    ), f"Expected program/kernel/factory error, got: {exc.value}"


def test_program_factory_selected(device):
    """select_program_factory should return valid factory type"""
    input_tensor = ttnn.from_torch(
        torch.randn(1, 1, 64, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # Operation should not fail at factory selection
    with pytest.raises(RuntimeError) as exc:
        ttnn.reduce_mean_w_rm(input_tensor)

    # Should not mention "select" or "factory selection"
    error_msg = str(exc.value).lower()
    assert "select" not in error_msg, f"Factory selection failed: {exc.value}"
