# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

# NOTE: Use the built-in `device` fixture from conftest.py - do NOT define your own.
# Before running: 'tt-smi -ls' to verify device, 'tt-smi -r 0' to reset (see CLAUDE.md)


def test_kernels_compile_at_runtime(device):
    """Kernels should compile without errors when operation runs"""
    input_tensor = ttnn.from_torch(
        torch.randn(1, 1, 32, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # Run the operation - kernel compilation happens here
    # Empty stub kernels will produce garbage output, but should not error
    try:
        result = ttnn.reduce_mean_w_rm(input_tensor)
    except RuntimeError as e:
        error_str = str(e)
        if ".cpp" in error_str or "error:" in error_str.lower():
            pytest.fail(f"Kernel compilation failed: {e}")
        raise


def test_output_shape_correct(device):
    """Output should have correct shape (width=1 logical, 32 padded)"""
    input_tensor = ttnn.from_torch(
        torch.randn(1, 1, 32, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # Stub kernels produce garbage values but correct shape
    result = ttnn.reduce_mean_w_rm(input_tensor)

    # Check shape
    assert result.shape[-1] == 1, f"Expected logical width=1, got {result.shape[-1]}"
    assert result.padded_shape()[-1] == 32, f"Expected padded width=32, got {result.padded_shape()[-1]}"
