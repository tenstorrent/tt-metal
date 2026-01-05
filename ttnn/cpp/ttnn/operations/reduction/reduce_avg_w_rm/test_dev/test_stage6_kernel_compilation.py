# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Stage 6: Kernel Compilation Tests

Test that kernels compile at runtime and execute without hanging.
Stub kernels may produce garbage output - correctness is Stage 7's job.
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


def test_kernels_compile_at_runtime(device):
    """Kernels should compile without errors when operation runs"""
    torch_input = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Run the operation - kernel compilation happens here
    # If there's a compilation error, it will raise RuntimeError
    # with messages containing the kernel source path or "error"
    try:
        result = ttnn.reduce_avg_w_rm(input_tensor)
    except RuntimeError as e:
        error_str = str(e)
        # Check if this is a kernel compilation error
        # Compilation errors typically contain source file paths or "error:"
        if ".cpp" in error_str or "error:" in error_str.lower():
            pytest.fail(f"Kernel compilation failed: {e}")
        # Re-raise if it's a different runtime error
        raise


def test_program_executes_without_hang(device):
    """Program should execute without hanging (stub kernels may produce garbage output)"""
    torch_input = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Should complete without hanging
    result = ttnn.reduce_avg_w_rm(input_tensor)

    # Basic sanity checks
    assert result is not None


def test_output_shape_dtype(device):
    """Output tensor should have correct shape and dtype (physical shape [N, C, H, 32])"""
    torch_input = torch.randn(2, 3, 64, 128, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    result = ttnn.reduce_avg_w_rm(input_tensor)

    # Shape from spec: output is [N, C, H, 32] (physical width=32)
    expected_shape = torch.Size([2, 3, 64, 32])
    assert result.shape == expected_shape, f"Expected {expected_shape}, got {result.shape}"

    # Dtype should match input
    assert result.dtype == input_tensor.dtype


def test_multi_tile_row_execution(device):
    """Should handle multiple tile rows across multiple cores"""
    # Multiple tile rows: H=128 (4 tile rows), W=96
    torch_input = torch.randn(2, 2, 128, 96, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    result = ttnn.reduce_avg_w_rm(input_tensor)

    # Should complete and have correct shape
    expected_shape = torch.Size([2, 2, 128, 32])
    assert result.shape == expected_shape, f"Expected {expected_shape}, got {result.shape}"
