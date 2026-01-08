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


def test_kernels_compile_at_runtime(device):
    """Kernels should compile without errors when operation runs"""
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

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
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Should complete without hanging
    result = ttnn.reduce_avg_w_rm(input_tensor)

    # Basic sanity checks
    assert result is not None


def test_output_shape_dtype(device):
    """Output tensor should have correct shape and dtype"""
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.reduce_avg_w_rm(input_tensor)

    # Shape from spec's "Output Tensor Specification"
    # Output shape: [N, C, H, 32] (physical, tile-aligned)
    expected_shape = torch.Size([1, 1, 32, 32])
    assert result.shape == expected_shape, f"Expected {expected_shape}, got {result.shape}"

    # Dtype should match input
    assert result.dtype == input_tensor.dtype

    # Layout should be ROW_MAJOR
    assert result.layout == ttnn.ROW_MAJOR_LAYOUT


def test_multi_tile_execution(device):
    """Should handle multi-tile inputs across multiple cores"""
    # Multiple tile rows to test work distribution
    torch_input = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.reduce_avg_w_rm(input_tensor)

    # Should complete and have correct shape
    # Output: [1, 4, 64, 32] (width reduced to 32)
    expected_shape = torch.Size([1, 4, 64, 32])
    assert result.shape == expected_shape, f"Expected {expected_shape}, got {result.shape}"
