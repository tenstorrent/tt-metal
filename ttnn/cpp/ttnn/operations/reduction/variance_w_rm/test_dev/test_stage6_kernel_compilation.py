import pytest
import torch
import ttnn

# NOTE: Use the built-in `device` fixture from conftest.py - do NOT define your own.
# Before running: 'tt-smi -ls' to verify device, 'tt-smi -r 0' to reset (see CLAUDE.md)


def test_kernels_compile(device):
    """Stub kernels should compile at runtime and produce output with correct shape"""
    input_tensor = ttnn.from_torch(
        torch.randn(1, 1, 32, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # Should complete without hanging (stub kernels with proper CB sync)
    output = ttnn.variance_w_rm(input_tensor)

    # Check output shape (reduced width to 1 logical, 32 padded)
    assert output.shape[-1] == 1, f"Output logical width should be 1, got {output.shape[-1]}"
    assert output.padded_shape()[-1] == 32, f"Output padded width should be 32, got {output.padded_shape()[-1]}"

    # Values will be garbage (stub kernels) - this is expected
    # Correctness is verified in Stage 7


def test_no_hang_on_multi_tile_rows(device):
    """Multiple tile-rows should not cause deadlock"""
    # 2 tile-rows (64 rows), 4 tiles wide (128 elements)
    input_tensor = ttnn.from_torch(
        torch.randn(1, 1, 64, 128, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # Should complete without hanging
    output = ttnn.variance_w_rm(input_tensor)

    # Check shape
    assert output.shape[-1] == 1
    assert output.shape[-2] == 64
