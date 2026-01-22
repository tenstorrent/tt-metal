# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

# NOTE: Use the built-in `device` fixture from conftest.py - do NOT define your own.


def test_kernels_compile_at_runtime(device):
    """Kernels should compile without errors when operation runs"""
    input_tensor = ttnn.from_torch(
        torch.randn(32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    gamma_tensor = ttnn.from_torch(
        torch.ones(32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    beta_tensor = ttnn.from_torch(
        torch.zeros(32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    # Run the operation - kernel compilation happens here
    # Stub kernels will produce garbage output, but should not error
    try:
        result = ttnn.layer_norm_w_rm(input_tensor, gamma_tensor, beta_tensor, epsilon=1e-5)
    except RuntimeError as e:
        error_str = str(e)
        if ".cpp" in error_str or "error:" in error_str.lower():
            pytest.fail(f"Kernel compilation failed: {e}")
        raise


def test_program_executes_without_hang(device):
    """Program should execute without hanging (stub kernels should pass data through)"""
    input_tensor = ttnn.from_torch(
        torch.randn(32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    gamma_tensor = ttnn.from_torch(
        torch.ones(32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    beta_tensor = ttnn.from_torch(
        torch.zeros(32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    # Stub kernels should complete without hang
    result = ttnn.layer_norm_w_rm(input_tensor, gamma_tensor, beta_tensor, epsilon=1e-5)
    assert result is not None
    # Check output shape matches input (stubs pass through shape)
    assert result.shape == input_tensor.shape
