# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

# NOTE: Use the built-in `device` fixture from conftest.py - do NOT define your own.
# Before running: 'tt-smi -ls' to verify device, 'tt-smi -r 0' to reset (see CLAUDE.md)


def test_program_factory_creates_cbs(device):
    """Program factory should create CBs before failing at kernel creation"""
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

    with pytest.raises(RuntimeError) as exc:
        ttnn.layer_norm_w_rm(input_tensor, gamma_tensor, beta_tensor, epsilon=1e-5)

    error_msg = str(exc.value).lower()
    # Should fail at kernel, not at CB or program
    assert "kernel" in error_msg or "not yet implemented" in error_msg, f"Expected kernel error, got: {exc.value}"
    assert "circular" not in error_msg, f"Should not fail at CB creation: {exc.value}"


def test_work_distribution(device):
    """Should handle various input sizes"""
    # Small input (1 tile-row)
    small_input = ttnn.from_torch(
        torch.randn(32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    small_gamma = ttnn.from_torch(
        torch.ones(32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    small_beta = ttnn.from_torch(
        torch.zeros(32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    # Large input (multiple tile-rows)
    large_input = ttnn.from_torch(
        torch.randn(64, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    large_gamma = ttnn.from_torch(
        torch.ones(64, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    large_beta = ttnn.from_torch(
        torch.zeros(64, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    for inp, gamma, beta in [(small_input, small_gamma, small_beta), (large_input, large_gamma, large_beta)]:
        with pytest.raises(RuntimeError) as exc:
            ttnn.layer_norm_w_rm(inp, gamma, beta, epsilon=1e-5)
        # Should reach kernel creation for all sizes
        error_str = str(exc.value).lower()
        assert (
            "kernel" in error_str or "not yet implemented" in error_str
        ), f"Expected kernel error, got: {exc.value}"
