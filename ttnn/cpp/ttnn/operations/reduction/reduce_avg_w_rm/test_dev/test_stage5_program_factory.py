# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Stage 5: Program Factory Structure Tests

Test that the program factory creates CBs and work distribution before failing at kernel creation.
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


def test_program_factory_creates_cbs(device):
    """Program factory should create CBs before failing at kernel creation"""
    torch_input = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with pytest.raises(RuntimeError) as exc:
        ttnn.reduce_avg_w_rm(input_tensor)

    error_msg = str(exc.value).lower()
    # Should fail at kernel, not at CB or program
    assert "kernel" in error_msg, f"Expected kernel error, got: {exc.value}"
    assert "circular" not in error_msg, f"Should not fail at CB creation: {exc.value}"


def test_work_distribution(device):
    """Should handle various input sizes"""
    # Small input (1 tile row: H=32, W=64)
    small_input = ttnn.from_torch(
        torch.randn(1, 1, 32, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # Larger input (multiple tile rows)
    large_input = ttnn.from_torch(
        torch.randn(2, 3, 64, 128, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    for inp in [small_input, large_input]:
        with pytest.raises(RuntimeError) as exc:
            ttnn.reduce_avg_w_rm(inp)
        # Should reach kernel creation for all sizes
        error_msg = str(exc.value).lower()
        assert "kernel" in error_msg, f"Expected kernel error, got: {exc.value}"
