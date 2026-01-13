# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Stage 2 Test: Input Validation

Verifies that the row_mean_sub_square_reduce operation correctly validates inputs
and raises appropriate errors for invalid inputs.
"""

import pytest
import torch
import ttnn


@pytest.fixture
def device():
    """Get a device for testing."""
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


def test_wrong_rank_raises(device):
    """Verify wrong tensor rank raises RuntimeError."""
    # Create tensor with wrong rank (expected: 4D)
    wrong_shape = (32, 32)  # 2D instead of 4D
    torch_tensor = torch.randn(wrong_shape, dtype=torch.bfloat16)
    wrong_rank_tensor = ttnn.from_torch(torch_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.row_mean_sub_square_reduce(wrong_rank_tensor)

    # Error should mention rank
    error_msg = str(exc_info.value).lower()
    assert (
        "rank" in error_msg or "dimension" in error_msg or "shape" in error_msg
    ), f"Error should mention rank issue, got: {exc_info.value}"


def test_wrong_layout_raises(device):
    """Verify wrong layout raises RuntimeError."""
    shape = (1, 1, 32, 32)
    torch_tensor = torch.randn(shape, dtype=torch.bfloat16)

    # Operation expects ROW_MAJOR, give it TILE
    wrong_layout_tensor = ttnn.from_torch(torch_tensor, device=device, layout=ttnn.TILE_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.row_mean_sub_square_reduce(wrong_layout_tensor)

    # Error should mention layout
    error_msg = str(exc_info.value).lower()
    assert (
        "layout" in error_msg or "row_major" in error_msg or "tile" in error_msg
    ), f"Error should mention layout issue, got: {exc_info.value}"


def test_valid_input_does_not_raise_validation_error(device):
    """Verify valid input passes validation (may fail later in program factory)."""
    shape = (1, 1, 32, 32)
    torch_tensor = torch.randn(shape, dtype=torch.bfloat16)

    valid_tensor = ttnn.from_torch(torch_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    try:
        ttnn.row_mean_sub_square_reduce(valid_tensor)
    except RuntimeError as e:
        error_msg = str(e).lower()
        # Should NOT fail on validation - if it fails, it should be in program factory
        validation_keywords = ["rank", "layout", "dtype", "dimension", "must be", "expected"]
        is_validation_error = any(kw in error_msg for kw in validation_keywords)

        # If it's a validation error, that's a test failure
        # If it's a kernel/program error, that's expected (Stage 2 doesn't require working kernels)
        if is_validation_error and "kernel" not in error_msg and "program" not in error_msg:
            pytest.fail(f"Valid input raised validation error: {e}")
        # Otherwise, non-validation errors are OK at this stage
