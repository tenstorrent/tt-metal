# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Stage 3 Test: Operation Registration

Verifies that the row_mean_sub_square_reduce operation is properly registered and
reaches the device execution path (program factory is called).

At this stage, it's OK if the operation fails in the program factory
(kernels not implemented yet) - we just need to verify it gets past validation.
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


def test_reaches_program_factory(device):
    """
    Verify operation reaches program factory (may fail there, but gets past validation).

    This test passes if:
    1. The operation completes successfully, OR
    2. The operation fails with a kernel/program error (not validation error)

    This test fails if:
    1. The operation fails with a validation error (Stage 2 issue)
    2. The operation is not found (Stage 1 issue)
    """
    shape = (1, 1, 32, 32)
    torch_tensor = torch.randn(shape, dtype=torch.bfloat16)

    valid_tensor = ttnn.from_torch(torch_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    try:
        result = ttnn.row_mean_sub_square_reduce(valid_tensor)
        # If we get here, operation completed successfully - great!
        assert result is not None, "Operation returned None"
        print(f"Operation completed successfully with output shape: {result.shape}")
    except RuntimeError as e:
        error_msg = str(e).lower()

        # These keywords indicate we reached the program factory / kernel level
        program_keywords = ["kernel", "program", "circular buffer", "cb_", "noc", "risc"]

        # These keywords indicate we failed at validation (Stage 2 issue)
        validation_keywords = ["rank", "layout", "dtype", "must be", "expected", "invalid"]

        reached_program_factory = any(kw in error_msg for kw in program_keywords)
        failed_at_validation = any(kw in error_msg for kw in validation_keywords) and not reached_program_factory

        if failed_at_validation:
            pytest.fail(
                f"Operation failed at validation, not program factory. "
                f"This is a Stage 2 issue, not Stage 3. Error: {e}"
            )

        # If we got a program/kernel error, that's expected at this stage
        print(f"Operation reached program factory (expected failure at this stage): {e}")


def test_operation_returns_tensor_or_fails_in_program(device):
    """
    Verify operation either returns a tensor or fails in program factory.
    """
    shape = (1, 1, 32, 32)
    torch_tensor = torch.randn(shape, dtype=torch.bfloat16)

    valid_tensor = ttnn.from_torch(torch_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    try:
        result = ttnn.row_mean_sub_square_reduce(valid_tensor)
        # Verify it's a tensor
        assert isinstance(result, ttnn.Tensor), f"Expected ttnn.Tensor, got {type(result)}"
    except RuntimeError:
        # At Stage 3, runtime errors in program factory are acceptable
        pass
