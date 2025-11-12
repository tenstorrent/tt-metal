# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Stage 3: TTNN Operation Registration Test
Goal: Verify the operation is properly registered with TTNN framework
"""

import pytest
import torch
import ttnn
import inspect


@pytest.fixture
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


def test_operation_signature():
    """Test that upsample3d is properly registered as a TTNN operation"""
    # Registered operations have function_args and function_kwargs wrapper
    sig = inspect.signature(ttnn.upsample3d)
    params = list(sig.parameters.keys())

    # TTNN registered operations expose a standardized signature
    assert len(params) >= 2, "Operation should have function_args and function_kwargs"
    assert "function_args" in params or "function_kwargs" in params, "Not a properly registered TTNN operation"


def test_operation_docstring():
    """Test that operation has proper documentation"""
    assert ttnn.upsample3d.__doc__ is not None, "Missing docstring"
    doc_lower = ttnn.upsample3d.__doc__.lower()
    assert "3d" in doc_lower or "upsample" in doc_lower, "Docstring doesn't describe operation"


def test_operation_with_memory_config(device):
    """Test that memory_config parameter is accepted"""
    input_5d = torch.ones((1, 2, 2, 2, 4), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_5d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Should accept memory config without error at parameter level
    try:
        ttnn.upsample3d(tt_input, scale_factor=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    except RuntimeError as e:
        # Should fail at device operation level, not parameter level
        assert "device" in str(e).lower() or "not implemented" in str(e).lower()


def test_registered_operation_dispatch(device):
    """Test that operation goes through TTNN registration system"""
    input_5d = torch.ones((1, 2, 2, 2, 4), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_5d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=2)

    # Should mention operation or device in error (proper dispatch happening)
    error_msg = str(exc_info.value).lower()
    assert "operation" in error_msg or "device" in error_msg or "not implemented" in error_msg
