# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Stage 1: API Existence Test
Goal: Verify the Python API exists and can be imported
"""

import pytest
import torch
import ttnn


def test_upsample3d_api_exists():
    """Test that upsample3d function exists in ttnn module"""
    assert hasattr(ttnn, "upsample3d"), "ttnn.upsample3d API does not exist"


def test_upsample3d_callable():
    """Test that upsample3d is callable"""
    assert callable(ttnn.upsample3d), "ttnn.upsample3d is not callable"


def test_upsample3d_basic_call_fails_gracefully():
    """Test that we can call upsample3d and it fails with meaningful error"""
    device = ttnn.open_device(device_id=0)
    try:
        input_tensor = torch.ones((1, 2, 2, 2, 4), dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        with pytest.raises((RuntimeError, AttributeError, NotImplementedError)) as exc_info:
            output = ttnn.upsample3d(tt_input, scale_factor=2)

        error_msg = str(exc_info.value).lower()
        assert "upsample3d" in error_msg or "not implemented" in error_msg or "attribute" in error_msg
    finally:
        ttnn.close_device(device)
