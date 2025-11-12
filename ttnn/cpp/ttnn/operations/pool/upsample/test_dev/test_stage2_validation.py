# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Stage 2: Parameter Validation Test
Goal: Verify input parameters are properly validated
"""

import pytest
import torch
import ttnn


@pytest.fixture
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


def test_input_rank_validation(device):
    """Test that input must be 5D"""
    # 4D tensor should fail
    input_4d = torch.ones((1, 2, 2, 4), dtype=torch.bfloat16)
    tt_input_4d = ttnn.from_torch(input_4d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.upsample3d(tt_input_4d, scale_factor=2)
    assert "5D" in str(exc_info.value) or "rank" in str(exc_info.value).lower()


def test_scale_factor_validation(device):
    """Test scale_factor parameter validation"""
    input_5d = torch.ones((1, 2, 2, 2, 4), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_5d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Test with negative scale factor
    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=-1)
    assert "positive" in str(exc_info.value).lower() or "scale" in str(exc_info.value).lower()

    # Test with zero scale factor
    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=0)
    assert "positive" in str(exc_info.value).lower() or "scale" in str(exc_info.value).lower()


def test_scale_factor_types(device):
    """Test that scale_factor accepts int or tuple of 3 ints"""
    input_5d = torch.ones((1, 2, 2, 2, 4), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input_5d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # These should pass validation (may fail later)
    try:
        ttnn.upsample3d(tt_input, scale_factor=2)  # int
    except RuntimeError as e:
        assert "not implemented" in str(e).lower() or "device" in str(e).lower()

    try:
        ttnn.upsample3d(tt_input, scale_factor=(2, 2, 2))  # tuple of 3
    except RuntimeError as e:
        assert "not implemented" in str(e).lower() or "device" in str(e).lower()

    # Wrong tuple size should fail validation
    with pytest.raises((RuntimeError, ValueError, TypeError)) as exc_info:
        ttnn.upsample3d(tt_input, scale_factor=(2, 2))  # tuple of 2
    assert "3" in str(exc_info.value) or "scale" in str(exc_info.value).lower()


def test_layout_validation(device):
    """Test that only ROW_MAJOR layout is supported initially"""
    input_5d = torch.ones((1, 2, 32, 32, 32), dtype=torch.bfloat16)  # Tile-aligned
    tt_input_tiled = ttnn.from_torch(input_5d, device=device, layout=ttnn.TILE_LAYOUT)

    with pytest.raises(RuntimeError) as exc_info:
        ttnn.upsample3d(tt_input_tiled, scale_factor=2)
    assert "ROW_MAJOR" in str(exc_info.value) or "layout" in str(exc_info.value).lower()
