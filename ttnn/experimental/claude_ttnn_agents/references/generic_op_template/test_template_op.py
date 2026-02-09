# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Template Op - Tests

Run from repo root:
    pytest ttnn/ttnn/operations/<op_name>/test_<op_name>.py -v

Tests are colocated with operation code for experimental convenience.
"""

import pytest
import torch
import ttnn

# Import from same package - no sys.path hacks needed
from .template_op import template_op


def pytorch_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation.

    MODIFY THIS: Implement expected behavior.
    """
    return input_tensor.clone()


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 32), id="single_tile"),
        pytest.param((64, 64), id="2x2_tiles"),
        pytest.param((32, 128), id="1x4_tiles"),
        pytest.param((1, 32, 64), id="batch1"),
        pytest.param((2, 64, 64), id="batch2"),
    ],
)
def test_template_op(device, shape):
    """Test template_op against PyTorch reference."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = template_op(ttnn_input)

    # Verify shape preserved
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {ttnn_output.shape} vs {shape}"

    # Compare with reference
    torch_output = ttnn.to_torch(ttnn_output)
    torch_expected = pytorch_reference(torch_input)

    assert torch.allclose(
        torch_output.float(),
        torch_expected.float(),
        rtol=1e-2,
        atol=1e-2,
    ), f"Output mismatch. Max diff: {(torch_output - torch_expected).abs().max()}"


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 32), id="minimal"),
    ],
)
def test_template_op_shape_preserved(device, shape):
    """Minimal test: verify operation runs and preserves shape."""
    torch_input = torch.ones(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = template_op(ttnn_input)

    assert list(ttnn_output.shape) == list(shape)
