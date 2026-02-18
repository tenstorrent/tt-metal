# TDD Stage: add_beta
# Tests full LayerNorm: y = gamma * (x - mean) * rsqrt(var + eps) + beta
"""
TDD Stage: add_beta
Full affine transform with both gamma and beta.
Reference: torch.nn.functional.layer_norm(x, [W], weight=gamma, bias=beta, eps=1e-5)
"""

import pytest
import ttnn

from .row_centralize import row_centralize


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="1x1x32x32"),
        pytest.param((1, 1, 32, 64), id="1x1x32x64"),
        pytest.param((1, 1, 64, 128), id="1x1x64x128"),
        pytest.param((1, 1, 128, 256), id="1x1x128x256"),
        pytest.param((2, 1, 32, 64), id="2x1x32x64"),
    ],
)
def test_add_beta(device, shape):
    """Verify full LayerNorm: gamma * standardize(x) + beta."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(42)
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.empty(W, dtype=torch.bfloat16).uniform_(0.5, 1.5)
    torch_beta = torch.empty(W, dtype=torch.bfloat16).uniform_(-0.5, 0.5)

    # Reference using torch.nn.functional.layer_norm
    expected = F.layer_norm(
        torch_input.float(),
        [W],
        weight=torch_gamma.float(),
        bias=torch_beta.float(),
        eps=1e-5,
    ).to(torch.bfloat16)

    # For TTNN: gamma/beta must be shape (1, W) for RM layout
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_beta = ttnn.from_torch(
        torch_beta.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = row_centralize(ttnn_input, gamma=ttnn_gamma, beta=ttnn_beta)

    assert list(ttnn_output.shape) == list(shape), f"Shape: {list(ttnn_output.shape)} vs {list(shape)}"

    torch_output = ttnn.to_torch(ttnn_output)
    assert torch.allclose(
        torch_output.float(),
        expected.float(),
        rtol=0.05,
        atol=0.2,
    ), f"Max diff: {(torch_output.float() - expected.float()).abs().max()}"
