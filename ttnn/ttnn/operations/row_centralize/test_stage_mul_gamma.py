# TDD Stage: mul_gamma
# Tests gamma scaling: y = gamma * (x - mean) * rsqrt(var + eps)
"""
TDD Stage: mul_gamma
Reader reads gamma, compute tilizes gamma at startup + Phase 8.5 (mul ROW).
Beta is all-ones so its effect is neutral (multiply by 1 after add = just add 0... no, beta isn't applied yet).
Actually, for this stage we only apply gamma (no beta), so we pass gamma AND beta=zeros to the op,
but we test against: gamma * standardize(x).
Wait - the plan says Stage 1 tests mul_gamma only. But the implementation has both gamma AND beta.
We need both gamma and beta provided (they're coupled). For Stage 1, use beta=0 to neutralize it.
Actually re-reading the plan: Stage 1 has compute doing tilize gamma + Phase 8.5 only.
But our implementation does both 8.5 and 8.6 together (if has_affine).
So for Stage 1 test: pass beta=0 (zeros), verify gamma * standardize(x).
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
def test_mul_gamma(device, shape):
    """Verify gamma * (x - mean) * rsqrt(var + eps) with beta=0."""
    import torch

    torch.manual_seed(42)
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.empty(1, W, dtype=torch.bfloat16).uniform_(0.5, 1.5)
    torch_beta = torch.zeros(1, W, dtype=torch.bfloat16)

    # Reference: gamma * standardize(x) + 0
    mean = torch_input.mean(dim=-1, keepdim=True)
    centered = torch_input - mean
    var = (centered**2).mean(dim=-1, keepdim=True)
    standardized = centered * (var + 1e-5).rsqrt()
    expected = standardized * torch_gamma  # beta=0, so no addition

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_beta = ttnn.from_torch(
        torch_beta,
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
