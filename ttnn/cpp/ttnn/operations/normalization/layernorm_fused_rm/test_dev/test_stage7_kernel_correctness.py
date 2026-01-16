"""Stage 7: Kernel Correctness Tests

Owned by: ttnn-kernel-writer agent

SIMPLIFIED TEST: Just verify tilize->untilize passthrough works (identity transform).
This is a debugging step before implementing the full layernorm.
"""
import pytest
import torch
import ttnn


@pytest.fixture
def device():
    """Device fixture with proper management"""
    with ttnn.manage_device(device_id=0) as dev:
        yield dev


def test_identity_passthrough(device):
    """Test that input data passes through tilize->untilize correctly."""
    torch.manual_seed(42)

    # Simple 32x32 input (1 tile)
    shape = (1, 1, 32, 32)
    W = shape[-1]

    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.ones(1, 1, 1, W, dtype=torch.bfloat16)
    beta_torch = torch.zeros(1, 1, 1, W, dtype=torch.bfloat16)
    eps = 1e-5

    # For passthrough, output should equal input
    expected = input_torch.clone()

    # Run on device
    input_tensor = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    gamma_tensor = ttnn.from_torch(gamma_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    beta_tensor = ttnn.from_torch(beta_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    result = ttnn.layernorm_fused_rm(
        input_tensor, gamma_tensor, beta_tensor, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output_torch = ttnn.to_torch(result)

    # Compare - for passthrough, should be exactly equal
    torch.testing.assert_close(output_torch, expected, rtol=0.0, atol=0.0)
