"""Stage 7: Kernel Correctness Tests

Owned by: ttnn-kernel-writer agent

Tests for full layernorm correctness against PyTorch reference.
"""
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout, assert_allclose
from loguru import logger


@pytest.fixture
def device():
    """Device fixture with proper management"""
    with ttnn.manage_device(device_id=0) as dev:
        yield dev


def torch_layernorm(x, gamma, beta, eps=1e-5):
    """PyTorch reference implementation of layernorm (normalized over last dimension)."""
    # x: [..., W]
    # gamma, beta: [W]
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return gamma * normalized + beta


def test_layernorm_basic(device):
    """Test basic layernorm with 32x32 tensor (1 tile)."""
    torch.manual_seed(42)

    shape = (1, 1, 32, 128)
    W = shape[-1]

    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    beta_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    eps = 1e-5

    # PyTorch reference
    expected = torch_layernorm(input_torch, gamma_torch, beta_torch, eps)

    # Run on device
    input_tensor = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    gamma_tensor = ttnn.from_torch(gamma_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    beta_tensor = ttnn.from_torch(beta_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    result = ttnn.layernorm_fused_rm(
        input_tensor, gamma_tensor, beta_tensor, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output_torch = ttnn.to_torch(result)

    # Compare with tolerance for bfloat16
    # bfloat16 limited precision + multiple ops (reduce, rsqrt, mul, add) compound numerical error

    passed, message = assert_with_pcc(output_torch, expected, pcc=0.99)
    logger.info(message)
    assert_allclose(output_torch, expected, rtol=0.2, atol=0.4)


def test_layernorm_with_gamma_beta(device):
    """Test layernorm with non-trivial gamma and beta."""
    torch.manual_seed(123)

    shape = (1, 1, 32, 1024)
    W = shape[-1]

    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    # Non-trivial gamma and beta
    gamma_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16).abs() + 0.5  # Positive values
    beta_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16) * 0.1
    eps = 1e-5

    # PyTorch reference
    expected = torch_layernorm(input_torch, gamma_torch, beta_torch, eps)

    # Run on device
    input_tensor = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    gamma_tensor = ttnn.from_torch(gamma_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    beta_tensor = ttnn.from_torch(beta_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    result = ttnn.layernorm_fused_rm(
        input_tensor, gamma_tensor, beta_tensor, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output_torch = ttnn.to_torch(result)

    # Compare with tolerance for bfloat16
    # bfloat16 has limited precision (7-bit mantissa), and LayerNorm involves
    # multiple ops (reduce, rsqrt, mul, add) which compound numerical error

    passed, message = assert_with_pcc(output_torch, expected, pcc=0.99)
    logger.info(message)

    assert_allclose(output_torch, expected, rtol=0.2, atol=0.4)

    # torch.testing.assert_close(output_torch, expected, rtol=0.15, atol=0.7)


def test_layernorm_multiple_rows(device):
    """Test layernorm with multiple tile rows (64x32)."""
    torch.manual_seed(456)

    shape = (1, 1, 64, 128)  # 2 tile rows
    W = shape[-1]

    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    beta_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    eps = 1e-5

    # PyTorch reference
    expected = torch_layernorm(input_torch, gamma_torch, beta_torch, eps)

    # Run on device
    input_tensor = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    gamma_tensor = ttnn.from_torch(gamma_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    beta_tensor = ttnn.from_torch(beta_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    result = ttnn.layernorm_fused_rm(
        input_tensor, gamma_tensor, beta_tensor, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output_torch = ttnn.to_torch(result)

    # Compare with tolerance for bfloat16
    # bfloat16 limited precision + multiple ops compound numerical error
    # torch.testing.assert_close(output_torch, expected, rtol=0.2, atol=0.4)

    passed, message = assert_with_pcc(output_torch, expected, pcc=0.99)
    assert_allclose(output_torch, expected, rtol=0.2, atol=0.4)
    logger.info(message)
