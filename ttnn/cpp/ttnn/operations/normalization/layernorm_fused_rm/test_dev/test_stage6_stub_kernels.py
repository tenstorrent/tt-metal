import pytest
import torch
import ttnn


@pytest.fixture
def device():
    """Device fixture with proper management"""
    with ttnn.manage_device(device_id=0) as dev:
        yield dev


def test_stub_kernels_execute(device):
    """Stub kernels should execute without hanging and produce output tensor"""
    input_torch = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    gamma_torch = torch.ones(1, 1, 1, 32, dtype=torch.bfloat16)
    beta_torch = torch.zeros(1, 1, 1, 32, dtype=torch.bfloat16)
    gamma_tensor = ttnn.from_torch(gamma_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    beta_tensor = ttnn.from_torch(beta_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Should not hang and should return a tensor
    result = ttnn.layernorm_fused_rm(
        input_tensor, gamma_tensor, beta_tensor, epsilon=1e-5, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Verify output is a tensor with correct shape
    assert result is not None, "Operation returned None"
    assert isinstance(result, ttnn.Tensor), f"Expected ttnn.Tensor, got {type(result)}"
    assert result.shape == input_tensor.shape, f"Expected shape {input_tensor.shape}, got {result.shape}"
    assert result.layout == ttnn.ROW_MAJOR_LAYOUT, "Output should be ROW_MAJOR"

    # Note: Values will be garbage at this stage - this is expected for stub kernels


def test_stub_kernels_multiple_sizes(device):
    """Stub kernels should handle different tensor sizes without hanging"""
    test_shapes = [
        (1, 1, 32, 32),  # Single tile row
        (1, 1, 64, 64),  # 2x2 tile rows
    ]

    for shape in test_shapes:
        input_torch = torch.randn(shape, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        W = shape[-1]
        gamma_torch = torch.ones(1, 1, 1, W, dtype=torch.bfloat16)
        beta_torch = torch.zeros(1, 1, 1, W, dtype=torch.bfloat16)
        gamma_tensor = ttnn.from_torch(gamma_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        beta_tensor = ttnn.from_torch(beta_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        result = ttnn.layernorm_fused_rm(
            input_tensor, gamma_tensor, beta_tensor, epsilon=1e-5, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        assert result is not None, f"Operation returned None for shape {shape}"
        assert result.shape == input_tensor.shape, f"Shape mismatch for {shape}"
