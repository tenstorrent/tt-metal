import pytest
import torch
import ttnn


@pytest.fixture
def device():
    """Device fixture with proper management"""
    with ttnn.manage_device(device_id=0) as dev:
        yield dev


def test_device_op_called(device):
    """Operation should reach program factory, not fail at validation"""
    # Create ROW_MAJOR input (spec requires ROW_MAJOR)
    input_torch = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Create gamma and beta (1D broadcast vectors)
    gamma_torch = torch.ones(1, 1, 1, 32, dtype=torch.bfloat16)
    beta_torch = torch.zeros(1, 1, 1, 32, dtype=torch.bfloat16)
    gamma_tensor = ttnn.from_torch(gamma_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    beta_tensor = ttnn.from_torch(beta_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with pytest.raises(RuntimeError) as exc:
        result = ttnn.layernorm_fused_rm(
            input_tensor, gamma_tensor, beta_tensor, epsilon=1e-5, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    # Error should be about program/kernel, not validation
    error_msg = str(exc.value).lower()
    assert (
        "kernel" in error_msg or "program" in error_msg or "factory" in error_msg or "not yet implemented" in error_msg
    ), f"Expected program/kernel error, got: {exc.value}"


def test_program_factory_selected(device):
    """select_program_factory should return valid factory type"""
    input_torch = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    gamma_torch = torch.ones(1, 1, 1, 32, dtype=torch.bfloat16)
    beta_torch = torch.zeros(1, 1, 1, 32, dtype=torch.bfloat16)
    gamma_tensor = ttnn.from_torch(gamma_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    beta_tensor = ttnn.from_torch(beta_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Operation should not fail at factory selection
    with pytest.raises(RuntimeError) as exc:
        result = ttnn.layernorm_fused_rm(
            input_tensor, gamma_tensor, beta_tensor, epsilon=1e-5, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    # Should not mention "select" or "factory selection" - should reach implementation
    error_msg = str(exc.value).lower()
    assert "select" not in error_msg, f"Should not fail at factory selection: {exc.value}"
