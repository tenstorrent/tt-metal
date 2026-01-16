import pytest
import torch
import ttnn


@pytest.fixture
def device():
    """Device fixture with proper management"""
    with ttnn.manage_device(device_id=0) as dev:
        yield dev


def test_program_factory_creates_cbs(device):
    """Program factory should create CBs before failing at kernel creation"""
    input_torch = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    gamma_torch = torch.ones(1, 1, 1, 32, dtype=torch.bfloat16)
    beta_torch = torch.zeros(1, 1, 1, 32, dtype=torch.bfloat16)
    gamma_tensor = ttnn.from_torch(gamma_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    beta_tensor = ttnn.from_torch(beta_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with pytest.raises(RuntimeError) as exc:
        result = ttnn.layernorm_fused_rm(
            input_tensor, gamma_tensor, beta_tensor, epsilon=1e-5, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    error_msg = str(exc.value).lower()
    # Should fail at kernel, not at CB or program
    assert "kernel" in error_msg or ".cpp" in error_msg, f"Expected kernel error, got: {exc.value}"
    assert "circular" not in error_msg, f"Should not fail at CB creation: {exc.value}"


def test_work_distribution(device):
    """Should handle various input sizes"""
    # Small input (1 tile row)
    small_input = ttnn.from_torch(
        torch.randn(1, 1, 32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    gamma_small = ttnn.from_torch(
        torch.ones(1, 1, 1, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    beta_small = ttnn.from_torch(
        torch.zeros(1, 1, 1, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    # Large input (many tile rows)
    large_input = ttnn.from_torch(
        torch.randn(1, 1, 64, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    gamma_large = ttnn.from_torch(
        torch.ones(1, 1, 1, 64, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    beta_large = ttnn.from_torch(
        torch.zeros(1, 1, 1, 64, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    for inp, gamma, beta in [(small_input, gamma_small, beta_small), (large_input, gamma_large, beta_large)]:
        with pytest.raises(RuntimeError) as exc:
            ttnn.layernorm_fused_rm(inp, gamma, beta, epsilon=1e-5, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Should reach kernel creation for all sizes
        error_msg = str(exc.value).lower()
        assert "kernel" in error_msg or ".cpp" in error_msg, f"Expected kernel error, got: {exc.value}"
