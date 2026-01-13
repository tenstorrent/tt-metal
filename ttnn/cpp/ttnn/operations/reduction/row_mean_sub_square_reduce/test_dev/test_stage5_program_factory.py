import pytest
import torch
import ttnn


@pytest.fixture
def device():
    """Setup device for testing. Run 'tt-smi -r 0' before if test hangs."""
    with ttnn.manage_device(device_id=0) as dev:
        yield dev


def test_program_factory_creates_cbs(device):
    """Program factory should create CBs before failing at kernel creation"""
    # Create input in ROW_MAJOR layout as specified
    input_torch = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(RuntimeError) as exc:
        ttnn.row_mean_sub_square_reduce(input_tensor)

    error_msg = str(exc.value).lower()
    # Should fail at kernel, not at CB or program
    assert "kernel" in error_msg, f"Expected kernel error, got: {exc.value}"
    assert "circular" not in error_msg, f"Should not fail at CB creation: {exc.value}"


def test_work_distribution(device):
    """Should handle various input sizes"""
    # Small input (32x64)
    small_input_torch = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    small_input = ttnn.from_torch(
        small_input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Large input (many tile-rows)
    large_input_torch = torch.randn(1, 1, 128, 128, dtype=torch.bfloat16)
    large_input = ttnn.from_torch(
        large_input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for inp in [small_input, large_input]:
        with pytest.raises(RuntimeError) as exc:
            ttnn.row_mean_sub_square_reduce(inp)
        # Should reach kernel creation for all sizes
        error_msg = str(exc.value).lower()
        assert "kernel" in error_msg, f"Expected kernel error for input {inp.shape}, got: {exc.value}"
