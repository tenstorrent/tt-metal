import pytest
import torch
import ttnn


@pytest.fixture
def device():
    """Setup device for testing. Run 'tt-smi -r 0' before if test hangs."""
    with ttnn.manage_device(device_id=0) as dev:
        yield dev


def test_device_op_called(device):
    """Operation should reach program factory, not fail at validation"""
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

    # Error should be about program/kernel, not validation
    error_msg = str(exc.value).lower()
    assert (
        "kernel" in error_msg or "program" in error_msg or "factory" in error_msg or "not yet implemented" in error_msg
    ), f"Expected program/kernel error, got: {exc.value}"


def test_program_factory_selected(device):
    """select_program_factory should return valid factory type"""
    input_torch = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Operation should not fail at factory selection
    with pytest.raises(RuntimeError) as exc:
        ttnn.row_mean_sub_square_reduce(input_tensor)

    # Should not mention "select" or "factory selection"
    error_msg = str(exc.value).lower()
    assert "select" not in error_msg, f"Error at factory selection: {exc.value}"
