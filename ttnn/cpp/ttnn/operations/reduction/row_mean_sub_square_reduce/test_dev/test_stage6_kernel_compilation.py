import pytest
import torch
import ttnn


@pytest.fixture
def device():
    """Setup device for testing. Run 'tt-smi -r 0' before if test hangs."""
    with ttnn.manage_device(device_id=0) as dev:
        yield dev


def test_kernels_compile_at_runtime(device):
    """Kernels should compile without errors when operation runs"""
    # Create input in ROW_MAJOR layout as specified
    input_torch = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run the operation - kernel compilation happens here
    try:
        result = ttnn.row_mean_sub_square_reduce(input_tensor)
    except RuntimeError as e:
        error_str = str(e)
        # Check if this is a kernel compilation error
        if ".cpp" in error_str or "error:" in error_str.lower():
            pytest.fail(f"Kernel compilation failed: {e}")
        # Re-raise if it's a different runtime error
        raise


def test_program_executes_without_hang(device):
    """Program should execute without hanging (stub kernels may produce garbage output)"""
    input_torch = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Should complete without hanging (timeout in pytest.ini should catch hangs)
    result = ttnn.row_mean_sub_square_reduce(input_tensor)

    # Basic sanity checks
    assert result is not None


def test_output_shape_dtype(device):
    """Output tensor should have correct shape and dtype"""
    input_torch = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.row_mean_sub_square_reduce(input_tensor)

    # Shape from spec: [N, C, H, 32] (TILE_WIDTH padded)
    expected_shape = torch.Size([1, 1, 32, 32])
    assert result.shape == expected_shape, f"Expected {expected_shape}, got {result.shape}"

    # Dtype should match input
    assert result.dtype == input_tensor.dtype


def test_multi_tile_execution(device):
    """Should handle multi-tile inputs across multiple cores"""
    # Multiple tile-rows to test work distribution
    input_torch = torch.randn(1, 1, 128, 128, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.row_mean_sub_square_reduce(input_tensor)

    # Should complete and have correct shape
    assert result.shape[0] == input_tensor.shape[0]
    assert result.shape[1] == input_tensor.shape[1]
    assert result.shape[2] == input_tensor.shape[2]
    assert result.shape[3] == 32  # TILE_WIDTH padded
