import pytest
import torch
import ttnn

# NOTE: Use the built-in `device` fixture from conftest.py - do NOT define your own.


def test_kernels_compile_at_runtime(device):
    """Kernels should compile without errors when operation runs"""
    input_tensor = ttnn.from_torch(
        torch.randn(1, 1, 32, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # Run the operation - kernel compilation happens here
    # Stub kernels will produce garbage output, but should not error
    try:
        result = ttnn.standardize_w_rm(input_tensor, epsilon=1e-5)
    except RuntimeError as e:
        error_str = str(e)
        if ".cpp" in error_str or "error:" in error_str.lower():
            pytest.fail(f"Kernel compilation failed: {e}")
        raise


def test_program_executes_without_hang(device):
    """Program should execute without hanging (CB sync must be balanced)"""
    input_tensor = ttnn.from_torch(
        torch.randn(1, 1, 32, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # Stub kernels must not hang - this verifies CB sync is correct
    result = ttnn.standardize_w_rm(input_tensor, epsilon=1e-5)
    assert result is not None
    assert result.shape == input_tensor.shape  # Output shape must match input


def test_output_shape_correct(device):
    """Output tensor should have correct shape (even if values are garbage)"""
    # Test with different sizes
    for H, W in [(32, 64), (64, 32), (32, 128)]:
        input_tensor = ttnn.from_torch(
            torch.randn(1, 1, H, W, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        result = ttnn.standardize_w_rm(input_tensor, epsilon=1e-5)
        result_torch = ttnn.to_torch(result)

        assert result.shape == input_tensor.shape, f"Shape mismatch for {H}x{W}"
        assert result_torch.shape == (1, 1, H, W), f"Torch shape mismatch for {H}x{W}"
