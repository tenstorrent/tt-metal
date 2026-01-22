import pytest
import torch
import ttnn

# NOTE: Use the built-in `device` fixture from conftest.py - do NOT define your own.
# Before running: 'tt-smi -ls' to verify device, 'tt-smi -r 0' to reset (see CLAUDE.md)


def test_program_factory_creates_cbs(device):
    """Program factory should create CBs before failing at kernel creation"""
    input_tensor = ttnn.from_torch(
        torch.randn(1, 1, 32, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    with pytest.raises(RuntimeError) as exc:
        ttnn.standardize_w_rm(input_tensor, epsilon=1e-5)

    error_msg = str(exc.value).lower()
    # Should fail at kernel, not at CB or program
    assert "kernel" in error_msg, f"Expected kernel error, got: {exc.value}"
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

    # Large input (many tile rows)
    large_input = ttnn.from_torch(
        torch.randn(1, 32, 64, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    for inp in [small_input, large_input]:
        with pytest.raises(RuntimeError) as exc:
            ttnn.standardize_w_rm(inp, epsilon=1e-5)
        # Should reach kernel creation for all sizes
        assert "kernel" in str(exc.value).lower()
