# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Simple test for reduce_avg_w_rm operation."""

import torch
import ttnn


def test_basic():
    """Test basic reduce_avg_w_rm functionality."""
    device = ttnn.open_device(device_id=0)

    # Create a simple test tensor [1, 1, 32, 64] - one batch, one channel, 32 height, 64 width
    torch_input = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)

    # Expected output: average along width dimension
    # PyTorch reference: mean along last dimension, keepdim=True, then pad to width=32
    torch_expected = torch.mean(torch_input.float(), dim=-1, keepdim=True).bfloat16()

    # Create row-major tensor on device
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print(f"Input shape: {ttnn_input.shape}")
    print(f"Input layout: {ttnn_input.layout}")

    # Run the operation
    ttnn_output = ttnn.reduce_avg_w_rm(ttnn_input)

    print(f"Output shape: {ttnn_output.shape}")
    print(f"Output layout: {ttnn_output.layout}")

    # Convert output back to torch
    torch_output = ttnn.to_torch(ttnn_output)

    # Compare only the first column (the actual reduced values)
    # Output is [1, 1, 32, 32] but only first column matters
    torch_actual = torch_output[:, :, :, 0:1]

    print(f"\nExpected shape: {torch_expected.shape}")
    print(f"Actual shape: {torch_actual.shape}")

    # Check numerical correctness
    max_diff = torch.max(torch.abs(torch_expected.float() - torch_actual.float())).item()
    print(f"Max absolute difference: {max_diff}")

    # Allow some tolerance for bfloat16 precision
    assert max_diff < 0.1, f"Max difference {max_diff} exceeds tolerance 0.1"

    print("\nTest PASSED!")

    ttnn.close_device(device)


if __name__ == "__main__":
    test_basic()
