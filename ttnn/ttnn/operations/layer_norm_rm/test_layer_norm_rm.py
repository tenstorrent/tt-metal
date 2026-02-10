# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest


def compute_pcc(tensor1, tensor2):
    """Compute Pearson Correlation Coefficient between two tensors."""
    import torch

    tensor1_flat = tensor1.flatten().float()
    tensor2_flat = tensor2.flatten().float()
    mean1 = tensor1_flat.mean()
    mean2 = tensor2_flat.mean()
    centered1 = tensor1_flat - mean1
    centered2 = tensor2_flat - mean2
    numerator = (centered1 * centered2).sum()
    denominator = torch.sqrt((centered1**2).sum() * (centered2**2).sum())
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return (numerator / denominator).item()


# Test shapes as specified
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 1, 128, 128],
        [1, 1, 32, 1024],
        [1, 1, 1024, 32],
        [1, 1, 4096, 32],
        [1, 1, 512, 512],
        [2, 3, 64, 128],
        [1, 64, 128],
    ],
)
@pytest.mark.parametrize("dtype_str", ["bfloat16", "float32"])
@pytest.mark.parametrize("has_gamma", [True, False])
@pytest.mark.parametrize("has_beta", [True, False])
@pytest.mark.parametrize("epsilon", [1e-5])
def test_layer_norm_rm(shape, dtype_str, has_gamma, has_beta, epsilon, device):
    """Test layer_norm_rm operation with various configurations."""
    import torch
    import ttnn
    from ttnn.operations.layer_norm_rm import layer_norm_rm

    torch.manual_seed(42)

    # Map dtype string to ttnn dtype
    if dtype_str == "bfloat16":
        ttnn_dtype = ttnn.bfloat16
        torch_dtype = torch.bfloat16
    else:
        ttnn_dtype = ttnn.float32
        torch_dtype = torch.float32

    # Get W dimension (last dimension)
    W = shape[-1]

    # Skip shapes that exceed L1 capacity for float32 (single-core)
    # Each CB with Wt tiles at 4096 bytes/tile; many CBs required for layer norm
    if dtype_str == "float32":
        element_size = 4
        Wt = W // 32
        # Rough L1 budget estimate: ~14 CBs of Wt tiles + small CBs
        estimated_l1 = 14 * Wt * (32 * 32 * element_size) + 32768
        if estimated_l1 > 1500000:
            pytest.skip(f"L1 overflow: shape {shape} with float32 needs ~{estimated_l1} bytes")

    # Create input tensor
    torch_input = torch.randn(shape, dtype=torch_dtype)

    # Create gamma and beta if needed
    gamma_shape = [1] * (len(shape) - 1) + [W]
    beta_shape = [1] * (len(shape) - 1) + [W]

    if has_gamma:
        torch_gamma = torch.randn(gamma_shape, dtype=torch_dtype)
    else:
        torch_gamma = None

    if has_beta:
        torch_beta = torch.randn(beta_shape, dtype=torch_dtype)
    else:
        torch_beta = None

    # Convert to ttnn tensors
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if has_gamma:
        ttnn_gamma = ttnn.from_torch(
            torch_gamma,
            dtype=ttnn_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        ttnn_gamma = None

    if has_beta:
        ttnn_beta = ttnn.from_torch(
            torch_beta,
            dtype=ttnn_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        ttnn_beta = None

    # Run TTNN operation
    ttnn_output = layer_norm_rm(
        ttnn_input,
        gamma=ttnn_gamma,
        beta=ttnn_beta,
        epsilon=epsilon,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Verify output shape
    assert (
        ttnn_output.shape == ttnn_input.shape
    ), f"Output shape {ttnn_output.shape} does not match input shape {ttnn_input.shape}"

    # Verify output layout
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT, "Output layout should be ROW_MAJOR"

    # Verify output dtype
    assert ttnn_output.dtype == ttnn_dtype, f"Output dtype {ttnn_output.dtype} does not match input dtype {ttnn_dtype}"

    # Compute PyTorch reference
    normalized_shape = [W]
    torch_output = torch.nn.functional.layer_norm(
        torch_input.float(),
        normalized_shape,
        weight=torch_gamma.flatten().float() if has_gamma else None,
        bias=torch_beta.flatten().float() if has_beta else None,
        eps=epsilon,
    )

    # Convert TTNN output to torch
    ttnn_output_torch = ttnn.to_torch(ttnn_output).float()

    # Compute PCC
    pcc = compute_pcc(torch_output, ttnn_output_torch)

    # Verify PCC threshold
    if dtype_str == "bfloat16":
        min_pcc = 0.99
    else:  # float32
        min_pcc = 0.999

    assert pcc > min_pcc, f"PCC {pcc:.6f} is below threshold {min_pcc} for dtype {dtype_str}, shape {shape}"

    print(
        f"PASSED: shape={shape}, dtype={dtype_str}, "
        f"has_gamma={has_gamma}, has_beta={has_beta}, epsilon={epsilon}, PCC={pcc:.6f}"
    )


@pytest.mark.parametrize("shape", [[1, 1, 32, 32]])
@pytest.mark.parametrize("dtype_str", ["bfloat16"])
def test_layer_norm_rm_validation(shape, dtype_str, device):
    """Test input validation errors."""
    import torch
    import ttnn
    from ttnn.operations.layer_norm_rm import layer_norm_rm

    # Map dtype string to ttnn dtype
    ttnn_dtype = ttnn.bfloat16 if dtype_str == "bfloat16" else ttnn.float32
    torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32

    W = shape[-1]

    # Test: Gamma dtype mismatch
    torch_input = torch.randn(shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create gamma with mismatched dtype
    wrong_dtype = ttnn.float32 if ttnn_dtype == ttnn.bfloat16 else ttnn.bfloat16
    wrong_torch_dtype = torch.float32 if torch_dtype == torch.bfloat16 else torch.bfloat16

    gamma_shape = [1] * (len(shape) - 1) + [W]
    torch_gamma = torch.randn(gamma_shape, dtype=wrong_torch_dtype)
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=wrong_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # This should raise an error
    with pytest.raises(RuntimeError, match="gamma dtype must match"):
        layer_norm_rm(ttnn_input, gamma=ttnn_gamma)

    print("Validation tests passed")
