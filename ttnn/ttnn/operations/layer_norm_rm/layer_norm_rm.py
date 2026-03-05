# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Layer normalization over the last dimension of a row-major interleaved tensor.

Mathematical definition:
    mean[n,c,h] = sum(x[n,c,h,:]) / W
    var[n,c,h]  = sum((x[n,c,h,:] - mean[n,c,h])^2) / W
    output[n,c,h,w] = gamma[w] * (x[n,c,h,w] - mean[n,c,h]) / sqrt(var[n,c,h] + eps) + beta[w]

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    output = layer_norm_rm(input_tensor, gamma=gamma_tensor, beta=beta_tensor)
"""

import ttnn

from .layer_norm_rm_program_descriptor import create_program_descriptor


def layer_norm_rm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    epsilon: float = 1e-6,
) -> ttnn.Tensor:
    """
    Layer normalization over the last dimension of a row-major interleaved tensor.

    Args:
        input_tensor: Input tensor, must be bfloat16, ROW_MAJOR, DRAM interleaved.
                      Shape must be [N, C, H, W] where H and W are multiples of 32.
        gamma:        Scale tensor of shape [1, 1, 1, W], bfloat16, ROW_MAJOR, DRAM.
                      If None, uses ones.
        beta:         Bias tensor of shape [1, 1, 1, W], bfloat16, ROW_MAJOR, DRAM.
                      If None, uses zeros.
        epsilon:      Variance stability constant. Default: 1e-6.

    Returns:
        Output tensor with same shape as input, bfloat16, ROW_MAJOR, DRAM interleaved.
    """
    _validate_input(input_tensor)

    device = input_tensor.device()
    shape = input_tensor.shape
    W = shape[-1]

    # Create default gamma (ones) and beta (zeros) if not provided
    gamma_tensor, beta_tensor = _prepare_gamma_beta(gamma, beta, W, device)

    # Allocate output tensor - same shape/dtype/layout as input
    output_shape = [shape[i] for i in range(len(shape))]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    program_descriptor = create_program_descriptor(input_tensor, gamma_tensor, beta_tensor, output_tensor, epsilon)

    # Output tensor MUST be last in the list
    return ttnn.generic_op([input_tensor, gamma_tensor, beta_tensor, output_tensor], program_descriptor)


def _prepare_gamma_beta(gamma, beta, W, device):
    """Prepare gamma and beta tensors, creating defaults if None."""
    import torch  # Local import to avoid global torch import in ttnn package

    if gamma is None:
        gamma_torch = torch.ones(1, 1, 1, W, dtype=torch.bfloat16)
        gamma = ttnn.from_torch(
            gamma_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    if beta is None:
        beta_torch = torch.zeros(1, 1, 1, W, dtype=torch.bfloat16)
        beta = ttnn.from_torch(
            beta_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return gamma, beta


def _validate_input(input_tensor: ttnn.Tensor) -> None:
    """Validate input tensor meets requirements."""
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("layer_norm_rm: input must be bfloat16")

    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("layer_norm_rm: input must be row-major layout")

    if len(input_tensor.shape) < 2:
        raise ValueError("layer_norm_rm: input must be at least 2D")

    shape = input_tensor.shape
    H = shape[-2]
    W = shape[-1]

    if H % 32 != 0:
        raise ValueError(f"layer_norm_rm: H={H} must be a multiple of 32")

    if W % 32 != 0:
        raise ValueError(f"layer_norm_rm: W={W} must be a multiple of 32")
