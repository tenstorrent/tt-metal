# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Row Centralize - Main Entry Point

Row-wise standardization (LayerNorm):
    mu      = mean(x, dim=-1, keepdim=True)
    c       = x - mu
    var     = mean(c^2, dim=-1, keepdim=True)
    s       = rsqrt(var + epsilon)
    y       = gamma * c * s + beta       (when gamma/beta provided)
    y       = c * s                      (when gamma/beta are None)

Input:  RM interleaved bfloat16, at least 2D, dims divisible by 32
Gamma:  RM interleaved bfloat16, shape (1,...,1,W) — optional
Beta:   RM interleaved bfloat16, shape (1,...,1,W) — optional
Output: RM interleaved bfloat16, same shape as input

Usage:
    from ttnn.operations.row_centralize import row_centralize
    output = row_centralize(input_tensor, epsilon=1e-5)
    output = row_centralize(input_tensor, gamma=gamma, beta=beta, epsilon=1e-5)
"""

import ttnn
from .row_centralize_program_descriptor import create_program_descriptor

TILE_SIZE = 32


def row_centralize(
    input_tensor: ttnn.Tensor,
    *,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    epsilon: float = 1e-5,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Row-wise standardization of a tensor (LayerNorm).

    For each row r:
        mu_r    = mean(x[r, :])
        c_r     = x[r, :] - mu_r
        var_r   = mean(c_r^2)
        s_r     = rsqrt(var_r + epsilon)
        y[r, :] = gamma * c_r * s_r + beta   (if gamma/beta provided)
        y[r, :] = c_r * s_r                  (if gamma/beta are None)

    Args:
        input_tensor: Input tensor. Must be:
            - At least 2D
            - bfloat16 dtype
            - ROW_MAJOR layout
            - INTERLEAVED memory layout
            - On device
            - Last dimension divisible by 32
            - Second-to-last dimension divisible by 32
        gamma: Optional scale tensor. Must be RM, bfloat16, interleaved, on device,
               last dim == input's W. Shape (1,...,1,W).
        beta: Optional bias tensor. Same requirements as gamma.
              gamma and beta must both be provided or both be None.
        epsilon: Small constant for numerical stability (default 1e-5)
        memory_config: Memory configuration for output (default: DRAM interleaved)

    Returns:
        Output tensor with same shape, dtype=bfloat16, layout=ROW_MAJOR, INTERLEAVED.
    """
    _validate_input(input_tensor)

    if (gamma is None) != (beta is None):
        raise ValueError("row_centralize: gamma and beta must both be provided or both be None")

    has_affine = gamma is not None
    if has_affine:
        _validate_affine(gamma, input_tensor, "gamma")
        _validate_affine(beta, input_tensor, "beta")

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape is identical to input shape
    shape = input_tensor.shape
    output_shape = [shape[i] for i in range(len(shape))]

    # Allocate output tensor on device (POSITIONAL args required)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, epsilon, gamma=gamma, beta=beta)

    # Output tensor MUST be last in the list
    # When affine: [input, gamma, beta, output]
    # When not:    [input, output]
    if has_affine:
        return ttnn.generic_op([input_tensor, gamma, beta, output_tensor], program_descriptor)
    else:
        return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor) -> None:
    """Validate input tensor meets all requirements from the spec."""
    shape = input_tensor.shape
    rank = len(shape)

    if rank < 2:
        raise ValueError(f"row_centralize: Input tensor must have rank >= 2, got rank {rank}")

    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"row_centralize: Input tensor must be in ROW_MAJOR layout, got {input_tensor.layout}")

    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"row_centralize: Input tensor dtype must be bfloat16, got {input_tensor.dtype}")

    if not input_tensor.is_allocated():
        raise ValueError("row_centralize: Input tensor must be on device (not allocated)")

    memory_config = input_tensor.memory_config()
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("row_centralize: Input tensor must have interleaved memory layout")

    last_dim = shape[rank - 1]
    if last_dim % TILE_SIZE != 0:
        raise ValueError(f"row_centralize: Input tensor last dimension must be divisible by 32, got {last_dim}")

    second_to_last_dim = shape[rank - 2]
    if second_to_last_dim % TILE_SIZE != 0:
        raise ValueError(
            f"row_centralize: Input tensor second-to-last dimension must be divisible by 32, "
            f"got {second_to_last_dim}"
        )


def _validate_affine(tensor: ttnn.Tensor, input_tensor: ttnn.Tensor, name: str) -> None:
    """Validate gamma or beta tensor meets requirements."""
    if tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"row_centralize: {name} must be in ROW_MAJOR layout, got {tensor.layout}")

    if tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"row_centralize: {name} dtype must be bfloat16, got {tensor.dtype}")

    if not tensor.is_allocated():
        raise ValueError(f"row_centralize: {name} must be on device (not allocated)")

    mem_cfg = tensor.memory_config()
    if mem_cfg.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"row_centralize: {name} must have interleaved memory layout")

    input_W = input_tensor.shape[len(input_tensor.shape) - 1]
    tensor_W = tensor.shape[len(tensor.shape) - 1]
    if tensor_W != input_W:
        raise ValueError(f"row_centralize: {name} last dim must match input's last dim ({input_W}), got {tensor_W}")
