# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM - Entry Point

Row-major interleaved layer normalization with optional affine transform.

Math:
    mean = row_sum(x) / W
    centered = x - mean
    var = row_sum(centered^2) / W
    output = centered * rsqrt(var + epsilon)
    if gamma: output *= gamma
    if beta:  output += beta

Data flow: RM -> tilize -> compute -> untilize -> RM
"""

import ttnn
from .layer_norm_rm_program_descriptor import create_program_descriptor


def layer_norm_rm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    epsilon: float = 1e-5,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Layer normalization on row-major interleaved tensors.

    Args:
        input_tensor: Input tensor (bfloat16, row-major, interleaved, >= 2D,
                       last two dims aligned to 32)
        gamma: Optional scale parameter, shape (1,1,1,W), bfloat16, row-major
        beta: Optional shift parameter, shape (1,1,1,W), bfloat16, row-major
        epsilon: Variance stabilization constant (default 1e-5)
        memory_config: Output memory configuration (default: DRAM interleaved)

    Returns:
        Normalized tensor, same shape/dtype/layout as input
    """
    _validate_inputs(input_tensor, gamma, beta)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape is identical to input shape
    output_shape = [input_tensor.shape[i] for i in range(len(input_tensor.shape))]

    # Allocate output tensor on device (POSITIONAL args required)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    # Convert gamma/beta to TILE_LAYOUT for the kernel.
    # Gamma/beta are (1,1,1,W) RM tensors. We need them as TILE_LAYOUT so the
    # reader can push tile pages into c_1/c_2 for the compute binary_op helpers.
    # Step 1: Repeat H dimension 32x so each tile row has the gamma/beta values.
    #         (1,1,1,W) -> (1,1,32,W) RM
    # Step 2: Tilize to convert to TILE_LAYOUT.
    #         (1,1,32,W) RM -> (1,1,32,W) TILE with Wt tiles, each 32x32.
    gamma_tiled = None
    beta_tiled = None
    if gamma is not None:
        gamma_repeated = ttnn.repeat(gamma, ttnn.Shape([1, 1, 32, 1]))
        gamma_tiled = ttnn.tilize(gamma_repeated)
    if beta is not None:
        beta_repeated = ttnn.repeat(beta, ttnn.Shape([1, 1, 32, 1]))
        beta_tiled = ttnn.tilize(beta_repeated)

    # Create program descriptor with all CB/kernel configuration
    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        gamma=gamma_tiled,
        beta=beta_tiled,
        epsilon=epsilon,
    )

    # Build io_tensors list: inputs first, output LAST
    io_tensors = [input_tensor]
    if gamma_tiled is not None:
        io_tensors.append(gamma_tiled)
    if beta_tiled is not None:
        io_tensors.append(beta_tiled)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate_inputs(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
) -> None:
    """Validate input tensors meet requirements."""
    # Check rank
    if len(input_tensor.shape) < 2:
        raise ValueError("layer_norm_rm: Input must be at least 2D")

    # Check dtype
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("layer_norm_rm: Input must be bfloat16")

    # Check layout
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("layer_norm_rm: Input must be row-major layout")

    # Check alignment: last two dims must be divisible by 32
    shape = input_tensor.shape
    rank = len(shape)
    H = shape[rank - 2]
    W = shape[rank - 1]

    if H % 32 != 0:
        raise ValueError(f"layer_norm_rm: Height ({H}) must be a multiple of 32")
    if W % 32 != 0:
        raise ValueError(f"layer_norm_rm: Width ({W}) must be a multiple of 32")

    # Validate gamma if provided
    if gamma is not None:
        if gamma.dtype != ttnn.bfloat16:
            raise ValueError("layer_norm_rm: Gamma must be bfloat16")
        if gamma.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("layer_norm_rm: Gamma must be row-major layout")
        gamma_W = gamma.shape[len(gamma.shape) - 1]
        if gamma_W != W:
            raise ValueError(f"layer_norm_rm: Gamma width ({gamma_W}) must match input width ({W})")

    # Validate beta if provided
    if beta is not None:
        if beta.dtype != ttnn.bfloat16:
            raise ValueError("layer_norm_rm: Beta must be bfloat16")
        if beta.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("layer_norm_rm: Beta must be row-major layout")
        beta_W = beta.shape[len(beta.shape) - 1]
        if beta_W != W:
            raise ValueError(f"layer_norm_rm: Beta width ({beta_W}) must match input width ({W})")
