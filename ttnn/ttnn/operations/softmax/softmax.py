# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn

from .softmax_program_descriptor import create_program_descriptor


def softmax(
    input_tensor: ttnn.Tensor,
    dim: int = -1,
    *,
    numeric_stable: bool = True,
    memory_config=None,
) -> ttnn.Tensor:
    """
    Compute softmax along the specified dimension.

    softmax(x, dim)_i = exp(x_i - max(x, dim)) / sum(exp(x_j - max(x, dim)), dim)

    When numeric_stable=False, the max subtraction is skipped.

    Args:
        input_tensor: Input tensor (bfloat16, TILE_LAYOUT, 4D with H,W divisible by 32)
        dim: Reduction dimension (-1 for W, -2 for H)
        numeric_stable: If True, subtract max before exp for numerical stability
        memory_config: Output memory configuration (defaults to DRAM_MEMORY_CONFIG)

    Returns:
        Output tensor with same shape as input
    """
    # Validate dtype
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("Input must be bfloat16")

    # Validate layout
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("Input must be TILE_LAYOUT")

    # Get shape and validate rank
    shape = input_tensor.shape
    rank = len(shape)
    if rank < 2:
        raise ValueError("Input must have rank >= 2")

    # Pad to 4D if needed
    if rank == 2:
        padded_shape = [1, 1, shape[0], shape[1]]
    elif rank == 3:
        padded_shape = [1, shape[0], shape[1], shape[2]]
    elif rank == 4:
        padded_shape = [shape[i] for i in range(4)]
    else:
        raise ValueError("Input must be 2D, 3D, or 4D")

    N, C, H, W = padded_shape

    # Validate tile alignment
    if H % 32 != 0:
        raise ValueError(f"H must be divisible by 32, got {H}")
    if W % 32 != 0:
        raise ValueError(f"W must be divisible by 32, got {W}")

    # Validate dim
    if dim not in (-1, -2):
        raise ValueError(f"Only dim=-1 (W) and dim=-2 (H) supported, got dim={dim}")

    # Output has same shape as input
    output_shape = [shape[i] for i in range(rank)]

    # Determine memory config
    mem_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

    # Allocate output tensor on device (POSITIONAL args)
    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        mem_config,
    )

    # Create program descriptor
    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        dim=dim,
        numeric_stable=numeric_stable,
    )

    # Execute - output tensor MUST be last in the list
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
