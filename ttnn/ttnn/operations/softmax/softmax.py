# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Softmax operation entry point using ttnn.generic_op."""

import ttnn

from ttnn.operations.softmax.softmax_program_descriptor import create_softmax_program_descriptor


def softmax(
    input_tensor: ttnn.Tensor,
    dim: int = -1,
    *,
    numeric_stable: bool = True,
    memory_config=None,
) -> ttnn.Tensor:
    """
    Compute softmax along the specified dimension.

    softmax(x, dim)[i] = exp(x[i] - max(x, dim)) / sum(exp(x - max(x, dim)), dim)

    Args:
        input_tensor: Input tensor (bfloat16, TILE_LAYOUT, rank >= 2).
        dim: Dimension along which softmax is computed. Must be -1 or -2.
        numeric_stable: If True, subtract max before exp for numerical stability.
        memory_config: Output memory configuration. Defaults to DRAM_MEMORY_CONFIG.

    Returns:
        Output tensor with softmax applied, same shape/dtype/layout as input.

    Raises:
        ValueError: If input dtype is not bfloat16, layout is not TILE_LAYOUT,
                     rank < 2, or dim is not in {-1, -2}.
    """
    # --- Validation ---
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"Input must be bfloat16, got {input_tensor.dtype}")

    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"Input must be TILE_LAYOUT, got {input_tensor.layout}")

    rank = len(input_tensor.shape)
    if rank < 2:
        raise ValueError(f"Input rank must be >= 2, got {rank}")

    if dim not in (-1, -2):
        raise ValueError(f"dim must be -1 or -2, got {dim}")

    # --- Output allocation ---
    output_shape = list(input_tensor.shape)
    device = input_tensor.device()
    mem_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        mem_config,
    )

    # --- Build program descriptor ---
    program_descriptor = create_softmax_program_descriptor(
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        dim=dim,
        numeric_stable=numeric_stable,
    )

    # --- Execute (output tensor MUST be last in list) ---
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
