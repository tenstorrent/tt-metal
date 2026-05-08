# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
backward_softmax — VJP of softmax.

    grad_input = output * (grad_output - sum(output * grad_output, dim))

Phase-0 constraints:
- Inputs: float32, TILE_LAYOUT, rank-4, H/W tile-aligned (multiple of 32),
  identical shape and dtype between the two inputs.
- Reduce dimension: dim ∈ {-1, -2}.
"""

import ttnn
from .backward_softmax_program_descriptor import create_program_descriptor


def backward_softmax(
    grad_output: ttnn.Tensor,
    output: ttnn.Tensor,
    *,
    dim: int = -1,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Vector-Jacobian product (VJP) of softmax.

    Computes::

        grad_input = output * (grad_output - sum(output * grad_output, dim))

    Args:
        grad_output: Upstream gradient (dy). float32, TILE_LAYOUT, rank-4,
            H/W tile-aligned, on-device.
        output: Forward softmax output (y). Identical shape & dtype as
            grad_output.
        dim: Reduction dimension. Must be -1 (W) or -2 (H). Defaults to -1.
        memory_config: Output memory config (default: DRAM interleaved).

    Returns:
        grad_input: same shape and dtype as grad_output.
    """
    _validate(grad_output, output, dim)

    device = grad_output.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_shape = list(grad_output.shape)

    # CRITICAL: positional args, not keyword args.
    grad_input = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        grad_output.dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(grad_output, output, grad_input, dim=dim)

    # Output tensor MUST be last in the io list.
    return ttnn.generic_op([grad_output, output, grad_input], program_descriptor)


def _validate(grad_output: ttnn.Tensor, output: ttnn.Tensor, dim: int) -> None:
    # Dtype.
    if grad_output.dtype != ttnn.float32:
        raise ValueError(f"backward_softmax: grad_output dtype must be float32, got {grad_output.dtype}")
    if output.dtype != ttnn.float32:
        raise ValueError(f"backward_softmax: output dtype must be float32, got {output.dtype}")
    if grad_output.dtype != output.dtype:
        raise ValueError(
            f"backward_softmax: grad_output dtype ({grad_output.dtype}) and "
            f"output dtype ({output.dtype}) must match"
        )

    # Layout.
    if grad_output.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"backward_softmax: grad_output must be TILE_LAYOUT, got {grad_output.layout}")
    if output.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"backward_softmax: output must be TILE_LAYOUT, got {output.layout}")

    # Rank.
    if len(grad_output.shape) != 4:
        raise ValueError(f"backward_softmax: grad_output rank must be 4, got {len(grad_output.shape)}")
    if len(output.shape) != 4:
        raise ValueError(f"backward_softmax: output rank must be 4, got {len(output.shape)}")

    # Shape match.
    if tuple(grad_output.shape) != tuple(output.shape):
        raise ValueError(
            f"backward_softmax: grad_output shape {tuple(grad_output.shape)} must "
            f"match output shape {tuple(output.shape)}"
        )

    # Tile alignment of H, W.
    H = grad_output.shape[-2]
    W = grad_output.shape[-1]
    if H % 32 != 0 or W % 32 != 0:
        raise ValueError(f"backward_softmax: H ({H}) and W ({W}) must each be a multiple of 32")

    # dim.
    if dim not in (-1, -2):
        raise ValueError(f"backward_softmax: dim must be -1 or -2, got {dim}")
