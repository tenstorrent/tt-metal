# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
linear — Phase 0: 2D matmul + optional row-broadcast bias.

  output = input @ weight              (no bias)
  output = input @ weight + bias[0, :] (bias mode; row 0 of [1,1,32,N] broadcast across M)

Single-core, single-Tensix, DRAM-interleaved bf16, TILE layout, M/K/N divisible by 32.
The compute path goes through the kernel_lib matmul_block (and bias_add) helpers.
"""

import ttnn

from .linear_program_descriptor import create_program_descriptor


def linear(
    input_tensor: ttnn.Tensor,
    weight_tensor: ttnn.Tensor,
    *,
    bias: ttnn.Tensor = None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Compute ``output = input_tensor @ weight_tensor (+ bias[0, :])``.

    Args:
        input_tensor: ``[1, 1, M, K]``, bf16, TILE_LAYOUT, DRAM-interleaved.
        weight_tensor: ``[1, 1, K, N]``, bf16, TILE_LAYOUT, DRAM-interleaved.
        bias: optional ``[1, 1, 32, N]``, bf16, TILE_LAYOUT, DRAM-interleaved.
              Row 0 holds the bias; rows 1-31 must be zero.
        memory_config: output memory config (default ``ttnn.DRAM_MEMORY_CONFIG``).

    Returns:
        ``[1, 1, M, N]``, bf16, TILE_LAYOUT.
    """
    _validate(input_tensor, weight_tensor, bias)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    M = int(input_tensor.shape[-2])
    N = int(weight_tensor.shape[-1])
    output_shape = [1, 1, M, N]

    # NOTE: allocate_tensor_on_device requires positional args, not keyword args.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        weight_tensor,
        output_tensor,
        bias=bias,
    )

    # Output tensor MUST be last in the tensor list passed to generic_op.
    if bias is not None:
        tensors = [input_tensor, weight_tensor, bias, output_tensor]
    else:
        tensors = [input_tensor, weight_tensor, output_tensor]
    return ttnn.generic_op(tensors, program_descriptor)


# ---------------------------------------------------------------------------
# Python-side validation (Phase 0 spec, op_design.md "Python-side validation").
# Failures raise ValueError; every check runs before any device work.
# ---------------------------------------------------------------------------


def _validate(input_tensor: ttnn.Tensor, weight_tensor: ttnn.Tensor, bias) -> None:
    _validate_one(input_tensor, "input")
    _validate_one(weight_tensor, "weight")
    if bias is not None:
        _validate_one(bias, "bias")

    in_shape = list(input_tensor.shape)
    w_shape = list(weight_tensor.shape)

    # Check 5: K match.
    if in_shape[-1] != w_shape[-2]:
        raise ValueError(f"linear: input K ({in_shape[-1]}) does not match weight K ({w_shape[-2]})")

    M = in_shape[-2]
    K = in_shape[-1]
    N = w_shape[-1]

    # Check 8: M, K, N divisible by 32.
    if (M % 32) != 0 or (K % 32) != 0 or (N % 32) != 0:
        raise ValueError(f"linear: M/K/N must be divisible by 32 (got M={M}, K={K}, N={N})")

    if bias is not None:
        b_shape = list(bias.shape)
        # Check 6: bias N matches weight N.
        if b_shape[-1] != N:
            raise ValueError(f"linear: bias N ({b_shape[-1]}) does not match weight N ({N})")
        # Check 7: bias height must be 32 (tile-padded single row).
        if b_shape[-2] != 32:
            raise ValueError(f"linear: bias height must be 32 (tile-padded single row), got {b_shape[-2]}")


def _validate_one(t: ttnn.Tensor, name: str) -> None:
    # Check 1: dtype must be bfloat16.
    if t.dtype != ttnn.bfloat16:
        raise ValueError(f"linear: {name} must be bfloat16 (got {t.dtype})")
    # Check 2: layout must be TILE_LAYOUT.
    if t.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"linear: {name} must be TILE_LAYOUT (got {t.layout})")
    # Check 3: rank must be 4.
    shape = list(t.shape)
    if len(shape) != 4:
        raise ValueError(f"linear: {name} must be rank 4 (got shape {shape})")
    # Check 4: leading two dims must be [1, 1, ...].
    if shape[0] != 1 or shape[1] != 1:
        raise ValueError(f"linear: {name} leading dims must be [1, 1, ...] (got shape {shape})")
