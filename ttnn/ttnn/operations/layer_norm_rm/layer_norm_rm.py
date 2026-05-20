# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm — LayerNorm op that natively handles both ROW_MAJOR and TILE input layouts.

Implements:
    out[..., i, j] = gamma[j] * (x[..., i, j] - mean_i) / sqrt(var_i + eps) + beta[j]

where mean_i and var_i are computed over the last dimension. gamma and beta are optional
and, when supplied, must be ROW_MAJOR (1, 1, 1, W) tensors whose dtype matches the input.

Algorithm: three-pass streaming (input read three times from DRAM):
  Pass 1: mean    via reduce<SUM, REDUCE_ROW> with scaler 1/W
  Pass 2: variance via sub<COL> + square_in_place + reduce<SUM, REDUCE_ROW>,
                  then transform_in_place to rsqrt(var + eps)
  Pass 3: normalize + optional affine + drain, per-row.

Single-core; loops over `total_tile_rows = ceil(prod(shape[:-1]) / 32)` row-blocks.
"""

import ttnn
from .layer_norm_rm_program_descriptor import create_program_descriptor


def layer_norm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor | None = None,
    beta: ttnn.Tensor | None = None,
    *,
    epsilon: float = 1e-5,
    compute_kernel_config: "ttnn.ComputeConfigDescriptor | None" = None,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """
    Per-last-dim LayerNorm with optional gamma scale and beta shift.

    Args:
        input_tensor: rank>=2 ttnn.Tensor on device. dtype ∈ {bfloat16, float32, bfloat8_b}.
            layout ∈ {ROW_MAJOR_LAYOUT, TILE_LAYOUT}.
        gamma: optional (1, 1, 1, W) ROW_MAJOR tensor with same dtype as input.
        beta: optional (1, 1, 1, W) ROW_MAJOR tensor with same dtype as input.
        epsilon: small positive constant added to variance before sqrt. Default 1e-5.
        compute_kernel_config: optional ComputeConfigDescriptor controlling math fidelity, etc.
        memory_config: output memory config (default: DRAM interleaved).

    Returns:
        ttnn.Tensor with same shape/dtype/layout as `input_tensor`.
    """
    _validate(input_tensor, gamma, beta, epsilon)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    # Build kernel inputs list — order must match the runtime args assumed by the kernels.
    op_inputs: list[ttnn.Tensor] = [input_tensor]
    if gamma is not None:
        op_inputs.append(gamma)
    if beta is not None:
        op_inputs.append(beta)
    # Output tensor MUST be last.
    op_inputs.append(output_tensor)

    program_descriptor = create_program_descriptor(
        input_tensor,
        gamma,
        beta,
        output_tensor,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
    )

    return ttnn.generic_op(op_inputs, program_descriptor)


def _validate(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor | None,
    beta: ttnn.Tensor | None,
    epsilon: float,
) -> None:
    if len(input_tensor.shape) < 2:
        raise ValueError(f"layer_norm: input must have rank >= 2 (got rank {len(input_tensor.shape)})")

    if input_tensor.dtype not in (ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b):
        raise ValueError(f"layer_norm: unsupported input dtype {input_tensor.dtype}")

    if input_tensor.dtype == ttnn.bfloat8_b and input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("layer_norm: bfloat8_b + ROW_MAJOR_LAYOUT is structurally invalid")

    if not (epsilon > 0):
        raise ValueError(f"layer_norm: epsilon must be > 0 (got {epsilon})")

    W = int(input_tensor.shape[-1])

    for name, t in (("gamma", gamma), ("beta", beta)):
        if t is None:
            continue
        # Width must match input's last dim.
        if int(t.shape[-1]) != W:
            raise ValueError(f"layer_norm: {name}.shape[-1] ({int(t.shape[-1])}) must equal input W ({W})")
        # Layout must be ROW_MAJOR.
        if t.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(f"layer_norm: {name} must be ROW_MAJOR_LAYOUT (got {t.layout})")
        # Dtype must equal input dtype.
        if t.dtype != input_tensor.dtype:
            raise ValueError(f"layer_norm: {name}.dtype ({t.dtype}) must equal input dtype ({input_tensor.dtype})")
