# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm — Main entry point.

Validates inputs, allocates output, creates program descriptor, launches via
ttnn.generic_op.

Math: output[..., i] = (x[..., i] - mean(x)) / sqrt(var(x) + eps) * gamma[i] + beta[i]
"""

import ttnn
from .layer_norm_rm_program_descriptor import create_program_descriptor


def layer_norm_rm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    *,
    epsilon: float = 1e-5,
    memory_config: ttnn.MemoryConfig = None,
    compute_kernel_config=None,
) -> ttnn.Tensor:
    """
    Layer-normalize each row of a row-major interleaved tensor.

    Args:
        input_tensor: Input tensor, bfloat16, ROW_MAJOR_LAYOUT, DRAM interleaved.
                      Shape [*, H, W] with H, W multiples of 32.
        gamma: Optional scale tensor, shape [1, 1, 1, W], bfloat16, ROW_MAJOR.
        beta:  Optional shift tensor, shape [1, 1, 1, W], bfloat16, ROW_MAJOR.
        epsilon: Small constant for numerical stability (default 1e-5).
        memory_config: Output memory config (default DRAM interleaved).
        compute_kernel_config: Optional dict with keys 'fp32_dest_acc_en' (bool)
            and/or 'math_fidelity' (ttnn.MathFidelity). Controls compute precision.

    Returns:
        Output tensor with same shape, dtype, and layout as input.
    """
    _validate_inputs(input_tensor, gamma, beta)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_shape = list(input_tensor.shape)

    # CRITICAL: allocate_tensor_on_device requires POSITIONAL args
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        gamma,
        beta,
        epsilon,
        compute_kernel_config=compute_kernel_config,
    )

    # Build tensor list — output MUST be last
    tensors = [input_tensor]
    if gamma is not None:
        tensors.append(gamma)
    if beta is not None:
        tensors.append(beta)
    tensors.append(output_tensor)

    return ttnn.generic_op(tensors, program_descriptor)


def _validate_inputs(input_tensor, gamma, beta):
    """Validate all input tensors meet requirements."""
    if len(input_tensor.shape) < 2:
        raise ValueError("layer_norm_rm: input must have at least 2 dimensions")

    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"layer_norm_rm: input must be ROW_MAJOR_LAYOUT, got {input_tensor.layout}")

    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"layer_norm_rm: input must be bfloat16, got {input_tensor.dtype}")

    W = input_tensor.shape[-1]
    H = input_tensor.shape[-2]

    if W % 32 != 0:
        raise ValueError(f"layer_norm_rm: width must be a multiple of 32, got {W}")
    if H % 32 != 0:
        raise ValueError(f"layer_norm_rm: height must be a multiple of 32, got {H}")

    if gamma is not None:
        if gamma.shape[-1] != W:
            raise ValueError(f"layer_norm_rm: gamma width ({gamma.shape[-1]}) must match " f"input width ({W})")

    if beta is not None:
        if beta.shape[-1] != W:
            raise ValueError(f"layer_norm_rm: beta width ({beta.shape[-1]}) must match " f"input width ({W})")
