# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from .layernorm_program_descriptor import create_program_descriptor


def layernorm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor,
    beta: ttnn.Tensor,
    *,
    eps: float = 1e-5,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    _validate_input(input_tensor)
    W = input_tensor.shape[-1]
    _validate_weight(gamma, W, "gamma")
    _validate_weight(beta, W, "beta")

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, gamma, beta, output_tensor, eps)

    # Output tensor MUST be last in the list
    return ttnn.generic_op([input_tensor, gamma, beta, output_tensor], program_descriptor)


def _validate_input(tensor: ttnn.Tensor) -> None:
    if len(tensor.shape) < 2:
        raise ValueError("layernorm: input must have at least 2 dimensions")
    if tensor.dtype != ttnn.bfloat16:
        raise ValueError("layernorm: input must be bfloat16")
    if tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("layernorm: input must be ROW_MAJOR_LAYOUT")
    if tensor.shape[-1] % 32 != 0:
        raise ValueError("layernorm: last dimension must be divisible by 32")
    if tensor.shape[-2] % 32 != 0:
        raise ValueError("layernorm: second-to-last dimension must be divisible by 32")


def _validate_weight(tensor: ttnn.Tensor, W: int, name: str) -> None:
    if len(tensor.shape) != 1:
        raise ValueError(f"layernorm: {name} must be 1D")
    if tensor.shape[0] != W:
        raise ValueError(f"layernorm: {name} must have shape [{W}], got [{tensor.shape[0]}]")
    if tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"layernorm: {name} must be bfloat16")
    if tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"layernorm: {name} must be ROW_MAJOR_LAYOUT")
