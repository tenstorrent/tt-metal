# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import tt_lib as ttl

from ttnn.tensor import (
    Tensor,
    MemoryConfig,
    DRAM_MEMORY_CONFIG,
)
from ttnn.core import reshape, _reshape_to_4D


def exp(input_tensor: Tensor) -> Tensor:
    original_shape = tuple(input_tensor.shape)
    input_tensor = _reshape_to_4D(input_tensor)
    ttl_input_tensor = input_tensor._tensor
    ttl_output_tensor = ttl.tensor.exp(ttl_input_tensor)
    output_tensor = Tensor(ttl_output_tensor)
    output_tensor = reshape(output_tensor, original_shape)
    return output_tensor


def gelu(input_tensor: Tensor, fast_and_approx=True) -> Tensor:
    original_shape = tuple(input_tensor.shape)
    input_tensor = _reshape_to_4D(input_tensor)
    ttl_input_tensor = input_tensor._tensor
    output_tensor = ttl.tensor.gelu(ttl_input_tensor, fast_and_approx=fast_and_approx)
    output_tensor = Tensor(output_tensor)
    output_tensor = reshape(output_tensor, original_shape)
    return output_tensor


def layer_norm(
    input_tensor: Tensor,
    *,
    epsilon: float = 1e-12,
    residual_input: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    memory_config: Optional[MemoryConfig] = DRAM_MEMORY_CONFIG,
) -> Tensor:
    original_shape = tuple(input_tensor.shape)
    input_tensor = _reshape_to_4D(input_tensor)
    if residual_input is not None:
        residual_input = _reshape_to_4D(residual_input)
    if weight is not None:
        weight = _reshape_to_4D(weight)
    if bias is not None:
        bias = _reshape_to_4D(bias)

    ttl_input_tensor = input_tensor._tensor
    residual_input = residual_input._tensor if residual_input is not None else None
    ttl_weight = weight._tensor if weight is not None else None
    ttl_bias = bias._tensor if bias is not None else None

    if residual_input is not None:
        output_tensor = ttl.tensor.add_layernorm(
            ttl_input_tensor, residual_input, epsilon, ttl_weight, ttl_bias, output_mem_config=memory_config
        )
    else:
        output_tensor = ttl.tensor.layernorm(
            ttl_input_tensor, epsilon, ttl_weight, ttl_bias, output_mem_config=memory_config
        )

    output_tensor = Tensor(output_tensor)
    output_tensor = reshape(output_tensor, original_shape)
    return output_tensor
