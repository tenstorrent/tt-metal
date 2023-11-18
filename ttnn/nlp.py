# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import tt_lib as ttl

from ttnn.tensor import (
    Tensor,
    MemoryConfig,
    DRAM_MEMORY_CONFIG,
)
from ttnn.core import reshape, softmax


def split_fused_qkv_and_split_heads(
    input_tensor: Tensor,
    *,
    core_grid: Tuple[int, int],
    memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    split_fused_qkv_and_split_heads(input_tensor: ttnn.Tensor, *, core_grid: Tuple[int, int], memory_config: MemoryConfig = DRAM_MEMORY_CONFIG) -> Tuple[Tensor, Tensor, Tensor]

    Splits tensor of shape [batch_size, sequence_size, 3 * hidden_size] into 3 tensors (Query, Key, Value) of shape [batch_size, sequence_size, hidden_size]. Then, reshapes and permutes them, to make them ready for computing attention scores

    Args:
        * :attr:`input_tensor`: Input Tensor
        * :attr:`core_grid`: Compute and Storage Core Grid to use for the operation
        * :attr:`memory_config`: Memory Config of the output tensor

    """
    if len(input_tensor.shape) != 3:
        raise RuntimeError("Input Tensor must have strictly 3 dimensions!")

    batch_size, sequence_size, hidden_size = input_tensor.shape
    input_tensor = reshape(input_tensor, (batch_size, 1, sequence_size, hidden_size))

    ttl_input_tensor = input_tensor._tensor

    core_y, core_x = core_grid
    query_key_value = ttl.operations.primary.transformers.split_fused_qkv_and_split_heads(
        ttl_input_tensor,
        ttl.tensor.CoreCoord(core_x, core_y),
        memory_config,
    )
    query_key_value = (Tensor(ttl_tensor) for ttl_tensor in query_key_value)
    query, key, value = query_key_value
    return query, key, value


def attention_softmax(
    input_tensor: Tensor,
    *,
    head_size: int,
    attention_mask: Optional[Tensor],
) -> Tensor:
    """
    attention_softmax(input_tensor: ttnn.Tensor, *, head_size: int, attention_mask: Optional[Tensor]) -> Tensor

    Divides :attr:`input_tensor` by the square root of :attr:`head_size`, adds :attr:`attention_mask` (optionally) and computes softmax

    Args:
        * :attr:`input_tensor`: Input Tensor
        * :attr:`core_grid`: Compute and Storage Core Grid to use for the operation
        * :attr:`memory_config`: Memory Config of the output tensor

    """
    if len(input_tensor.shape) != 4:
        raise RuntimeError("Input Tensor must have strictly 3 dimensions!")

    scaler = 1 / (head_size**0.5)

    if attention_mask is not None:
        output_tensor = ttl.tensor.scale_mask_softmax(input_tensor._tensor, scaler, attention_mask._tensor)
        return Tensor(output_tensor)
    else:
        scaled_input_tensor = input_tensor * scaler
        ttl_scaled_input_tensor = scaled_input_tensor._tensor
        ttl_output_tensor = ttl.tensor.softmax(ttl_scaled_input_tensor)
        return Tensor(ttl_output_tensor)


def attention_softmax_(
    input_tensor: Tensor,
    *,
    head_size: int,
    attention_mask: Optional[Tensor],
) -> Tensor:
    """
    attention_softmax_(input_tensor: ttnn.Tensor, *, head_size: int, attention_mask: Optional[Tensor]) -> Tensor

    Divides :attr:`input_tensor` by the square root of :attr:`head_size`, adds :attr:`attention_mask` (optionally) and computes softmax. In-Place.

    Args:
        * :attr:`input_tensor`: Input Tensor
        * :attr:`core_grid`: Compute and Storage Core Grid to use for the operation
        * :attr:`memory_config`: Memory Config of the output tensor

    """
    if len(input_tensor.shape) != 4:
        raise RuntimeError("Input Tensor must have strictly 3 dimensions!")

    scaler = 1 / (head_size**0.5)

    if attention_mask is not None:
        output_tensor = ttl.operations.primary.transformers.scale_mask_softmax_in_place(
            input_tensor._tensor, scaler, attention_mask._tensor
        )
        return Tensor(output_tensor)
    else:
        raise RuntimeError("Cannot apply divide by sqrt(head_size) using in-place version!")


def concatenate_heads(
    input_tensor: Tensor,
    *,
    core_grid: Tuple[int, int],
    memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
) -> Tensor:
    """
    concatenate_heads(input_tensor: ttnn.Tensor, *, core_grid: Tuple[int, int], memory_config: MemoryConfig = DRAM_MEMORY_CONFIG) -> Tensor

    Takes in a tensor of shape [batch_size, num_heads, sequence_size, head_size], concatenates heads back along the width dimension and return the tensor of [batch_size, sequence_size, num_heads * head_size]

    Args:
        * :attr:`input_tensor`: Input Tensor
        * :attr:`core_grid`: Compute and Storage Core Grid to use for the operation
        * :attr:`memory_config`: Memory Config of the output tensor

    """
    if len(input_tensor.shape) != 4:
        raise RuntimeError("Input Tensor must have strictly 4 dimensions!")

    batch_size, num_heads, sequence_size, head_size = input_tensor.shape
    core_y, core_x = core_grid

    ttl_input_tensor = input_tensor._tensor
    ttl_output_tensor = ttl.operations.primary.transformers.concatenate_heads(
        ttl_input_tensor,
        ttl.tensor.CoreCoord(core_x, core_y),
        memory_config,
    )
    output_tensor = Tensor(ttl_output_tensor)
    output_tensor = reshape(output_tensor, (batch_size, sequence_size, num_heads * head_size))

    return output_tensor
