# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import tt_lib as ttl

import ttnn


def _golden_function(
    input_tensor: ttnn.Tensor,
    kv_input_tensor: Optional[ttnn.Tensor] = None,
    *,
    num_heads,
    num_kv_heads=None,
    transpose_key=True,
    **_,
):
    import torch

    if kv_input_tensor is not None:
        input_tensor = torch.cat([input_tensor, kv_input_tensor], dim=-1)

    if num_kv_heads is None:
        num_kv_heads = num_heads

    batch_size, sequence_size, hidden_size = input_tensor.shape
    # Subtract head sizes for key and value
    head_size = (hidden_size) // (num_heads + num_kv_heads * 2)
    tensor = torch.reshape(input_tensor, (batch_size, sequence_size, num_heads + num_kv_heads * 2, head_size))
    query, key, value = (
        tensor[..., :num_heads, :],
        tensor[..., num_heads : num_heads + num_kv_heads, :],
        tensor[..., num_heads + num_kv_heads :, :],
    )

    query = torch.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    key = torch.reshape(key, (batch_size, sequence_size, num_kv_heads, head_size))
    value = torch.reshape(value, (batch_size, sequence_size, num_kv_heads, head_size))

    query = torch.permute(query, (0, 2, 1, 3)).contiguous().clone()
    key = torch.permute(key, (0, 2, 1, 3)).contiguous().clone()
    value = torch.permute(value, (0, 2, 1, 3)).contiguous().clone()
    if transpose_key:
        key = torch.permute(key, (0, 1, 3, 2)).contiguous().clone()

    return query, key, value


split_doc_string = r"""
    split_query_key_value_and_split_heads(input_tensor: ttnn.Tensor, kv_input_tensor: Optional[ttnn.Tensor] = None, *, num_heads: int, num_kv_heads: Optional[int] = None, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]

    Splits :attr:`input_tensor` of shape ``[batch_size, sequence_size, 3 * hidden_size]`` into 3 tensors (Query, Key, Value) of shape ``[batch_size, sequence_size, hidden_size]``.
    Then, reshapes and permutes the output tensors, to make them ready for computing attention scores.

    If :attr:`kv_input_tensor` is passed in, then :attr:`input_tensor` of shape ``[batch_size, sequence_size, hidden_size]`` is only used for Query,
    and :attr:`kv_input_tensor` of shape ``[batch_size, sequence_size, 2 * hidden_size]`` is used for Key and Value.

    For the sharded implementation, the input query, key and value are expected to be concatenated such that the heads are interleaved (q1 k1 v1...qn kn vn).

    Equivalent pytorch code:

    .. code-block:: python

        if kv_input_tensor is not None:
            input_tensor = torch.cat([input_tensor, kv_input_tensor], dim=-1)

        if num_kv_heads is None:
            num_kv_heads = num_heads

        batch_size, sequence_size, hidden_size = input_tensor.shape
        # Subtract head sizes for key and value
        head_size = (hidden_size) // (num_heads + num_kv_heads * 2)
        tensor = torch.reshape(input_tensor, (batch_size, sequence_size, num_heads + num_kv_heads * 2, head_size))
        query, key, value = (
            tensor[..., :num_heads, :],
            tensor[..., num_heads:num_heads + num_kv_heads, :],
            tensor[..., num_heads + num_kv_heads:, :],
        )

        query = torch.reshape(query, (batch_size, sequence_size, num_heads, head_size))
        key = torch.reshape(key, (batch_size, sequence_size, num_kv_heads, head_size))
        value = torch.reshape(value, (batch_size, sequence_size, num_kv_heads, head_size))

        query = torch.permute(query, (0, 2, 1, 3)).contiguous().clone()
        key = torch.permute(key, (0, 2, 1, 3)).contiguous().clone()
        value = torch.permute(value, (0, 2, 1, 3)).contiguous().clone()
        if transpose_key:
            key = torch.permute(key, (0, 1, 3, 2)).contiguous().clone()

        return query, key, value

    Args:
        * :attr:`input_tensor`: Input Tensor for Query, Key and Value. If :attr:`kv_input_tensor` is not None, then :attr:`input_tensor` is only used for Query.
        * :attr:`kv_input_tensor`: Input Tensor for Key and Value. If passed in, :attr:`input_tensor` has to be used only for Query.
        * :attr:`num_heads`: num heads to split into
        * :attr:`num_kv_heads`: num heads of Key and num heads of Value. If not passed in, then :attr:`num_kv_heads` is set to :attr:`num_heads`
        * :attr:`transpose_key`: Whether to transpose the Key tensor on the last two dimensions
        * :attr:`memory_config`: Memory Config of the output tensor

"""

split_query_key_value_and_split_heads = ttnn.register_operation(
    name="ttnn.transformer.split_query_key_value_and_split_heads",
    golden_function=_golden_function,
    doc=split_doc_string,
)(ttnn._ttnn.operations.transformer.split_query_key_value_and_split_heads)


def _golden_function(input_tensor: ttnn.Tensor, *, head_size: int, attention_mask, **_):
    import torch

    if head_size is not None:
        scaler = 1 / (head_size**0.5)
    else:
        scaler = 1.0

    input_tensor = input_tensor * scaler

    if attention_mask is not None:
        input_tensor += attention_mask

    return torch.softmax(input_tensor, -1)


def _attention_softmax_validate_input_tensors(operation_name, input_tensor, *args, attention_mask, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        attention_mask,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
        is_optional=True,
    )


@ttnn.register_operation(
    name="ttnn.transformer.attention_softmax",
    validate_input_tensors=_attention_softmax_validate_input_tensors,
    golden_function=_golden_function,
)
def attention_softmax(
    input_tensor: ttnn.Tensor,
    *,
    head_size: Optional[int],
    attention_mask: Optional[ttnn.Tensor],
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    program_config: Optional[
        ttl.operations.primary.transformers.SoftmaxProgramConfig
    ] = ttl.operations.primary.transformers.SoftmaxDefaultProgramConfig(),
) -> ttnn.Tensor:
    """
    attention_softmax(input_tensor: ttnn.Tensor, *, head_size: int, attention_mask: Optional[ttnn.Tensor], memory_config: MemoryConfig = DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Divides :attr:`input_tensor` by the square root of :attr:`head_size`, adds :attr:`attention_mask` (optionally) and computes softmax

    Args:
        * :attr:`input_tensor`: Input Tensor
        * :attr:`head_size`: Number of heads
        * :attr:`attention_mask`: Attention Mask
        * :attr:`memory_config`: Memory Config of the output tensor

    """
    if head_size is not None:
        scaler = 1 / (head_size**0.5)
    else:
        scaler = 1.0

    if attention_mask is not None:
        output_tensor = ttl.tensor.scale_mask_softmax(
            input_tensor,
            scaler,
            attention_mask,
            output_mem_config=memory_config,
        )
    else:
        scaled_input_tensor = input_tensor * scaler
        output_tensor = ttl.tensor.softmax(scaled_input_tensor, output_mem_config=memory_config)
    return output_tensor


doc = r"""
attention_softmax_(tensor: ttnn.Tensor, *, head_size: int, attention_mask: Optional[ttnn.Tensor], program_config: Optional[SoftmaxProgramConfig] = SoftmaxDefaultProgramConfig(),  memory_config: Optional[ttnn.MemoryConfig] = input_tensor.memory_config()) -> ttnn.Tensor

In-Place divides :attr:`tensor` by the square root of :attr:`head_size`, adds :attr:`attention_mask` (optionally) and computes softmax.

Args:
    * :attr:`tensor`: Input Tensor
    * :attr:`head_size`: Number of heads
    * :attr:`attention_mask`: Attention Mask
    * :attr:`program_config`: Program Config of the output tensor
    * :attr:`memory_config`: Memory Config of the output tensor, defaults to input_tensor.memory_config()

"""
attention_softmax_ = ttnn.register_operation(
    name="ttnn.transformer.attention_softmax_",
    golden_function=_golden_function,
    doc=doc,
)(ttnn._ttnn.operations.transformer.attention_softmax_)


def _golden_function(input_tensor: ttnn.Tensor, **_):
    import torch

    batch_size, num_heads, sequence_size, head_size = input_tensor.shape

    output_tensor = torch.permute(input_tensor, (0, 2, 1, 3)).contiguous().clone()
    output_tensor = (
        torch.reshape(output_tensor, (batch_size, sequence_size, num_heads * head_size)).contiguous().clone()
    )
    return output_tensor


doc = r"""
concatenate_heads(input_tensor: ttnn.Tensor, *, memory_config: MemoryConfig = input_tensor.memory_config()) -> ttnn.Tensor

Takes in a tensor of shape ``[batch_size, num_heads, sequence_size, head_size]``, concatenates heads back along the width dimension and returns the tensor of shape ``[batch_size, sequence_size, num_heads * head_size]``

Args:
    * :attr:`input_tensor`: Input Tensor
    * :attr:`memory_config`: Memory Config of the output tensor, defaults to input_tensor.memory_config()
"""

concatenate_heads = ttnn.register_operation(
    name="ttnn.transformer.concatenate_heads", golden_function=_golden_function, doc=doc
)(ttnn._ttnn.operations.transformer.concatenate_heads)


def _rotary_embedding_validate_input_tensors(operation_name, input_tensor, cos_cache, sin_cache, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        cos_cache,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        sin_cache,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.transformer.rotary_embedding",
    validate_input_tensors=_rotary_embedding_validate_input_tensors,
)
def rotary_embedding(
    input_tensor: ttnn.Tensor,
    cos_cache: ttnn.Tensor,
    sin_cache: ttnn.Tensor,
    token_index: int,
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    """

    rotary_embedding(input_tensor: ttnn.Tensor, cos_cache: ttnn.Tensor, sin_cache: ttnn.Tensor, token_index: int, memory_config: MemoryConfig) -> ttnn.Tensor

    Applies the rotary embedding to the input_tensor tensor using the cos_cache and sin_cache tensors.

    """
    return ttnn.experimental.tensor.rotary_embedding(
        input_tensor, cos_cache, sin_cache, token_index, output_mem_config=memory_config
    )


__all__ = []
