# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import tt_lib as ttl

import ttnn


def _split_query_key_value_and_split_heads_validate_input_tensors(
    operation_name, input_tensor, kv_input_tensor=None, *args, **kwargs
):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(3,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        kv_input_tensor,
        ranks=(3,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
        is_optional=True,
    )


def _torch_split_query_key_value_and_split_heads(
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


def _fallback_split_query_key_value_and_split_heads(
    input_tensor: ttnn.Tensor,
    kv_input_tensor: Optional[ttnn.Tensor] = None,
    *,
    num_heads,
    num_kv_heads=None,
    transpose_key=True,
    **_,
):
    input_tensor = ttnn.to_torch(input_tensor)
    return _torch_split_query_key_value_and_split_heads(
        input_tensor, kv_input_tensor, num_heads=num_heads, num_kv_heads=num_kv_heads, transpose_key=transpose_key
    )


@ttnn.register_operation(
    name="ttnn.transformer.split_query_key_value_and_split_heads",
    torch_function=_fallback_split_query_key_value_and_split_heads,
    validate_input_tensors=_split_query_key_value_and_split_heads_validate_input_tensors,
)
def split_query_key_value_and_split_heads(
    input_tensor: ttnn.Tensor,
    kv_input_tensor: Optional[ttnn.Tensor] = None,
    *,
    num_heads: int,
    num_kv_heads: Optional[int] = None,
    transpose_key: bool = True,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """
    split_query_key_value_and_split_heads(input_tensor: ttnn.Tensor, kv_input_tensor: Optional[ttnn.Tensor] = None, *, num_heads: int, num_kv_heads: Optional[int] = None, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]

    Splits :attr:`input_tensor` of shape ``[batch_size, sequence_size, 3 * hidden_size]`` into 3 tensors (Query, Key, Value) of shape ``[batch_size, sequence_size, hidden_size]``.
    Then, reshapes and permutes the output tensors, to make them ready for computing attention scores.

    If :attr:`kv_input_tensor` is passed in, then :attr:`input_tensor` of shape ``[batch_size, sequence_size, hidden_size]`` is only used for Query,
    and :attr:`kv_input_tensor` of shape ``[batch_size, sequence_size, 2 * hidden_size]`` is used for Key and Value.

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
    if len(input_tensor.shape) != 3:
        raise RuntimeError("Input Tensor must have strictly 3 dimensions!")

    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError("Input Tensor must be in a TILE_LAYOUT!")

    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise RuntimeError("input_tensor must be on device!")

    if num_kv_heads is not None:
        if transpose_key is True:
            raise RuntimeError("transpose_key cannot be True when num_kv_heads is passed in!")

        batch_size, sequence_size, num_heads_plus_num_kv_heads_times_2_times_hidden_size = input_tensor.shape
        (
            _,
            sequence_size_padded,
            num_heads_plus_num_kv_heads_times_2_times_hidden_size_padded,
        ) = input_tensor.shape.with_tile_padding()
        head_size = num_heads_plus_num_kv_heads_times_2_times_hidden_size // (num_heads + num_kv_heads * 2)
        padded_head_size = num_heads_plus_num_kv_heads_times_2_times_hidden_size_padded // (
            num_heads + num_kv_heads * 2
        )

        if head_size % ttnn.TILE_SIZE != 0:
            raise RuntimeError(
                f"Head size must be a multiple of {ttnn.TILE_SIZE}! Update the preceding matmul to have the padding in the weights!"
            )

        if padded_head_size - head_size != 0:
            raise RuntimeError("Head size cannot have tile padding!")

        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
        query, key, value = ttnn.experimental.tensor.nlp_create_qkv_heads_falcon7b(
            input_tensor,
            output_mem_config=memory_config,
        )
        return query, key, value

    if kv_input_tensor is not None:
        if num_kv_heads is not None:
            raise RuntimeError("num_kv_heads cannot be True when kv_input_tensor is passed in!")
        batch_size, sequence_size, hidden_size = input_tensor.shape
        _, sequence_size_padded, hidden_size_padded = input_tensor.shape.with_tile_padding()
        if kv_input_tensor.shape != (batch_size, sequence_size, hidden_size * 2):
            raise RuntimeError(
                "kv_input_tensor must be of shape (batch_size, sequence_size, hidden_size * 2) when input_tensor is of shape (batch_size, sequence_size, hidden_size)"
            )
    else:
        if num_kv_heads is not None:
            raise RuntimeError("num_kv_heads cannot be True!")
        batch_size, sequence_size, three_times_hidden_size = input_tensor.shape
        _, sequence_size_padded, three_times_hidden_size_padded = input_tensor.shape.with_tile_padding()
        hidden_size = three_times_hidden_size // 3
        hidden_size_padded = three_times_hidden_size_padded // 3

    if not transpose_key:
        raise RuntimeError("transpose_key must be True when num_kv_heads is not passed in!")

    head_size = hidden_size // num_heads
    padded_head_size = hidden_size_padded // num_heads

    if head_size % ttnn.TILE_SIZE != 0:
        raise RuntimeError(
            f"Head size must be a multiple of {ttnn.TILE_SIZE}! Update the preceding matmul to have the padding in the weights!"
        )

    if padded_head_size - head_size != 0:
        raise RuntimeError("Head size cannot have tile padding!")

    if ttnn.is_sharded(input_tensor):
        input_tensor = ttnn.reshape(
            input_tensor,
            ttnn.Shape(
                [batch_size, 1, sequence_size, three_times_hidden_size],
                [batch_size, 1, sequence_size_padded, three_times_hidden_size_padded],
            ),
        )

        query, key, value = ttnn.experimental.operations.primary.transformers.split_query_key_value_and_split_heads(
            input_tensor,
            input_tensor.device().compute_with_storage_grid_size(),
            memory_config,
            num_heads=num_heads,
        )
        return query, key, value
    else:
        if kv_input_tensor is not None:
            input_tensor = ttnn.reshape(
                input_tensor,
                ttnn.Shape(
                    [batch_size, 1, sequence_size, hidden_size],
                    [batch_size, 1, sequence_size_padded, hidden_size_padded],
                ),
            )

            _, kv_sequence_size, _ = kv_input_tensor.shape
            _, kv_sequence_size, _ = kv_input_tensor.shape.with_tile_padding()
            desired_shape = ttnn.Shape(
                [batch_size, 1, kv_sequence_size, hidden_size * 2],
                [batch_size, 1, kv_sequence_size, hidden_size_padded * 2],
            )
            kv_input_tensor = ttnn.reshape(kv_input_tensor, desired_shape)
        else:
            input_tensor = ttnn.reshape(
                input_tensor,
                ttnn.Shape(
                    [batch_size, 1, sequence_size, three_times_hidden_size],
                    [batch_size, 1, sequence_size_padded, three_times_hidden_size_padded],
                ),
            )

        query_key_value = ttnn.experimental.tensor.nlp_create_qkv_heads(
            input_tensor,
            kv_input_tensor,
            num_heads=num_heads,
            output_mem_config=memory_config,
        )
        query, key, value = query_key_value

        head_size = hidden_size // num_heads
        query = ttnn.reshape(
            query,
            ttnn.Shape(
                [batch_size, num_heads, sequence_size, head_size],
                [batch_size, num_heads, sequence_size_padded, head_size],
            ),
        )
        key = ttnn.reshape(
            key,
            ttnn.Shape(
                [batch_size, num_heads, head_size, sequence_size],
                [
                    batch_size,
                    num_heads,
                    head_size,
                    sequence_size_padded,
                ],
            ),
        )
        value = ttnn.reshape(
            value,
            ttnn.Shape(
                [batch_size, num_heads, sequence_size, head_size],
                [batch_size, num_heads, sequence_size_padded, head_size],
            ),
        )

        return query, key, value


def _torch_attention_softmax(input_tensor: ttnn.Tensor, *, head_size: int, attention_mask, **_):
    import torch

    if head_size is not None:
        scaler = 1 / (head_size**0.5)
    else:
        scaler = 1.0

    input_tensor *= scaler

    if attention_mask is not None:
        input_tensor += attention_mask[..., :1, :]

    return torch.softmax(input_tensor, -1)


def _fallback_attention_softmax(input_tensor: ttnn.Tensor, *, head_size: int, attention_mask, **_):
    input_tensor = ttnn.to_torch(input_tensor)

    if attention_mask is not None:
        attention_mask = ttnn.from_device(attention_mask)
        attention_mask = ttnn.to_layout(attention_mask, ttnn.ROW_MAJOR_LAYOUT)
        attention_mask = ttnn.to_torch(attention_mask)

    return _torch_attention_softmax(input_tensor, head_size=head_size, attention_mask=attention_mask)


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
    torch_function=_fallback_attention_softmax,
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
            program_config=program_config,
        )
    else:
        scaled_input_tensor = input_tensor * scaler
        output_tensor = ttl.tensor.softmax(scaled_input_tensor, output_mem_config=memory_config)
    return output_tensor


@ttnn.register_operation(
    name="ttnn.transformer.attention_softmax_",
    validate_input_tensors=_attention_softmax_validate_input_tensors,
    torch_function=_torch_attention_softmax,
)
def attention_softmax_(
    tensor: ttnn.Tensor,
    *,
    head_size: Optional[int],
    attention_mask: Optional[ttnn.Tensor],
    program_config: Optional[
        ttl.operations.primary.transformers.SoftmaxProgramConfig
    ] = ttl.operations.primary.transformers.SoftmaxDefaultProgramConfig(),
) -> ttnn.Tensor:
    """
    attention_softmax_(tensor: ttnn.Tensor, *, head_size: int, attention_mask: Optional[ttnn.Tensor], program_config: Optional[SoftmaxProgramConfig] = SoftmaxDefaultProgramConfig()) -> ttnn.Tensor

    In-Place divides :attr:`tensor` by the square root of :attr:`head_size`, adds :attr:`attention_mask` (optionally) and computes softmax.

    Args:
        * :attr:`tensor`: Input Tensor
        * :attr:`head_size`: Number of heads
        * :attr:`attention_mask`: Attention Mask
        * :attr:`program_config`: Program Config of the output tensor


    """
    if len(tensor.shape) != 4:
        raise RuntimeError("Input Tensor must have strictly 4 dimensions!")

    if tensor.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError("Input Tensor must be in a TILE_LAYOUT!")

    if head_size is not None:
        scaler = 1 / (head_size**0.5)
    else:
        scaler = 1.0

    if attention_mask is not None:
        input_shape = tensor.shape
        input_shape_with_tile_padding = tensor.shape.with_tile_padding()
        tensor = ttnn.reshape(
            tensor,
            (
                input_shape_with_tile_padding[0],
                1,
                input_shape_with_tile_padding[1] * input_shape_with_tile_padding[2],
                input_shape_with_tile_padding[3],
            ),
        )
        tensor = ttl.operations.primary.transformers.scale_mask_softmax_in_place(
            tensor, scaler, attention_mask, program_config=program_config
        )
        tensor = ttnn.reshape(tensor, input_shape)
        return tensor
    else:
        raise RuntimeError("Cannot apply divide by sqrt(head_size) using in-place version!")


def _torch_concatenate_heads(input_tensor: ttnn.Tensor, **_):
    import torch

    batch_size, num_heads, sequence_size, head_size = input_tensor.shape

    output_tensor = torch.permute(input_tensor, (0, 2, 1, 3)).contiguous().clone()
    output_tensor = (
        torch.reshape(output_tensor, (batch_size, sequence_size, num_heads * head_size)).contiguous().clone()
    )
    return output_tensor


def _fallback_concatenate_heads(input_tensor: ttnn.Tensor, **_):
    input_tensor = ttnn.to_torch(input_tensor)
    return _torch_concatenate_heads(input_tensor)


def _concatenate_heads_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.transformer.concatenate_heads",
    validate_input_tensors=_concatenate_heads_validate_input_tensors,
    torch_function=_fallback_concatenate_heads,
)
def concatenate_heads(
    input_tensor: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """
    concatenate_heads(input_tensor: ttnn.Tensor, *, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Takes in a tensor of shape ``[batch_size, num_heads, sequence_size, head_size]``, concatenates heads back along the width dimension and returns the tensor of shape ``[batch_size, sequence_size, num_heads * head_size]``

    Args:
        * :attr:`input_tensor`: Input Tensor
        * :attr:`memory_config`: Memory Config of the output tensor

    """
    batch_size, num_heads, sequence_size, head_size = input_tensor.shape
    batch_size, num_heads, padded_sequence_size, padded_head_size = input_tensor.shape.with_tile_padding()

    if head_size % ttnn.TILE_SIZE != 0:
        raise RuntimeError(
            f"Head size must be a multiple of {ttnn.TILE_SIZE}! Update matmul that uses the output of this operation to have the padding in the weights!"
        )

    if padded_head_size - head_size != 0:
        raise RuntimeError("Head size cannot have tile padding!")

    output_tensor = ttl.tensor.nlp_concat_heads(
        input_tensor,
        memory_config,
    )
    output_tensor = ttnn.reshape(
        output_tensor,
        ttnn.Shape(
            (batch_size, sequence_size, num_heads * head_size),
            (batch_size, padded_sequence_size, num_heads * padded_head_size),
        ),
    )

    return output_tensor


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
