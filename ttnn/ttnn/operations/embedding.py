# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Optional


from loguru import logger

import tt_lib as ttl

import ttnn


def _golden_function(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, **_):
    import torch

    output_tensor = torch.nn.functional.embedding(input_tensor, weight)
    return output_tensor


def _embedding_validate_input_tensors(operation_name, input_tensor, weight, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.uint32, ttnn.bfloat16),
        layouts=(ttnn.ROW_MAJOR_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        weight,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16,),
        layouts=(ttnn.ROW_MAJOR_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.embedding",
    validate_input_tensors=_embedding_validate_input_tensors,
    golden_function=_golden_function,
)
def embedding(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    pad_token: Optional[int] = None,
    layout: ttnn.Layout = ttnn.ROW_MAJOR_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
):
    r"""
    embedding(inxput_tensor: ttnn.Tensor, weight: ttnn.Tensor, *, pad_token: Optional[int] = None, layout: ttnn.Layout = ttnn.ROW_MAJOR_LAYOUT, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Retrieves word embeddings using input_tensor. The input_tensor is a list of indices, and the embedding matrix, and the output is the corresponding word embeddings.

    Args:
        * :attr:`input_tensor`: the indices ttnn.Tensor
        * :attr:`weight`: the embeddings ttnn.Tensor that correspond to the indices ttnn.Tensor

    Keyword Args:
        * :attr:`pad_token`: the padding token. Default is None.
        * :attr:`layout`: the layout of the input and output tensors. Default is ttnn.ROW_MAJOR_LAYOUT.
        * :attr:`memory_config`: the memory configuration of the output tensor. Default is ttnn.DRAM_MEMORY_CONFIG.

    Example::
        >>> device_id = 0
        >>> device = ttnn.open_device(device_id=device_id)
        >>> input_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]]), dtype=ttnn.uint32), device)
        >>> # an embedding matrix containing 10 tensors of size 4
        >>> weight = ttnn.to_device(ttnn.from_torch(torch.rand(10, 4), dtype=ttnn.bfloat16), device)
        >>> ttnn.embedding(input_tensor, weight)
        ttnn.Tensor([ [[1, 0.106445, 0.988281, 0.59375],
            [0.212891, 0.964844, 0.199219, 0.996094],
            [3.78362e-38, 0, 7.89785e-39, 0],
            [8.04479e-38, 0, 1.25815e-38, 0]],
           [[2.71833e-38, 0, 3.59995e-38, 0],
            [7.60398e-38, 0, 1.83671e-38, 0],
            [2.22242e-38, 0, 1.88263e-38, 0],
            [1.35917e-38, 0, 4.49994e-39, 0]]], dtype=bfloat16 )

    """

    if pad_token is not None:
        embeddings_type = ttl.tensor.EmbeddingsType.PADDED
    else:
        embeddings_type = ttl.tensor.EmbeddingsType.GENERIC

    *_, hidden_embedding_dim = weight.shape
    *_, padded_hidden_embedding_dim = weight.shape.with_tile_padding()
    weight = ttnn.unsqueeze_to_4D(weight)

    batch_size, sentence_size = input_tensor.shape
    input_tensor = ttnn.reshape(input_tensor, shape=(batch_size, 1, 1, sentence_size))

    tilized = layout == ttnn.TILE_LAYOUT
    embeddings = ttl.tensor.embeddings(
        input_tensor,
        weight,
        tilized,
        embeddings_type=embeddings_type,
        pad_token=pad_token,
        output_mem_config=memory_config,
    )
    embeddings = ttnn.reshape(embeddings, shape=(batch_size, sentence_size, hidden_embedding_dim))

    return embeddings


__all__ = []
