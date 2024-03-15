# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Optional
import os

import tt_lib as ttl

import ttnn


def _torch_embedding(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, **_):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_torch(input_tensor)

    weight = ttnn.from_device(weight)
    weight = ttnn.to_torch(weight)

    output_tensor = torch.nn.functional.embedding(input_tensor, weight)

    return output_tensor


def _embedding_validate_input_tensors(operation_name, input_tensor, weight, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.uint32,),
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
    torch_function=_torch_embedding,
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


def _torch_softmax(input_tensor: ttnn.Tensor, dim: int, **_):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    return torch.softmax(input_tensor, dim)


def _softmax_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.softmax",
    validate_input_tensors=_softmax_validate_input_tensors,
    torch_function=_torch_softmax,
)
def softmax(
    input_tensor: ttnn.Tensor, dim: int, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
) -> ttnn.Tensor:
    r"""
    softmax(input_tensor: ttnn.Tensor, dim: int, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Compute softmax over :attr:`input_tensor` along :attr:`dim`.

    Args:
        * :attr:`input_tensor`: the input tensor
        * :attr:`dim`: the dimension along which to compute softmax.

    Example::

        >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
        >>> output = ttnn.softmax(tensor, -1)
        >>> print(output[0, 0, 0, :3])
        ttnn.Tensor([ 0.0310059, 0.0310059, 0.0310059], dtype=bfloat16 )

    """

    input_shape = input_tensor.shape
    rank = len(input_shape)
    if dim < 0:
        dim = rank + dim

    input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

    is_tile_padded = list(input_tensor.shape)[-2:] != list(input_tensor.shape.with_tile_padding())[-2:]
    if not is_tile_padded and dim == rank - 1:
        # TODO: #4599 Research why softmax appears to not be stable when using a padded ttnn.TILE_LAYOUT
        output_tensor = ttl.tensor.softmax(input_tensor, output_mem_config=memory_config)
    else:
        dim_4D = dim + 4 - rank
        output_tensor = ttl.operations.primary.moreh_softmax(input_tensor, dim=dim_4D)
    output_tensor = ttnn.reshape(output_tensor, input_shape)
    return output_tensor


def _torch_mean(input_tensor: ttnn.Tensor, dim: int, keepdim=False, **_):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    return torch.mean(input_tensor, dim=dim, keepdim=keepdim)


def _mean_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.mean",
    validate_input_tensors=_mean_validate_input_tensors,
    torch_function=_torch_mean,
)
def mean(input_tensor: ttnn.Tensor, dim: Union[int, Tuple[int]], keepdim: bool = False) -> ttnn.Tensor:
    """
    mean(input_tensor: ttnn.Tensor, dim: Union[int, Tuple[int]], keepdim: bool = False) -> ttnn.Tensor
    """

    input_shape = tuple(input_tensor.shape)
    rank = len(input_shape)

    if isinstance(dim, int):
        if dim < 0:
            dim = rank + dim
        dim = (dim,)

    if isinstance(dim, tuple):
        if dim == (rank - 1,):
            reduce_op_dim = ttl.tensor.ReduceOpDim.W
        elif dim == (rank - 2,):
            reduce_op_dim = ttl.tensor.ReduceOpDim.H
        elif dim == (rank - 1, rank - 2):
            reduce_op_dim = ttl.tensor.ReduceOpDim.HW
        else:
            raise RuntimeError("Unsupported dim")
    else:
        raise RuntimeError("Invalid dim")

    output_shape = []
    padded_output_shape = []
    for axis, size in enumerate(input_shape):
        if axis in dim:
            if keepdim:
                output_shape.append(1)
                padded_output_shape.append(ttnn.TILE_SIZE)
        else:
            output_shape.append(size)
            padded_output_shape.append(size)
    output_shape = tuple(output_shape)
    padded_output_shape = tuple(padded_output_shape)

    input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
    output_tensor = ttl.tensor.reduce(input_tensor, ttl.tensor.ReduceOpMath.SUM, reduce_op_dim, 1 / input_shape[-1])
    output_tensor = ttl.tensor.reduce(input_tensor, ttl.tensor.ReduceOpMath.SUM, reduce_op_dim, 1 / input_shape[-1])
    output_tensor = ttnn.reshape(output_tensor, ttnn.Shape(output_shape, padded_output_shape))
    return output_tensor


## helper function for upsample. currently only supports HEIGHT sharding
def _get_upsample_shard_grid_from_num_shards(ncores: int, device):
    max_grid_size = (8, 8) if device.arch() == ttl.device.Arch.WORMHOLE_B0 else (9, 12)  ## (y, x)
    if ncores % max_grid_size[1] == 0:
        core_grid = ttnn.CoreGrid(y=ncores // max_grid_size[1], x=max_grid_size[1])
        grid_coord = ttnn.experimental.tensor.CoreCoord(core_grid.x - 1, core_grid.y - 1)
        return ttnn.experimental.tensor.CoreRangeSet(
            {ttnn.experimental.tensor.CoreRange(ttnn.experimental.tensor.CoreCoord(0, 0), grid_coord)}
        )
    else:
        if ncores < max_grid_size[1]:
            core_grid = ttnn.CoreGrid(y=1, x=ncores)
            grid_coord = ttnn.experimental.tensor.CoreCoord(core_grid.x - 1, 0)
            return ttnn.experimental.tensor.CoreRangeSet(
                {ttnn.experimental.tensor.CoreRange(ttnn.experimental.tensor.CoreCoord(0, 0), grid_coord)}
            )
        else:
            core_grid_1 = ttnn.CoreGrid(y=ncores // max_grid_size[1], x=max_grid_size[1])
            core_grid_2 = ttnn.CoreGrid(y=ncores // max_grid_size[1] + 1, x=ncores % max_grid_size[1])
            grid_coord_1 = ttnn.experimental.tensor.CoreCoord(core_grid_1.x - 1, core_grid_1.y - 1)
            grid_coord_2 = ttnn.experimental.tensor.CoreCoord(core_grid_2.x - 1, core_grid_2.y - 1)
            return ttnn.experimental.tensor.CoreRangeSet(
                {
                    ttnn.experimental.tensor.CoreRange(ttnn.experimental.tensor.CoreCoord(0, 0), grid_coord_1),
                    ttnn.experimental.tensor.CoreRange(
                        ttnn.experimental.tensor.CoreCoord(0, grid_coord_2.y), grid_coord_2
                    ),
                }
            )


## helper function for upsample
def _get_upsample_num_shards(
    batch_size: int, height: int, num_channels: int, shard_strategy: ttnn.ShardStrategy, device
):
    ## calculate ncores, corresponding grid_size and in_shard_shape based on the input_shape
    max_grid_size = (8, 8) if device.arch() == ttl.device.Arch.WORMHOLE_B0 else (9, 12)  ## (y, x)
    if shard_strategy == ttnn.ShardStrategy.HEIGHT:
        ## nsticks per shard should be divisible by in_w
        max_nshards = min(batch_size * height, max_grid_size[0] * max_grid_size[1])
        nshards = max_nshards
        while nshards > 0:
            if batch_size * height % nshards == 0:
                break
            nshards -= 1
        if nshards == 0:
            ## should fallback to single core (TODO)
            raise ValueError("nshards is 0")

    ## TODO: support BLOCK strategy?
    # elif shard_strategy == ttnn.ShardStrategy.BLOCK:
    #     max_nshards_h = min(batch_size * height, max_grid_size[0])  ## height along NHW
    #     max_nshards_w = min(num_channels, max_grid_size[1])  ## width along C
    #     ## find nshards_h along NHW
    #     nshards_h = max_nshards_h
    #     while nshards_h > 0:
    #         if batch_size * height % nshards_h == 0:
    #             break
    #         nshards_h -= 1
    #     ## find nshards_w along C
    #     nshards_w = max_nshards_w
    #     while nshards_w > 0:
    #         if num_channels % nshards_w == 0:
    #             break
    #         nshards_w -= 1
    #     if nshards_w == 0 or nshards_h == 0:
    #         ## should fallback to single core (TODO)
    #         raise ValueError("nshards_h or nshards_w is 0")
    #     return (nshards_h, nshards_w)

    return nshards


def _upsample_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.ROW_MAJOR_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


def _torch_upsample(input_tensor: ttnn.Tensor, scale_factor: [float, float], **_):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    return torch.nn.functional.upsample(input_tensor, scale_factor=scale_factor)


@ttnn.register_operation(
    name="ttnn.upsample",
    validate_input_tensors=_upsample_validate_input_tensors,
    torch_function=_torch_upsample,
)
def upsample(
    input_tensor: ttnn.Tensor,
    scale_factor: Union[float, Tuple[float, float], Tuple[float, float, float], Tuple[float, float, float, float]],
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""
    Upsamples a given multi-channel 2D (spatial) data.
    The input data is assumed to be of the form [N, H, W, C].

    The algorithms available for upsampling are 'nearest' for now.

    Args:
        * :attr:`input_tensor`: the input tensor
        * :attr:`scale_factor`: multiplier for spatial size. Has to match input size if it is a tuple.
    """

    scale_h, scale_w = 1, 1
    if isinstance(scale_factor, float) or isinstance(scale_factor, int):
        scale_h = scale_factor
        scale_w = scale_factor
    elif isinstance(scale_factor, tuple):
        if len(scale_factor) == 2:
            scale_w, scale_c = scale_factor
            assert scale_c == 1, "scale_c should be 1"
        elif len(scale_factor) == 3:
            scale_h, scale_w, scale_c = scale_factor
            assert scale_c == 1, "scale_c should be 1"
        elif len(scale_factor) == 4:
            scale_n, scale_h, scale_w, scale_c = scale_factor
            assert scale_n == 1, "scale_n should be 1"
            assert scale_c == 1, "scale_c should be 1"
        else:
            RuntimeError("Invalid scale factor")

    ## check if the incoming sharding spec is compatible with the upsample operation
    ## if the sharding spec is not compatible, then the input tensor needs to be resharded
    if ttnn.is_sharded(input_tensor):
        if memory_config == ttnn.DRAM_MEMORY_CONFIG:
            ## input is sharded, make the output config sharded too if not provided
            memory_config = ttnn.get_memory_config(input_tensor)

        ## check if shard height % input_w == 0
        input_mem_config = ttnn.get_memory_config(input_tensor)
        shard_shape = input_mem_config.shard_spec.shape
        batch_size, input_h, input_w, num_channels = input_tensor.shape
        if shard_shape[0] % input_w != 0:
            ## perform a resharding operation:
            ## calculate ideal shard_grid
            nshards = _get_upsample_num_shards(
                batch_size, input_h, num_channels, ttnn.ShardStrategy.HEIGHT, input_tensor.device()
            )
            shard_grid = _get_upsample_shard_grid_from_num_shards(nshards, input_tensor.device())

            print(f"Resharding input tensor to {nshards} shards with HEIGHT sharding")

            ## sharded to interleaved
            input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

            ## construct new shard_spec
            shard_width = num_channels
            shard_height = batch_size * input_h * input_w // nshards
            shard_spec = ttnn.experimental.tensor.ShardSpec(
                shard_grid, (shard_height, shard_width), ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR, False
            )
            sharded_mem_config = ttnn.MemoryConfig(
                ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
            )

            ## interleaved to sharded
            input_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_mem_config)

            ## also update the output memory_config
            shard_height = shard_height * scale_h * scale_w
            shard_spec = ttnn.experimental.tensor.ShardSpec(
                shard_grid, (shard_height, shard_width), ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR, False
            )
            memory_config = ttnn.MemoryConfig(
                ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
            )

    output_tensor = ttl.tensor.upsample(input_tensor, int(scale_h), int(scale_w), output_mem_config=memory_config)
    return output_tensor


__all__ = []
