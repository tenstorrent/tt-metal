# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, List

import tt_lib as ttl

import ttnn
import ttnn.decorators


def _preprocess_golden_function_inputs(args, kwargs):
    input_tensor, args, kwargs = ttnn.reflection.pop_argument("input_tensor", args, kwargs)
    padding, args, kwargs = ttnn.reflection.pop_argument("padding", args, kwargs)

    if len(padding) != len(input_tensor.shape):
        raise RuntimeError("ttnn.pad: padding must be the same length as the input tensor rank")

    for start, end in padding:
        if start < 0 or end < 0:
            raise RuntimeError("ttnn.pad: padding must be non-negative")

    pad_start = tuple(start for start, _ in padding)
    *_, pad_start_height, pad_start_width = pad_start
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        if pad_start_height % ttnn.TILE_SIZE != 0 or pad_start_width % ttnn.TILE_SIZE != 0:
            raise RuntimeError(
                "ttnn.pad: padding end must be a multiple of the tile size on height and width for a tensor in tile layout"
            )

    pad_end = tuple(end for _, end in padding)
    *_, pad_end_height, pad_end_width = pad_end
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        if pad_end_height % ttnn.TILE_SIZE != 0 or pad_end_width % ttnn.TILE_SIZE != 0:
            raise RuntimeError(
                "ttnn.pad: padding end must be a multiple of the tile size on height and width for a tensor in tile layout"
            )

    input_tensor = ttnn.to_torch(input_tensor)

    return (input_tensor, padding, *args), kwargs


def _golden_function(input_tensor: ttnn.Tensor, padding, value):
    import torch

    torch_padding = []
    for dimension in reversed(padding):
        torch_padding.append(dimension[0])
        torch_padding.append(dimension[1])
    return torch.nn.functional.pad(input_tensor, pad=torch_padding, mode="constant", value=value)


def _postprocess_golden_function_outputs(output_tensor, args, kwargs):
    output_tensor = ttnn.decorators.default_postprocess_golden_function_outputs(output_tensor, args, kwargs)
    # Padding always turns the intended shape to the shape with tile padding. For simplicity of the operation
    output_tensor = ttnn.reshape(output_tensor, shape=output_tensor.shape.with_tile_padding())
    return output_tensor


def _pad_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.int32, ttnn.uint32),
        layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.pad",
    validate_input_tensors=_pad_validate_input_tensors,
    golden_function=_golden_function,
    preprocess_golden_function_inputs=_preprocess_golden_function_inputs,
    postprocess_golden_function_outputs=_postprocess_golden_function_outputs,
    allow_to_fallback_to_golden_function_on_failure=True,
)
def pad(
    input_tensor: ttnn.Tensor,
    padding: Tuple[Tuple[int, int], ...],
    value: Union[int, float],
    *,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""

    pad(input_tensor: ttnn.Tensor, padding: Tuple[Tuple[int, int], ...], value: Union[int, float], *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Pad tensor with constant value.

    Padded shape is accumulated if ttnn.pad is called on a tensor with padding.

    Args:
        * :attr:`input_tensor`: input tensor
        * :attr:`padding`: padding to apply. Each element of padding should be a tuple of 2 integers, with the first integer specifying the number of values to add before the tensor and the second integer specifying the number of values to add after the tensor.
        * :attr:`value`: value to pad with
        * :attr:`memory_config`: the memory configuration to use for the operation

    """
    if input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
        raise RuntimeError(
            "ttnn.pad: row-major tensors have to use fallback because the kernel currently causes a PCC error"
        )

    original_rank = len(input_tensor.shape)
    if len(padding) != original_rank:
        raise RuntimeError("ttnn.pad: padding must be the same length as the input tensor rank")

    for start, end in padding:
        if start < 0 or end < 0:
            raise RuntimeError("ttnn.pad: padding must be non-negative")

    if original_rank < 4:
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
        padding = tuple((0, 0) for _ in range(4 - original_rank)) + padding

    input_shape_with_tile_padding = input_tensor.shape.with_tile_padding()

    pad_start = tuple(start for start, _ in padding)
    if sum(pad_start) != 0:
        raise RuntimeError("ttnn.pad: padding start must be 0 currently")

    pad_end = tuple(end for _, end in padding)
    *_, pad_end_height, pad_end_width = pad_end
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        if pad_end_height % ttnn.TILE_SIZE != 0 or pad_end_width % ttnn.TILE_SIZE != 0:
            raise RuntimeError(
                "ttnn.pad: padding end must be a multiple of the tile size on height and width for a tensor in tile layout"
            )

    padded_shape = tuple(dim + end for dim, end in zip(input_shape_with_tile_padding, pad_end))

    output_tensor = ttl.tensor.pad(
        input_tensor, padded_shape, pad_start, value, output_mem_config=memory_config, use_multicore=True
    )

    while len(output_tensor.shape) > original_rank:
        output_tensor = ttnn.squeeze(output_tensor, dim=0)

    # Padding always turn the intended shape to the shape with tile padding. For simplicity of the operation
    output_tensor = ttnn.reshape(output_tensor, shape=output_tensor.shape.with_tile_padding())

    return output_tensor


def _golden_function(input_tensor: ttnn.Tensor, order: Tuple[int, ...], **_):
    if len(input_tensor.shape) != len(order):
        raise RuntimeError(
            "The number of dimensions in the tensor input does not match the length of the desired ordering"
        )

    return input_tensor.permute(order).contiguous().clone()


def _permute_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(1, 2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.int32, ttnn.uint32),
        layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


doc = r"""
permute(input_tensor: ttnn.Tensor, order: Tuple[int, ...]) -> ttnn.Tensor

Permutes :attr:`input_tensor` using :attr:`order`.

Args:
    * :attr:`input_tensor`: the input tensor
    * :attr:`order`: the desired ordering of dimensions.

Example::

    >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
    >>> output = ttnn.permute(tensor, (0, 1, 3, 2))
    >>> print(output.shape)
    [1, 1, 32, 64]

"""
permute = ttnn.register_operation(
    name="ttnn.permute",
    validate_input_tensors=_permute_validate_input_tensors,
    golden_function=_golden_function,
    doc=doc,
)(ttnn._ttnn.operations.data_movement.permute)


def _golden_function(tensors, dim=0, **_):
    import torch

    return torch.concat(tensors, dim)


def _concat_validate_input_tensors(operation_name, tensors, dim, *args, **kwargs):
    for input_tensor in tensors:
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.int32, ttnn.uint32),
            layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )


doc = r"""
concat(tensors: List[ttnn.Tensor], dim: int = 0, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

Concats :attr:`tensors` in the given :attr:`dim`.

Args:
    * :attr:`tensors`: the tensors to be concatenated.
    * :attr:`dim`: the concatenating dimension.

Keyword Args:
    * :attr:`memory_config`: the memory configuration to use for the operation

Example::

    >>> tensor = ttnn.concat(ttnn.from_torch(torch.zeros((1, 1, 64, 32), ttnn.from_torch(torch.zeros((1, 1, 64, 32), dim=3)), device)

    >>> tensor1 = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
    >>> tensor2 = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
    >>> output = ttnn.concat([tensor1, tensor2], dim=4)
    >>> print(output.shape)
    [1, 1, 32, 64]

"""
concat = ttnn.register_operation(
    name="ttnn.concat",
    validate_input_tensors=_concat_validate_input_tensors,
    golden_function=_golden_function,
    doc=doc,
)(ttnn._ttnn.operations.data_movement.concat)


def _golden_function(input_tensor, split_size, dim):
    import torch

    return torch.split(input_tensor, split_size, dim=dim)


def _split_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.int32, ttnn.uint32),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.split",
    validate_input_tensors=_split_validate_input_tensors,
    golden_function=_golden_function,
    allow_to_fallback_to_golden_function_on_failure=True,
)
def split(input_tensor: ttnn.Tensor, split_size: int, dim: int) -> ttnn.Tensor:
    r"""
    split(input_tensor: ttnn.Tensor, split_size: int, dim: int) -> Tuple[ttnn.Tensor, ...]

    Split tensor into chunks of :attr:`split_size` along :attr:`dim`.

    Args:
        * :attr:`input_tensor`: input tensor.
        * :attr:`split_size`: size of a single chunk.
        * :attr:`dim`:  dimension along which to split the tensor.
    """
    raise NotImplementedError


def _golden_function(tensor, repeats, dim=0, **_):
    import torch

    return torch.repeat_interleave(tensor, repeats, dim=dim)


def _repeat_interleave_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.int32, ttnn.uint32),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=True,
    )


@ttnn.register_operation(
    name="ttnn.repeat_interleave",
    validate_input_tensors=_repeat_interleave_validate_input_tensors,
    golden_function=_golden_function,
    allow_to_fallback_to_golden_function_on_failure=True,
)
def repeat_interleave(input_tensor: ttnn.Tensor, repeats: Union[ttnn.Tensor, int], dim: int = 0) -> ttnn.Tensor:
    r"""
    repeat_interleave(input_tensor: ttnn.Tensor, repeats : Union[ttnn.Tensor,int], dim: int = 0) -> ttnn.Tensor

    Repeats elements of a :attr:`tensor` in the given :attr:`dim`.

    Args:
        * :attr:`input_tensor`: the input_tensor to apply the repeate interleave operation.
        * :attr:`repeats`: The number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis.
        * :attr:`dim`: the dimension to expand with the repetitions.

    Example::

        >>> a = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]]), device=device, layout=ttnn.TILE_LAYOUT)
        >>> b = ttnn.repeat_interleave(a, 2, dim=0)
        >>> print(a.shape, b.shape)
        ttnn.Shape([2[32], 2[32]]) ttnn.Shape([4[32], 2[32]])

    """

    if not isinstance(repeats, int) and not isinstance(repeats, ttnn.Tensor):
        raise RuntimeError("ttnn: Expected repeat to either be an int or a ttnn.Tensor")

    rank_of_tensor = len(input_tensor.shape)
    if dim >= rank_of_tensor:
        dimension_range = f"[{-rank_of_tensor}, {rank_of_tensor - 1}]"
        raise RuntimeError(
            f"ttnn: Dimension out of range (expected to be in range of {dimension_range}, but got {dim})"
        )

    def custom_numel(tensor):
        total_elements = 1
        for dimension in tensor.shape:
            total_elements *= dimension
        return total_elements

    if isinstance(repeats, ttnn.Tensor):
        if input_tensor.shape[dim] != custom_numel(repeats):
            raise RuntimeError("ttnn: repeats must have the same size as input along dim")
        elif len(repeats.shape) != 1:
            raise RuntimeError("ttnn: repeats must be 0-dim or 1-dim tensor")

    dtype = input_tensor.dtype
    rank = len(input_tensor.shape)
    if dtype == ttnn.bfloat16 and rank == 4 and dim != 2 and dim != 3:
        output_tensor = ttl.tensor.repeat_interleave(input_tensor, repeats, dim=dim)
        *batch, _, _ = output_tensor.shape
        *_, h, w = input_tensor.shape
        *_, padded_h, padded_w = input_tensor.shape.with_tile_padding()
        if dim == 2:
            *_, h, _ = output_tensor.shape
            *_, padded_h, _ = output_tensor.shape.with_tile_padding()
        elif dim == 3:
            *_, _, w = output_tensor.shape
            *_, _, padded_w = output_tensor.shape.with_tile_padding()
        output_tensor = ttnn.reshape(output_tensor, shape=ttnn.Shape(batch + [h, w], batch + [padded_h, padded_w]))
        return output_tensor
    else:
        raise NotImplementedError


def _golden_function(tensor, shape, **_):
    return tensor.repeat(shape[0], shape[1], shape[2], shape[3])


def _repeat_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.int32, ttnn.uint32),
        layouts=(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=True,
    )


@ttnn.register_operation(
    name="ttnn.repeat",
    validate_input_tensors=_repeat_validate_input_tensors,
    golden_function=_golden_function,
    allow_to_fallback_to_golden_function_on_failure=True,
)
def repeat(
    input_tensor: ttnn.Tensor,
    shape: ttnn.Shape,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""
    repeat(input_tensor: ttnn.Tensor, shape : ttnn.Shape) -> ttnn.Tensor

    Returns a new tensor filled with repetition of input :attr:`input_tensor` according to number of times specified in :attr:`shape`.

    Args:
        * :attr:`input_tensor`: the input_tensor to apply the repeate operation.
        * :attr:`shape`: The number of repetitions for each element.

    Example::

        >>> tensor = ttnn.repeat(ttnn.from_torch(torch.tensor([[1, 2], [3, 4]]), 2,)), device)
        >>> print(tensor)
        tensor([[1, 2],
        [1, 2],
        [3, 4],
        [3, 4]])

    """

    if not isinstance(shape, ttnn.Shape):
        raise RuntimeError("ttnn: Expected shape to be a ttnn.Shape")

    rank = len(input_tensor.shape)
    if rank == 4:
        output_tensor = ttl.tensor.repeat(input_tensor, shape, output_mem_config=memory_config)
        *batch, _, _ = output_tensor.shape
        *_, h, w = output_tensor.shape
        *_, padded_h, padded_w = output_tensor.shape.with_tile_padding()

        output_tensor = ttnn.reshape(output_tensor, shape=ttnn.Shape(batch + [h, w], batch + [padded_h, padded_w]))
        return output_tensor
    else:
        raise NotImplementedError


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


def _golden_function(input_tensor: ttnn.Tensor, scale_factor: Tuple[float, float], **_):
    import torch

    input_tensor = input_tensor.permute(0, 3, 1, 2)
    ret = torch.nn.functional.upsample(input_tensor, scale_factor=scale_factor)
    ret = ret.permute(0, 2, 3, 1)
    return ret


@ttnn.register_operation(
    name="ttnn.upsample",
    validate_input_tensors=_upsample_validate_input_tensors,
    golden_function=_golden_function,
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
