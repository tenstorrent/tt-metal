# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, List

import tt_lib as ttl

import ttnn


def _torch_pad(input_tensor: ttnn.Tensor, padding, value):
    import torch

    input_tensor = ttnn.to_torch(input_tensor)
    torch_padding = []
    for dimension in reversed(padding):
        torch_padding.append(dimension[0])
        torch_padding.append(dimension[1])
    return torch.nn.functional.pad(input_tensor, pad=torch_padding, mode="constant", value=value)


def _fallback_pad(input_tensor: ttnn.Tensor, padding, value, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
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

    output_tensor = _torch_pad(input_tensor, padding, value)
    output_tensor = ttnn.from_torch(
        output_tensor,
        dtype=input_tensor.dtype,
        device=input_tensor.device(),
        layout=input_tensor.layout,
        memory_config=memory_config,
    )

    # Padding always turns the intended shape to the shape with tile padding. For simplicity of the operation
    output_tensor = ttnn.reshape(output_tensor, shape=output_tensor.shape.with_tile_padding())
    return output_tensor


def _pad_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32),
        layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.pad",
    validate_input_tensors=_pad_validate_input_tensors,
    torch_function=_torch_pad,
    fallback=_fallback_pad,
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


def _torch_permute(input_tensor: ttnn.Tensor, order: Tuple[int, ...], **_):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    return torch.permute(input_tensor, order).contiguous().clone()


def _permute_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(1, 2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32),
        layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.permute",
    validate_input_tensors=_permute_validate_input_tensors,
    torch_function=_torch_permute,
    # TODO(arakhmati): add proper fallback
)
def permute(input_tensor: ttnn.Tensor, order: Tuple[int, ...]) -> ttnn.Tensor:
    r"""
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
    if not isinstance(order, tuple):
        raise RuntimeError("order must be a tuple")

    if len(input_tensor.shape) != len(order):
        raise RuntimeError(
            "The number of dimensions in the tensor input does not match the length of the desired ordering"
        )

    on_device = ttnn.is_tensor_storage_on_device(input_tensor)
    device = input_tensor.device()
    layout = input_tensor.layout
    dtype = input_tensor.dtype
    rank = len(input_tensor.shape)

    if len(input_tensor.shape) < 4:
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
        adjusted_order_for_4D_tensor = order
        while len(adjusted_order_for_4D_tensor) < 4:
            adjusted_order_for_4D_tensor = (0,) + tuple(x + 1 for x in adjusted_order_for_4D_tensor)
        order = adjusted_order_for_4D_tensor

    if ttnn.has_tile_padding(input_tensor):
        input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)

    if ttnn.is_tensor_storage_on_device(input_tensor) and len(input_tensor.shape) == 4:
        output_tensor = ttl.tensor.permute(input_tensor, order)
        output_tensor = ttnn.to_layout(output_tensor, layout)
        rank_should_be_updated = len(output_tensor.shape) > rank
        while rank_should_be_updated:
            prior_rank = len(output_tensor.shape)
            output_tensor = ttnn.squeeze(output_tensor, dim=0)
            rank_should_be_updated = prior_rank != len(output_tensor.shape) and len(output_tensor.shape) > rank

        if on_device and not ttnn.is_tensor_storage_on_device(output_tensor):
            output_tensor = ttnn.to_device(output_tensor, device)
        return output_tensor
    else:

        def torch_permute(tensor, order):
            return tensor.permute(order).contiguous().clone()

        tensor = ttnn.to_torch(input_tensor)
        tensor = ttl.tensor.decorate_external_operation(torch_permute, function_name="torch.permute")(tensor, order)
        tensor = ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device)
        return tensor


def _torch_concat(tensors, dim=0, **_):
    import torch

    torch_tensors = [ttnn.to_torch(tensor) for tensor in tensors]
    return torch.concat(torch_tensors, dim)


def _fallback_concat(tensors, dim=0, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    dtype = tensors[0].dtype
    device = tensors[0].device()
    layout = tensors[0].layout
    output_tensor = _torch_concat(tensors, dim=dim)
    return ttnn.from_torch(output_tensor, dtype=dtype, device=device, layout=layout, memory_config=memory_config)


def _concat_validate_input_tensors(operation_name, tensors, dim, *args, **kwargs):
    for input_tensor in tensors:
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32),
            layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )


@ttnn.register_operation(
    name="ttnn.concat",
    validate_input_tensors=_concat_validate_input_tensors,
    torch_function=_torch_concat,
    fallback=_fallback_concat,
)
def concat(
    tensors: List[ttnn.Tensor],
    dim: int = 0,
    *,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""
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
    if len(tensors) < 1:
        raise RuntimeError("ttnn.concat: expected a non-empty list of Tensors!")

    if len(tensors) == 1:
        return ttnn.to_memory_config(tensors[0], memory_config)

    first_tensor = tensors[0]
    first_tensor_shape = first_tensor.shape
    for tensor in tensors:
        shape = tensor.shape
        if (
            len(shape) != len(first_tensor_shape)
            or any(shape[i] != first_tensor_shape[i] for i in range(len(shape)) if i != dim)
            or any(
                shape.with_tile_padding()[i] != first_tensor_shape.with_tile_padding()[i]
                for i in range(len(shape))
                if i != dim
            )
        ):
            raise ValueError(
                "All dimensions must be the same size except for the dimension along which the contenation is taking place."
            )

    rank = len(tensors[0].shape)
    original_dim = dim
    if dim < 0:
        dim = rank + dim
    if dim < 0 or dim >= rank:
        raise RuntimeError(
            f"ttnn: Dimension out of range: dim {original_dim} cannot be used for tensors of rank {rank}"
        )

    rank = len(tensors[0].shape)

    all_tensors_are_tile_layout_without_padding = all(
        tensor.layout == ttnn.TILE_LAYOUT and not ttnn.has_tile_padding(tensor) for tensor in tensors
    )

    if rank <= 4 and all_tensors_are_tile_layout_without_padding:
        tensors_4d = [ttnn.unsqueeze_to_4D(tensor) for tensor in tensors]
        dim = dim + 4 - rank
        output_tensor = ttl.tensor.concat(tensors_4d, dim=dim, output_mem_config=memory_config)
        while len(output_tensor.shape) > rank:
            output_tensor = ttnn.squeeze(output_tensor, dim=0)
        return output_tensor
    else:
        raise NotImplementedError


def _torch_split(input_tensor: ttnn.Tensor, split_size, dim):
    import torch

    input_tensor = ttnn.to_torch(input_tensor)
    return torch.split(input_tensor, split_size, dim=dim)


def _split_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.split",
    validate_input_tensors=_split_validate_input_tensors,
    torch_function=_torch_split,
    # TODO(arakhmati): add proper fallback
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

    output_tensors = _torch_split(input_tensor, split_size, dim)
    output_tensors = tuple(
        ttnn.from_torch(
            output_tensor, device=input_tensor.device(), dtype=input_tensor.dtype, layout=input_tensor.layout
        )
        for output_tensor in output_tensors
    )
    return output_tensors


def _torch_repeat_interleave(tensor, repeats, dim=0, **_):
    import torch

    if isinstance(repeats, ttnn.Tensor):
        repeats = ttnn.to_torch(repeats)
    return torch.repeat_interleave(ttnn.to_torch(tensor), repeats, dim=dim)


def _repeat_interleave_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=True,
    )


@ttnn.register_operation(
    name="ttnn.repeat_interleave",
    validate_input_tensors=_repeat_interleave_validate_input_tensors,
    torch_function=_torch_repeat_interleave,
    # TODO(arakhmati): add proper fallback
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
    device = input_tensor.device()
    layout = input_tensor.layout
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

        def torch_repeat_interleave(tensor, repeats, dim=dim):
            return _torch_repeat_interleave(tensor, repeats, dim)

        output_tensor = ttl.tensor.decorate_external_operation(
            torch_repeat_interleave, function_name="torch_repeat_interleave"
        )(input_tensor, repeats, dim=dim)
        return ttnn.from_torch(output_tensor, device=device, dtype=dtype, layout=layout)


def _torch_repeat(tensor, shape, **_):
    return ttnn.to_torch(tensor).repeat(shape[0], shape[1], shape[2], shape[3])


def _repeat_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32),
        layouts=(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=True,
    )


@ttnn.register_operation(
    name="ttnn.repeat",
    validate_input_tensors=_repeat_validate_input_tensors,
    torch_function=_torch_repeat,
    # TODO(arakhmati): add proper fallback
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

    dtype = input_tensor.dtype
    device = input_tensor.device()
    layout = input_tensor.layout
    rank = len(input_tensor.shape)
    if rank == 4:
        output_tensor = ttl.tensor.repeat(input_tensor, shape)
        *batch, _, _ = output_tensor.shape
        *_, h, w = output_tensor.shape
        *_, padded_h, padded_w = output_tensor.shape.with_tile_padding()

        output_tensor = ttnn.reshape(output_tensor, shape=ttnn.Shape(batch + [h, w], batch + [padded_h, padded_w]))
        return output_tensor
    else:

        def torch_repeat(tensor, shape):
            return _torch_repeat(tensor, shape)

        output_tensor = ttl.tensor.decorate_external_operation(torch_repeat, function_name="torch_repeat")(
            input_tensor, shape
        )
        return ttnn.from_torch(output_tensor, device=device, dtype=dtype, layout=layout)


__all__ = []
