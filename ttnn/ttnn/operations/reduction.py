# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Optional


import tt_lib as ttl

import ttnn


def _create_golden_function(torch_function_name):
    import torch

    torch_function = getattr(torch, torch_function_name)

    def golden_function(input_tensor: ttnn.Tensor, dim: Optional[Union[int, Tuple[int]]] = None, keepdim=False, **_):
        if dim == None:
            return torch_function(input_tensor, keepdim=keepdim)
        else:
            return torch_function(input_tensor, dim=dim, keepdim=keepdim)

    return golden_function


def _validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


def reduce(
    input_tensor: ttnn.Tensor,
    reduction_op: str,
    dim: Optional[Union[int, Tuple[int]]],
    keepdim: bool = True,
    memory_config: Optional[ttnn.MemoryConfig] = None,
):
    if not keepdim:
        raise RuntimeError("keepdim=False is not supported")

    input_shape = tuple(input_tensor.shape)
    rank = len(input_shape)
    memory_config = memory_config or input_tensor.memory_config()

    original_dim = dim
    if isinstance(dim, int):
        if dim < 0:
            dim = rank + dim
        dim = (dim,)
    elif dim is None:
        dim = list(range(rank))

    output_shape = []
    padded_output_shape = []
    for axis, size in enumerate(input_shape):
        if axis in dim:
            if keepdim:
                output_shape.append(1)
                padded_output_shape.append(ttnn.TILE_SIZE if axis >= rank - 2 else 1)
        else:
            output_shape.append(size)
            padded_output_shape.append(size)
    output_shape = tuple(output_shape)
    padded_output_shape = tuple(padded_output_shape)

    input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

    if original_dim is None:
        if reduction_op == "mean":
            output_tensor = ttl.tensor.global_mean(input_tensor, output_mem_config=memory_config)
        elif reduction_op == "sum":
            output_tensor = ttl.tensor.global_sum(input_tensor, output_mem_config=memory_config)
        elif reduction_op == "max":
            output_tensor = ttl.tensor.global_max(input_tensor, output_mem_config=memory_config)
        elif reduction_op == "min":
            output_tensor = ttl.tensor.global_min(input_tensor, output_mem_config=memory_config)
        else:
            raise RuntimeError("Unsupported reduction operation")

    else:
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

        reduced_volume = 1
        for axis in dim:
            reduced_volume *= input_shape[axis]

        if reduction_op == "sum":
            output_tensor = ttl.tensor.reduce(
                input_tensor, ttl.tensor.ReduceOpMath.SUM, reduce_op_dim, 1.0, output_mem_config=memory_config
            )
        elif reduction_op == "mean":
            output_tensor = ttl.tensor.reduce(
                input_tensor,
                ttl.tensor.ReduceOpMath.SUM,
                reduce_op_dim,
                1 / reduced_volume,
                output_mem_config=memory_config,
            )
        elif reduction_op == "max":
            output_tensor = ttl.tensor.reduce(
                input_tensor, ttl.tensor.ReduceOpMath.MAX, reduce_op_dim, 1.0, output_mem_config=memory_config
            )
        elif reduction_op == "min":
            output_tensor = ttl.tensor.reduce(
                input_tensor, ttl.tensor.ReduceOpMath.MIN, reduce_op_dim, 1.0, output_mem_config=memory_config
            )
        elif reduction_op == "std":
            mean_tensor = ttl.tensor.reduce(
                input_tensor,
                ttl.tensor.ReduceOpMath.SUM,
                reduce_op_dim,
                1 / reduced_volume,
                output_mem_config=memory_config,
            )
            mean_square_tensor = ttl.tensor.reduce(
                ttl.tensor.pow(input_tensor, 2.0),
                ttl.tensor.ReduceOpMath.SUM,
                reduce_op_dim,
                1 / reduced_volume,
                output_mem_config=memory_config,
            )
            output_tensor = ttl.tensor.sqrt(
                ttl.tensor.sub(
                    mean_square_tensor,
                    ttl.tensor.pow(mean_tensor, 2.0, output_mem_config=memory_config),
                    output_mem_config=memory_config,
                ),
                output_mem_config=memory_config,
            )
        elif reduction_op == "var":
            mean_tensor = ttl.tensor.reduce(
                input_tensor,
                ttl.tensor.ReduceOpMath.SUM,
                reduce_op_dim,
                1 / reduced_volume,
                output_mem_config=memory_config,
            )
            mean_square_tensor = ttl.tensor.reduce(
                ttl.tensor.pow(input_tensor, 2.0),
                ttl.tensor.ReduceOpMath.SUM,
                reduce_op_dim,
                1 / reduced_volume,
                output_mem_config=memory_config,
            )
            output_tensor = ttl.tensor.sub(
                mean_square_tensor, ttl.tensor.pow(mean_tensor, 2.0), output_mem_config=memory_config
            )
        else:
            raise RuntimeError("Unsupported reduction operation")

    output_tensor = ttnn.reshape(output_tensor, ttnn.Shape(output_shape, padded_output_shape))

    return output_tensor


@ttnn.register_operation(
    name="ttnn.std",
    validate_input_tensors=_validate_input_tensors,
    golden_function=_create_golden_function("std"),
)
def std(
    input_tensor: ttnn.Tensor,
    dim: Optional[Union[int, Tuple[int]]],
    keepdim: bool = True,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """
    std(input_tensor: ttnn.Tensor, dim: Optional[Union[int, Tuple[int]]]) -> ttnn.Tensor
    """
    return reduce(input_tensor, "std", dim, keepdim, memory_config)


@ttnn.register_operation(
    name="ttnn.var",
    validate_input_tensors=_validate_input_tensors,
    golden_function=_create_golden_function("var"),
)
def var(
    input_tensor: ttnn.Tensor,
    dim: Optional[Union[int, Tuple[int]]],
    keepdim: bool = True,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """
    var(input_tensor: ttnn.Tensor, dim: Optional[Union[int, Tuple[int]]]) -> ttnn.Tensor
    """
    return reduce(input_tensor, "var", dim, keepdim, memory_config)


def _golden_function(input_tensor: ttnn.Tensor, dim: Union[int, None], keepdim=False, **_):
    import torch

    if dim == None:
        return torch.max(input_tensor)
    else:
        return torch.max(input_tensor, dim=dim, keepdim=keepdim)


@ttnn.register_operation(
    name="ttnn.max",
    validate_input_tensors=_validate_input_tensors,
    golden_function=_create_golden_function("max"),
)
def max(
    input_tensor: ttnn.Tensor,
    dim: Optional[Union[int, Tuple[int]]] = None,
    keepdim: bool = True,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """
    max(input_tensor: ttnn.Tensor, dim: Optional[Union[int, Tuple[int]]]) -> ttnn.Tensor
    """
    return reduce(input_tensor, "max", dim, keepdim, memory_config)


@ttnn.register_operation(
    name="ttnn.min",
    validate_input_tensors=_validate_input_tensors,
    golden_function=_create_golden_function("min"),
)
def min(
    input_tensor: ttnn.Tensor,
    dim: Optional[Union[int, Tuple[int]]] = None,
    keepdim: bool = True,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """
    min(input_tensor: ttnn.Tensor, dim: Optional[Union[int, Tuple[int]]]) -> ttnn.Tensor
    """
    return reduce(input_tensor, "min", dim, keepdim, memory_config)


@ttnn.register_operation(
    name="ttnn.sum",
    validate_input_tensors=_validate_input_tensors,
    golden_function=_create_golden_function("sum"),
)
def sum(
    input_tensor: ttnn.Tensor,
    dim: Optional[Union[int, Tuple[int]]] = None,
    keepdim: bool = True,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """
    sum(input_tensor: ttnn.Tensor, dim: Optional[Union[int, Tuple[int]]]) -> ttnn.Tensor
    """
    return reduce(input_tensor, "sum", dim, keepdim, memory_config)


def _golden_function(input_tensor: ttnn.Tensor, dim: int, keepdim=False, **_):
    import torch

    return torch.mean(input_tensor, dim=dim, keepdim=keepdim)


@ttnn.register_operation(
    name="ttnn.mean",
    validate_input_tensors=_validate_input_tensors,
    golden_function=_create_golden_function("mean"),
)
def mean(
    input_tensor: ttnn.Tensor,
    dim: Optional[Union[int, Tuple[int]]] = None,
    keepdim: bool = True,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """
    mean(input_tensor: ttnn.Tensor, dim: Optional[Union[int, Tuple[int]]] = None, keepdim: bool = True, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor
    """
    return reduce(input_tensor, "mean", dim, keepdim, memory_config)


__all__ = []
