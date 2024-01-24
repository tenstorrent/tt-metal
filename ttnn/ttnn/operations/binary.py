# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Union

import math
import sys

import tt_lib as ttl

import ttnn.core as ttnn
from ttnn.decorators import decorate_operation


THIS_MODULE = sys.modules[__name__]

__all__ = []


def register_ttl_binary_function(name, ttl_binary_function, doc):
    def _torch_unary(input_tensor: ttnn.Tensor, parameter, **_):
        import torch

        name_to_torch_function = {"pow": torch.pow}
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)
        return torch_function(input_tensor, parameter)

    @decorate_operation(torch_function=_torch_unary, name=name)
    def binary_function(
        input_tensor: ttnn.Tensor, parameter: float, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        original_shape = input_tensor.shape
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
        ttl_input_tensor = input_tensor.value

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE):
            raise RuntimeError("input_tensor must be on device!")
        ttl_input_tensor = input_tensor.value

        ttl_output_tensor = ttl_binary_function(ttl_input_tensor, parameter, output_mem_config=memory_config)

        output_tensor = ttnn.Tensor(ttl_output_tensor)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    binary_function.__name__ = f"ttnn.{name}"
    binary_function.__doc__ = doc

    setattr(THIS_MODULE, name, binary_function)


TTL_BINARY_FUNCTIONS = [
    (
        "pow",
        ttl.tensor.pow,
        r"""pow(input_tensor: ttnn.Tensor, exponent: Union[ttnn.Tensor, float, int]) -> ttnn.Tensor

        Takes the power of each element in input with exponent and returns a tensor with the result.

        .. math::
            pow(\mathrm{{input\_tensor}}_i, \mathrm{{exponent}})

        Args:
            * :attr:`input_tensor`
            * :attr:`exponent`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.pow(tensor, 2)

        """,
    ),
]


for binary_function_name, ttl_binary_function, doc in TTL_BINARY_FUNCTIONS:
    register_ttl_binary_function(binary_function_name, ttl_binary_function, doc)


def _is_scalar(value):
    return isinstance(value, (int, float, complex))


def _torch_add(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, **_):
    input_shape_a = input_tensor_a.shape
    slices = [slice(0, dim) for dim in input_shape_a]
    input_tensor_a = ttnn.from_device(input_tensor_a)
    input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor_a = ttnn.to_torch(input_tensor_a)
    input_tensor_a = input_tensor_a[slices]

    if not _is_scalar(input_tensor_b):
        input_shape_b = input_tensor_b.shape
        slices = [slice(0, dim) for dim in input_shape_b]
        input_tensor_b = ttnn.from_device(input_tensor_b)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b = ttnn.to_torch(input_tensor_b)
        input_tensor_b = input_tensor_b[slices]

    return input_tensor_a + input_tensor_b


@decorate_operation(torch_function=_torch_add)
def add(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: Union[ttnn.Tensor, int, float],
    *,
    alpha: Union[int, float] = 1,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""
    add(input_tensor_a: ttnn.Tensor, input_tensor_b: Union[ttnn.Tensor, int, float], *, alpha: Union[int, float]=1) -> ttnn.Tensor

    Adds :attr:`input_tensor_b`, scaled by :attr:`alpha`, to :attr:`input_tensor_a` and returns the tensor with the same layout as :attr:`input_tensor_a`

    .. math::
        \mathrm{{input\_tensor\_a}}_i + \mathrm{{alpha}} \\times \mathrm{{input\_tensor\_b}}_i

    Supports broadcasting.

    Args:
        * :attr:`input_tensor_a`
        * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to add to :attr:`input_tensor_a`.

    Keyword args:
        :attr:`alpha` (Number): the multiplier for :attr:`input_tensor_b`.

    Example::

        >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
        >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
        >>> output = ttnn.add(tensor1, tensor2, alpha=2)
        >>> print(output)
        ttnn.Tensor([ 1, 4], dtype=bfloat16 )

    """

    if not isinstance(input_tensor_a, ttnn.Tensor):
        raise TypeError("Expected first argument to be a ttnn.Tensor")

    # Swap tensors if input_tensor_a needs to be broadcasted to input_tensor_b
    if (
        isinstance(input_tensor_a, ttnn.Tensor)
        and isinstance(input_tensor_b, ttnn.Tensor)
        and math.prod(input_tensor_a.shape) < math.prod(input_tensor_b.shape)
    ):
        input_tensor_a, input_tensor_b = input_tensor_b, input_tensor_a

    original_shape = input_tensor_a.shape
    input_tensor_a = ttnn.unsqueeze_to_4D(input_tensor_a)
    ttl_input_tensor_a = input_tensor_a.value

    if not ttnn.has_storage_type_of(input_tensor_a, ttl.tensor.StorageType.DEVICE):
        raise RuntimeError("input_tensor_a must be on device!")

    if _is_scalar(input_tensor_b):
        output_tensor = ttnn.Tensor(
            ttl.tensor.add_unary(
                ttl_input_tensor_a,
                input_tensor_b * alpha,
                output_mem_config=memory_config,
            )
        )
        return ttnn.reshape(output_tensor, original_shape)
    elif isinstance(input_tensor_b, ttnn.Tensor):
        input_shape_b = input_tensor_b.shape

        if len(input_shape_b) == 1:
            height_b = 1
            (width_b,) = input_shape_b
        else:
            *_, height_b, width_b = input_shape_b

        input_tensor_b = ttnn.unsqueeze_to_4D(input_tensor_b)
        ttl_input_tensor_b = input_tensor_b.value
        if ttl_input_tensor_b.storage_type() != ttl.tensor.StorageType.DEVICE:
            raise RuntimeError("input_tensor_a must be on device!")
    else:
        raise TypeError("Expected second argument to be a ttnn.Tensor or a scalar")

    ttl_input_tensor_b = input_tensor_b.value
    if alpha != 1:
        ttl_input_tensor_b = ttl.tensor.mul_unary(
            ttl_input_tensor_b,
            alpha,
            output_mem_config=memory_config,
        )

    if height_b == 1 and width_b == 1:
        output_tensor = ttnn.Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                ttl.tensor.BcastOpMath.ADD,
                ttl.tensor.BcastOpDim.HW,
                output_mem_config=memory_config,
            )
        )
    elif height_b == 1:
        output_tensor = ttnn.Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                ttl.tensor.BcastOpMath.ADD,
                ttl.tensor.BcastOpDim.H,
                output_mem_config=memory_config,
            )
        )
    elif width_b == 1:
        output_tensor = ttnn.Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                ttl.tensor.BcastOpMath.ADD,
                ttl.tensor.BcastOpDim.W,
                output_mem_config=memory_config,
            )
        )
    else:
        output_tensor = ttnn.Tensor(
            ttl.tensor.add(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                output_mem_config=memory_config,
            )
        )

    output_tensor = ttnn.reshape(output_tensor, original_shape)
    return output_tensor


def _torch_sub(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, **_):
    input_shape_a = input_tensor_a.shape
    slices = [slice(0, dim) for dim in input_shape_a]
    input_tensor_a = ttnn.from_device(input_tensor_a)
    input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor_a = ttnn.to_torch(input_tensor_a)
    input_tensor_a = input_tensor_a[slices]

    if not _is_scalar(input_tensor_b):
        input_shape_b = input_tensor_b.shape
        slices = [slice(0, dim) for dim in input_shape_b]
        input_tensor_b = ttnn.from_device(input_tensor_b)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b = ttnn.to_torch(input_tensor_b)
        input_tensor_b = input_tensor_b[slices]

    return input_tensor_a - input_tensor_b


@decorate_operation(torch_function=_torch_sub)
def sub(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: Union[ttnn.Tensor, int, float],
    *,
    alpha: Union[int, float] = 1,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""
    sub(input_tensor_a: ttnn.Tensor, input_tensor_b: Union[ttnn.Tensor, int, float], *, alpha: Union[int, float]=1) -> ttnn.Tensor:

    Subtracts :attr:`input_tensor_b`, scaled by :attr:`alpha`, from :attr:`input_tensor_a`.

    .. math::
        \mathrm{{input\_tensor\_a}}_i - \mathrm{{alpha}} \\times \mathrm{{input\_tensor\_b}}_i

    Supports broadcasting.

    Args:
        * :attr:`input_tensor_a`
        * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to subtract from :attr:`input_tensor_a`.

    Keyword args:
        :attr:`alpha` (Number): the multiplier for :attr:`input_tensor_b`.

    Example::

        >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
        >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
        >>> output = ttnn.sub(tensor1, tensor2, alpha=2)
        >>> print(output)
        ttnn.Tensor([ 1, 0], dtype=bfloat16 )
    """
    if not isinstance(input_tensor_a, ttnn.Tensor):
        raise TypeError("Expected first argument to be a ttnn.Tensor")

    original_shape = tuple(input_tensor_a.shape)
    input_tensor_a = ttnn.unsqueeze_to_4D(input_tensor_a)
    ttl_input_tensor_a = input_tensor_a.value

    if ttl_input_tensor_a.storage_type() != ttl.tensor.StorageType.DEVICE:
        raise RuntimeError("input_tensor_a must be on device!")

    if _is_scalar(input_tensor_b):
        output_tensor = ttnn.Tensor(
            ttl.tensor.sub_unary(
                ttl_input_tensor_a,
                input_tensor_b * alpha,
                output_mem_config=memory_config,
            )
        )
        return ttnn.reshape(output_tensor, original_shape)
    elif isinstance(input_tensor_b, ttnn.Tensor):
        input_shape_b = input_tensor_b.shape

        if len(input_shape_b) == 1:
            height_b = 1
            (width_b,) = input_shape_b
        else:
            *_, height_b, width_b = input_shape_b

        input_tensor_b = ttnn.unsqueeze_to_4D(input_tensor_b)
        ttl_input_tensor_b = input_tensor_b.value
        if ttl_input_tensor_b.storage_type() != ttl.tensor.StorageType.DEVICE:
            raise RuntimeError("input_tensor_a must be on device!")
    else:
        raise TypeError("Expected second argument to be a ttnn.Tensor or a scalar")

    ttl_input_tensor_b = input_tensor_b.value

    if alpha != 1:
        ttl_input_tensor_b = ttl.tensor.mul_unary(
            ttl_input_tensor_b,
            alpha,
            output_mem_config=memory_config,
        )

    if height_b == 1 and width_b == 1:
        output_tensor = ttnn.Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                ttl.tensor.BcastOpMath.SUB,
                ttl.tensor.BcastOpDim.HW,
                output_mem_config=memory_config,
            )
        )
    elif height_b == 1:
        output_tensor = ttnn.Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                ttl.tensor.BcastOpMath.SUB,
                ttl.tensor.BcastOpDim.H,
                output_mem_config=memory_config,
            )
        )
    elif width_b == 1:
        output_tensor = ttnn.Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                ttl.tensor.BcastOpMath.SUB,
                ttl.tensor.BcastOpDim.W,
                output_mem_config=memory_config,
            )
        )
    else:
        output_tensor = ttnn.Tensor(
            ttl.tensor.sub(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                output_mem_config=memory_config,
            )
        )

    output_tensor = ttnn.reshape(output_tensor, original_shape)
    return output_tensor


def _torch_mul(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, **_):
    input_shape_a = input_tensor_a.shape
    slices = [slice(0, dim) for dim in input_shape_a]
    input_tensor_a = ttnn.from_device(input_tensor_a)
    input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor_a = ttnn.to_torch(input_tensor_a)
    input_tensor_a = input_tensor_a[slices]

    if not _is_scalar(input_tensor_b):
        input_shape_b = input_tensor_b.shape
        slices = [slice(0, dim) for dim in input_shape_b]
        input_tensor_b = ttnn.from_device(input_tensor_b)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b = ttnn.to_torch(input_tensor_b)
        input_tensor_b = input_tensor_b[slices]

    return input_tensor_a * input_tensor_b


@decorate_operation(torch_function=_torch_mul)
def mul(
    input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
) -> ttnn.Tensor:
    r"""
    mul(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor) -> ttnn.Tensor

    Multiples :attr:`input_tensor_a` and :attr:`input_tensor_b` element-wise.

    .. math::
        \mathrm{{input\_tensor\_a}}_i + \mathrm{{input\_tensor\_b}}_i

    Supports broadcasting.

    Args:
        * :attr:`input_tensor_a`
        * :attr:`input_tensor_b` (ttnn.Tensor or Number): the tensor or number to multiply with :attr:`input_tensor_a`.

    Example::

        >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
        >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
        >>> output = ttnn.mul(tensor1, tensor2)
        >>> print(output)
        ttnn.Tensor([ 0, 2], dtype=bfloat16 )

    """

    original_shape = input_tensor_a.shape
    input_tensor_a = ttnn.unsqueeze_to_4D(input_tensor_a)
    ttl_input_tensor_a = input_tensor_a.value

    if not isinstance(input_tensor_a, ttnn.Tensor):
        raise TypeError("Expected first argument to be a ttnn.Tensor")

    ttl_input_tensor_a = input_tensor_a.value

    if not ttnn.has_storage_type_of(input_tensor_a, ttl.tensor.StorageType.DEVICE):
        raise RuntimeError("input_tensor_a must be on device!")

    if _is_scalar(input_tensor_b):
        return ttnn.reshape(
            ttnn.Tensor(
                ttl.tensor.mul_unary(
                    ttl_input_tensor_a,
                    input_tensor_b,
                    output_mem_config=memory_config,
                )
            ),
            original_shape,
        )
    elif not isinstance(input_tensor_b, ttnn.Tensor):
        raise TypeError("Expected second argument to be a ttnn.Tensor or a scalar")

    input_shape_b = input_tensor_b.shape

    if len(input_shape_b) == 1:
        height_b = 1
        (width_b,) = input_shape_b
    else:
        *_, height_b, width_b = input_shape_b

    input_tensor_b = ttnn.unsqueeze_to_4D(input_tensor_b)
    ttl_input_tensor_b = input_tensor_b.value

    if height_b == 1 and width_b == 1:
        return ttnn.reshape(
            ttnn.Tensor(
                ttl.tensor.bcast(
                    ttl_input_tensor_a,
                    ttl_input_tensor_b,
                    ttl.tensor.BcastOpMath.MUL,
                    ttl.tensor.BcastOpDim.HW,
                    output_mem_config=memory_config,
                ),
                original_shape,
            )
        )
    elif height_b == 1:
        return ttnn.reshape(
            ttnn.Tensor(
                ttl.tensor.bcast(
                    ttl_input_tensor_a,
                    ttl_input_tensor_b,
                    ttl.tensor.BcastOpMath.MUL,
                    ttl.tensor.BcastOpDim.H,
                    output_mem_config=memory_config,
                )
            ),
            original_shape,
        )
    elif width_b == 1:
        return ttnn.reshape(
            ttnn.Tensor(
                ttl.tensor.bcast(
                    ttl_input_tensor_a,
                    ttl_input_tensor_b,
                    ttl.tensor.BcastOpMath.MUL,
                    ttl.tensor.BcastOpDim.W,
                    output_mem_config=memory_config,
                )
            ),
            original_shape,
        )

    return ttnn.reshape(
        ttnn.Tensor(ttl.tensor.mul(ttl_input_tensor_a, ttl_input_tensor_b, output_mem_config=memory_config)),
        original_shape,
    )


subtract = sub
multiply = mul


ttnn.Tensor.__add__ = add
ttnn.Tensor.__radd__ = add
ttnn.Tensor.__sub__ = sub
ttnn.Tensor.__mul__ = mul
ttnn.Tensor.__rmul__ = mul

__all__ = []
