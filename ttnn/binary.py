# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import sys
import tt_lib as ttl
from ttnn.core import reshape, _reshape_to_4D
from ttnn.decorators import decorate_operation
from typing import Union
from ttnn.tensor import (
    Tensor,
    has_storage_type_of,
    MemoryConfig,
    DRAM_MEMORY_CONFIG,
    DEVICE_STORAGE_TYPE,
)
from ttnn.core import reshape, _reshape_to_4D
from ttnn.decorators import decorate_operation
import torch
import torch.nn.functional as F

THIS_MODULE = sys.modules[__name__]

__all__ = []


def register_ttl_binary_function(name, ttl_binary_function, torch_function):
    def _torch_binary(input_tensor_a: Tensor, input_tensor_b: Tensor, **_):
        import torch
        import ttnn

        input_tensor_a = ttnn.from_device(input_tensor_a)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a = ttnn.to_torch(input_tensor_a)

        input_tensor_b = ttnn.from_device(input_tensor_b)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b = ttnn.to_torch(input_tensor_b)
        assert torch_function, f"Torch function not implemented for {str(ttl_binary_function)}"
        return torch_function(input_tensor_a, input_tensor_b)

    @decorate_operation(torch_function=_torch_binary, name=name)
    def binary_function(
        input_tensor_a: Tensor, input_tensor_b: Tensor, *, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG
    ) -> Tensor:
        f"""{name}(input_tensor_a: Tensor, input_tensor_b: Tensor) -> Tensor

        Applies {name} to :attr:`input_tensor_a` and  :attr:`input_tensor_b` element-wise.

        .. math::
            {name}(\\mathrm{{input\\_tensor_a}}_i)
            {name}(\\mathrm{{input\\_tensor_b}}_i)

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`

        Example::

            >>> tensor_a = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor_b = ttnn.to_device(ttnn.from_torch(torch.tensor((2, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor_a, tensor_b)
            >>> print(output)
            Tensor([ 1, 0], dtype=bfloat16 )

        """
        if not (input_tensor_a.shape == input_tensor_b.shape):
            raise RuntimeError("input_tensors must be of same size!")

        original_shape = input_tensor_a.shape
        input_tensor_a = _reshape_to_4D(input_tensor_a)
        input_tensor_b = _reshape_to_4D(input_tensor_b)

        if not isinstance(input_tensor_a, Tensor) or not isinstance(input_tensor_b, Tensor):
            raise TypeError("Expected both arguments to be a ttnn.Tensor")

        if not has_storage_type_of(input_tensor_a, DEVICE_STORAGE_TYPE) or not has_storage_type_of(
            input_tensor_b, DEVICE_STORAGE_TYPE
        ):
            raise RuntimeError("input_tensors must be on device!")

        ttl_input_tensor_a = input_tensor_a.value
        ttl_input_tensor_b = input_tensor_b.value

        ttl_output_tensor = ttl_binary_function(ttl_input_tensor_a, ttl_input_tensor_b, output_mem_config=memory_config)

        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = reshape(output_tensor, original_shape)
        return output_tensor

    setattr(THIS_MODULE, name, binary_function)
    __all__.append(name)
    return binary_function


def register_ttl_binary_function_with_float_parameter(name, ttl_binary_function, torch_function):
    def _torch_binary(input_tensor_a: Tensor, input_tensor_b: Tensor, parameter, **_):
        import torch
        import ttnn

        input_tensor_a = ttnn.from_device(input_tensor_a)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a = ttnn.to_torch(input_tensor_a)

        input_tensor_b = ttnn.from_device(input_tensor_b)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b = ttnn.to_torch(input_tensor_b)
        assert torch_function, f"Torch function not implemented for {str(ttl_binary_function)}"
        return torch_function(input_tensor_a, input_tensor_b, parameter)

    @decorate_operation(torch_function=_torch_binary, name=name)
    def binary_function(
        input_tensor_a: Tensor,
        input_tensor_b: Tensor,
        parameter: float,
        *,
        memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
    ) -> Tensor:
        f"""{name}(input_tensor_a: Tensor, input_tensor_b: Tensor) -> Tensor

        Applies {name} to :attr:`input_tensor_a`  and  :attr:`input_tensor_b` element-wise.

        .. math::
            {name}(\\mathrm{{input\\_tensor_a}}_i)
            {name}(\\mathrm{{input\\_tensor_b}}_i)

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`

        Example::

            >>> tensor_a = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> tensor_b = ttnn.to_device(ttnn.from_torch(torch.tensor((2, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor_a, tensor_b, 2.0)
            >>> print(output)
            Tensor([ -3, -2], dtype=bfloat16 )

        """

        if not (input_tensor_a.shape == input_tensor_b.shape):
            raise RuntimeError("input_tensors must be of same size!")

        original_shape = input_tensor_a.shape
        input_tensor_a = _reshape_to_4D(input_tensor_a)
        input_tensor_b = _reshape_to_4D(input_tensor_b)

        if not isinstance(input_tensor_a, Tensor) or not isinstance(input_tensor_b, Tensor):
            raise TypeError("Expected both arguments to be a ttnn.Tensor")

        if not has_storage_type_of(input_tensor_a, DEVICE_STORAGE_TYPE) or not has_storage_type_of(
            input_tensor_b, DEVICE_STORAGE_TYPE
        ):
            raise RuntimeError("input_tensors must be on device!")

        ttl_input_tensor_a = input_tensor_a.value
        ttl_input_tensor_b = input_tensor_b.value

        ttl_output_tensor = ttl_binary_function(
            ttl_input_tensor_a, ttl_input_tensor_b, parameter, output_mem_config=memory_config
        )

        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = reshape(output_tensor, original_shape)
        return output_tensor

    setattr(THIS_MODULE, name, binary_function)
    __all__.append(name)
    return binary_function


# register functions


def torch_squared_difference(x, y, *args, **kwargs):
    t_diff = torch.sub(x, y)
    return torch.square(t_diff)


TTL_BINARY_FUNCTIONS = [
    ("atan2", ttl.tensor.atan2, torch.atan2),
    ("hypot", ttl.tensor.hypot, torch.hypot),
    ("ldexp", ttl.tensor.ldexp, torch.ldexp),
    ("logaddexp", ttl.tensor.logaddexp, torch.logaddexp),
    ("logaddexp2", ttl.tensor.logaddexp2, torch.logaddexp2),
    ("logical_and", ttl.tensor.logical_and, torch.logical_and),
    ("logical_or", ttl.tensor.logical_or, torch.logical_or),
    ("logical_xor", ttl.tensor.logical_xor, torch.logical_xor),
    ("squared_difference", ttl.tensor.squared_difference, torch_squared_difference),
]


for binary_function_name, ttl_binary_function, torch_function in TTL_BINARY_FUNCTIONS:
    register_ttl_binary_function(binary_function_name, ttl_binary_function, torch_function)


TTL_BINARY_FUNCTIONS_WITH_FLOAT_PARAMETER = [
    ("addalpha", ttl.tensor.addalpha, torch.add),
    ("subalpha", ttl.tensor.subalpha, torch.sub),
]

for binary_function_name, ttl_binary_function, torch_function in TTL_BINARY_FUNCTIONS_WITH_FLOAT_PARAMETER:
    register_ttl_binary_function_with_float_parameter(binary_function_name, ttl_binary_function, torch_function)
