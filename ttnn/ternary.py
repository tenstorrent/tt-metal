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
import torch
import torch.nn.functional as F

THIS_MODULE = sys.modules[__name__]

__all__ = []


def register_ttl_ternary_function(name, ttl_ternary_function, torch_function):
    def _torch_ternary(input_tensor_a: Tensor, input_tensor_b: Tensor, input_tensor_c: Tensor, **_):
        import torch
        import ttnn

        input_tensor_a = ttnn.from_device(input_tensor_a)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a = ttnn.to_torch(input_tensor_a)

        input_tensor_b = ttnn.from_device(input_tensor_b)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b = ttnn.to_torch(input_tensor_b)

        input_tensor_c = ttnn.from_device(input_tensor_c)
        input_tensor_c = ttnn.to_layout(input_tensor_c, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_c = ttnn.to_torch(input_tensor_c)

        assert torch_function, f"Torch function not implemented for {str(ttl_ternary_function)}"
        return torch_function(input_tensor_a, input_tensor_b, input_tensor_c)

    @decorate_operation(torch_function=_torch_ternary, name=name)
    def ternary_function(
        input_tensor_a: Tensor,
        input_tensor_b: Tensor,
        input_tensor_c: Tensor,
        *,
        memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
    ) -> Tensor:
        f"""{name}(input_tensor_a: Tensor, input_tensor_b: Tensor, input_tensor_c: Tensor) -> Tensor
        Applies {name} to :attr:`input_tensor_a` , :attr:`input_tensor_b` and  :attr:`input_tensor_c` element-wise.
        .. math::
            {name}(\\mathrm{{input\\_tensor}}_i)
        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`
            * :attr:`input_tensor_c`
        Example::
            >>> tensor_a = ttnn.to_device(ttnn.from_torch(torch.tensor((2, 2), dtype=torch.bfloat16)), device)
            >>> tensor_b = ttnn.to_device(ttnn.from_torch(torch.tensor((2, 2), dtype=torch.bfloat16)), device)
            >>> tensor_c = ttnn.to_device(ttnn.from_torch(torch.tensor((2, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor_a, tensor_b, tensor_c)
            >>> print(output)
            Tensor([ 1, 0], dtype=bfloat16 )
        """

        original_shape = input_tensor_a.shape
        input_tensor_a = _reshape_to_4D(input_tensor_a)
        input_tensor_b = _reshape_to_4D(input_tensor_b)
        input_tensor_c = _reshape_to_4D(input_tensor_c)

        if (
            not isinstance(input_tensor_a, Tensor)
            or not isinstance(input_tensor_b, Tensor)
            or not isinstance(input_tensor_c, Tensor)
        ):
            raise TypeError("Expected three arguments to be a ttnn.Tensor")

        if (
            not has_storage_type_of(input_tensor_a, DEVICE_STORAGE_TYPE)
            or not has_storage_type_of(input_tensor_b, DEVICE_STORAGE_TYPE)
            or not has_storage_type_of(input_tensor_c, DEVICE_STORAGE_TYPE)
        ):
            raise RuntimeError("input_tensors must be on device!")

        ttl_input_tensor_a = input_tensor_a.value
        ttl_input_tensor_b = input_tensor_b.value
        ttl_input_tensor_c = input_tensor_c.value

        ttl_output_tensor = ttl_ternary_function(
            ttl_input_tensor_a, ttl_input_tensor_b, ttl_input_tensor_c, output_mem_config=memory_config
        )

        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = reshape(output_tensor, original_shape)
        return output_tensor

    setattr(THIS_MODULE, name, ternary_function)
    __all__.append(name)
    return ternary_function


def register_ttl_unary_function_with_two_float_parameters(name, ttl_unary_function, torch_function):
    def _torch_ternary(input_tensor_a: Tensor, parameter_1: float, parameter_2: float, **_):
        import torch
        import ttnn

        input_tensor_a = ttnn.from_device(input_tensor_a)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a = ttnn.to_torch(input_tensor_a)

        assert torch_function, f"Torch function not implemented for {str(ttl_ternary_function)}"
        return torch_function(input_tensor_a, parameter_1, parameter_2)

    @decorate_operation(torch_function=_torch_ternary, name=name)
    def ternary_function(
        input_tensor_a: Tensor,
        parameter_1: float,
        parameter_2: float,
        *,
        memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
    ) -> Tensor:
        f"""{name}(input_tensor_a: Tensor) -> Tensor
        Applies {name} to :attr:`input_tensor_a` element-wise.
        .. math::
            {name}(\\mathrm{{input\\_tensor}}_i)
        Args:
            * :attr:`input_tensor_a`
        Example::
            >>> tensor_a = ttnn.to_device(ttnn.from_torch(torch.tensor((2, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor_a, 2, 3)
            >>> print(output)
            Tensor([ 1, 0], dtype=bfloat16 )
        """

        original_shape = input_tensor_a.shape
        input_tensor_a = _reshape_to_4D(input_tensor_a)

        if not isinstance(input_tensor_a, Tensor):
            raise TypeError("Expected to be a ttnn.Tensor")

        if not has_storage_type_of(input_tensor_a, DEVICE_STORAGE_TYPE):
            raise RuntimeError("input_tensors must be on device!")

        ttl_input_tensor_a = input_tensor_a.value

        ttl_output_tensor = ttl_ternary_function(
            ttl_input_tensor_a, parameter_1, parameter_2, output_mem_config=memory_config
        )

        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = reshape(output_tensor, original_shape)
        return output_tensor

    setattr(THIS_MODULE, name, ternary_function)
    __all__.append(name)
    return ternary_function


def register_ttl_ternary_function_with_float_parameter(name, ttl_ternary_function, torch_function):
    def _torch_ternary(input_tensor_a: Tensor, input_tensor_b: Tensor, input_tensor_c: Tensor, parameter, **_):
        import torch
        import ttnn

        input_tensor_a = ttnn.from_device(input_tensor_a)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a = ttnn.to_torch(input_tensor_a)

        input_tensor_b = ttnn.from_device(input_tensor_b)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b = ttnn.to_torch(input_tensor_b)

        input_tensor_c = ttnn.from_device(input_tensor_c)
        input_tensor_c = ttnn.to_layout(input_tensor_c, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_c = ttnn.to_torch(input_tensor_c)

        assert torch_function, f"Torch function not implemented for {str(ttl_ternary_function)}"
        return torch_function(input_tensor_a, input_tensor_b, input_tensor_c, parameter)

    @decorate_operation(torch_function=_torch_ternary, name=name)
    def ternary_function(
        input_tensor_a: Tensor,
        input_tensor_b: Tensor,
        input_tensor_c: Tensor,
        parameter: float,
        *,
        memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
    ) -> Tensor:
        f"""{name}(input_tensor_a: Tensor, input_tensor_b: Tensor, input_tensor_c: Tensor) -> Tensor
        Applies {name} to :attr:`input_tensor_a` , :attr:`input_tensor_b` and  :attr:`input_tensor_c` element-wise.
        .. math::
            {name}(\\mathrm{{input\\_tensor}}_i)
        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`
            * :attr:`input_tensor_c`
        Example::
            >>> tensor_a = ttnn.to_device(ttnn.from_torch(torch.tensor((2, 2), dtype=torch.bfloat16)), device)
            >>> tensor_b = ttnn.to_device(ttnn.from_torch(torch.tensor((2, 2), dtype=torch.bfloat16)), device)
            >>> tensor_c = ttnn.to_device(ttnn.from_torch(torch.tensor((2, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor_a, tensor_b, tensor_c, 2)
            >>> print(output)
            Tensor([ 1, 0], dtype=bfloat16 )
        """

        original_shape = input_tensor_a.shape
        input_tensor_a = _reshape_to_4D(input_tensor_a)
        input_tensor_b = _reshape_to_4D(input_tensor_b)
        input_tensor_c = _reshape_to_4D(input_tensor_c)

        if (
            not isinstance(input_tensor_a, Tensor)
            or not isinstance(input_tensor_b, Tensor)
            or not isinstance(input_tensor_c, Tensor)
        ):
            raise TypeError("Expected three arguments to be a ttnn.Tensor")

        if (
            not has_storage_type_of(input_tensor_a, DEVICE_STORAGE_TYPE)
            or not has_storage_type_of(input_tensor_b, DEVICE_STORAGE_TYPE)
            or not has_storage_type_of(input_tensor_c, DEVICE_STORAGE_TYPE)
        ):
            raise RuntimeError("input_tensors must be on device!")

        ttl_input_tensor_a = input_tensor_a.value
        ttl_input_tensor_b = input_tensor_b.value
        ttl_input_tensor_c = input_tensor_c.value

        ttl_output_tensor = ttl_ternary_function(
            ttl_input_tensor_a, ttl_input_tensor_b, ttl_input_tensor_c, parameter, output_mem_config=memory_config
        )

        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = reshape(output_tensor, original_shape)
        return output_tensor

    setattr(THIS_MODULE, name, ternary_function)
    __all__.append(name)
    return ternary_function


# register functions


def torch_mac(x, y, z, *args, **kwargs):
    return x * y + z


TTL_TERNARY_FUNCTIONS = [
    ("mac", ttl.tensor.mac, torch_mac),
    ("where", ttl.tensor.where, torch.where),
]

for ternary_function_name, ttl_ternary_function, torch_function in TTL_TERNARY_FUNCTIONS:
    register_ttl_ternary_function(ternary_function_name, ttl_ternary_function, torch_function)

TTL_TERNARY_FUNCTIONS_WITH_FLOAT_PARAMETER = [
    ("addcmul", ttl.tensor.addcmul, torch.addcmul),
    ("addcdiv", ttl.tensor.addcdiv, torch.addcdiv),
]

for ternary_function_name, ttl_ternary_function, torch_function in TTL_TERNARY_FUNCTIONS_WITH_FLOAT_PARAMETER:
    register_ttl_ternary_function_with_float_parameter(ternary_function_name, ttl_ternary_function, torch_function)

TTL_UNARY_FUNCTIONS_WITH_TWO_FLOAT_PARAMETERS = [
    ("clip", ttl.tensor.clip, torch.clip),
]

for ternary_function_name, ttl_ternary_function, torch_function in TTL_UNARY_FUNCTIONS_WITH_TWO_FLOAT_PARAMETERS:
    register_ttl_unary_function_with_two_float_parameters(ternary_function_name, ttl_ternary_function, torch_function)

Tensor.mac = mac
Tensor.where = where
Tensor.addcmul = addcmul
Tensor.addcdiv = addcdiv
Tensor.clip = clip
