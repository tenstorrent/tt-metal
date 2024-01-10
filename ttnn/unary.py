# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import tt_lib as ttl

from ttnn.tensor import (
    Tensor,
    has_storage_type_of,
    MemoryConfig,
    DRAM_MEMORY_CONFIG,
    DEVICE_STORAGE_TYPE,
)
from ttnn.core import reshape, _reshape_to_4D
from ttnn.decorators import decorate_operation


THIS_MODULE = sys.modules[__name__]

__all__ = []


def register_ttl_unary_function(name, ttl_unary_function):
    def _torch_unary(input_tensor: Tensor, **_):
        import torch
        import ttnn

        name_to_torch_function = {
            "exp": torch.exp,
            "tanh": torch.tanh,
            "gelu": torch.nn.functional.gelu,
            "rsqrt": torch.rsqrt,
            "relu": torch.relu,
        }
        torch_function = name_to_torch_function[name]

        input_tensor = ttnn.from_device(input_tensor)
        input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor = ttnn.to_torch(input_tensor)

        return torch_function(input_tensor)

    @decorate_operation(torch_function=_torch_unary, name=name)
    def unary_function(input_tensor: Tensor, *, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG) -> Tensor:
        original_shape = input_tensor.shape
        input_tensor = _reshape_to_4D(input_tensor)
        ttl_input_tensor = input_tensor.value

        if not isinstance(input_tensor, Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not has_storage_type_of(input_tensor, DEVICE_STORAGE_TYPE):
            raise RuntimeError("input_tensor must be on device!")
        ttl_input_tensor = input_tensor.value

        ttl_output_tensor = ttl_unary_function(ttl_input_tensor, output_mem_config=memory_config)

        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = reshape(output_tensor, original_shape)
        return output_tensor

    unary_function.__name__ = f"ttnn.{name}"
    unary_function.__doc__ = f"""{name}(input_tensor: Tensor) -> Tensor

        Applies {name} to :attr:`input_tensor` element-wise.

        .. math::
            {name}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.{name}(tensor)

        """
    setattr(THIS_MODULE, name, unary_function)
    __all__.append(name)


TTL_UNARY_FUNCTIONS = [
    ("exp", ttl.tensor.exp),
    ("tanh", ttl.tensor.tanh),
    ("gelu", ttl.tensor.gelu),
    ("relu", ttl.tensor.relu),
    ("rsqrt", ttl.tensor.rsqrt),
]


for unary_function_name, ttl_unary_function in TTL_UNARY_FUNCTIONS:
    register_ttl_unary_function(unary_function_name, ttl_unary_function)


def register_ttl_unary_function_with_float_parameter(name, ttl_unary_function):
    def _torch_unary(input_tensor: Tensor, parameter, **_):
        import torch
        import ttnn

        name_to_torch_function = {"pow": torch.pow}
        torch_function = name_to_torch_function[name]

        input_tensor = ttnn.from_device(input_tensor)
        input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor = ttnn.to_torch(input_tensor)

        return torch_function(input_tensor, parameter)

    @decorate_operation(torch_function=_torch_unary, name=name)
    def unary_function(
        input_tensor: Tensor, parameter: float, *, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG
    ) -> Tensor:
        original_shape = input_tensor.shape
        input_tensor = _reshape_to_4D(input_tensor)
        ttl_input_tensor = input_tensor.value

        if not isinstance(input_tensor, Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not has_storage_type_of(input_tensor, DEVICE_STORAGE_TYPE):
            raise RuntimeError("input_tensor must be on device!")
        ttl_input_tensor = input_tensor.value

        ttl_output_tensor = ttl_unary_function(ttl_input_tensor, parameter, output_mem_config=memory_config)

        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = reshape(output_tensor, original_shape)
        return output_tensor

    unary_function.__name__ = f"ttnn.{name}"
    unary_function.__doc__ = f"""{name}(input_tensor: Tensor) -> Tensor

        Applies {name} to :attr:`input_tensor` element-wise.

        .. math::
            {name}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.{name}(tensor, 2)

        """

    setattr(THIS_MODULE, name, unary_function)
    __all__.append(name)


TTL_UNARY_FUNCTIONS_WITH_FLOAT_PARAMETER = [
    ("pow", ttl.tensor.power),
]


for unary_function_name, ttl_unary_function in TTL_UNARY_FUNCTIONS_WITH_FLOAT_PARAMETER:
    register_ttl_unary_function_with_float_parameter(unary_function_name, ttl_unary_function)
