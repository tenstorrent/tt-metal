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


def register_ttl_function_with_float_parameter(name, ttl_binary_function, torch_function):
    def _torch_creator(input_shape: tuple, parameter, **_):
        assert torch_function, f"Torch function not implemented for {str(ttl_binary_function)}"
        return torch_function(input_shape, parameter)

    @decorate_operation(torch_function=_torch_creator, name=name)
    def binary_function(
        input_shape: tuple,
        parameter: float,
        *,
        memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
    ) -> Tensor:
        f"""{name}(input_shape: tuple, parameter: float) -> Tensor

        Generates a Tensor of {name} with attributes :attr:`input_shape`  and  :attr:`parameter` .

        .. math::
            {name}(\\mathrm{{input\\_shape}}_i)
            {name}(\\mathrm{{parameter}}_i)

        Args:
            * :attr:`input_shape`
            * :attr:`parameter`

        Example::

            >>> output = ttnn.{name}(input_shape, parameter)
            >>> print(output)

        """

        ttl_output_tensor = ttl_binary_function(input_shape, parameter, output_mem_config=memory_config)

        output_tensor = Tensor(ttl_output_tensor)
        return output_tensor

    setattr(THIS_MODULE, name, binary_function)
    __all__.append(name)


def register_ttl_function_with_shape(name, ttl_binary_function, torch_function):
    def _torch_creator(input_shape: tuple, **_):
        assert torch_function, f"Torch function not implemented for {str(ttl_binary_function)}"
        return torch_function(input_shape)

    @decorate_operation(torch_function=_torch_creator, name=name)
    def binary_function(
        input_shape: tuple,
        *,
        memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
    ) -> Tensor:
        f"""{name}(input_shape: tuple) -> Tensor

        Generates a Tensor of {name} with attributes :attr:`input_shape`.

        .. math::
            {name}(\\mathrm{{input\\_shape}}_i)

        Args:
            * :attr:`input_shape`

        Example::

            >>> output = ttnn.{name}(input_shape)
            >>> print(output)

        """

        ttl_output_tensor = ttl_binary_function(input_shape, output_mem_config=memory_config)

        output_tensor = Tensor(ttl_output_tensor)
        return output_tensor

    setattr(THIS_MODULE, name, binary_function)
    __all__.append(name)


# register functions


TTL_CREATE_FUNCTIONS = [
    ("ones", ttl.tensor.ones, torch.ones),
    ("zeros", ttl.tensor.zeros, torch.zeros),
]

for binary_function_name, ttl_binary_function, torch_function in TTL_CREATE_FUNCTIONS:
    register_ttl_function_with_shape(binary_function_name, ttl_binary_function, torch_function)


TTL_CREATE_FUNCTIONS_WITH_FLOAT_PARAMETER = [
    ("full", ttl.tensor.full, torch.full),
]

for binary_function_name, ttl_binary_function, torch_function in TTL_CREATE_FUNCTIONS_WITH_FLOAT_PARAMETER:
    register_ttl_function_with_float_parameter(binary_function_name, ttl_binary_function, torch_function)
