# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib as ttl
from ttnn.core import reshape, _reshape_to_4D
from ttnn.decorators import decorate_operation
from ttnn.tensor import (
    Tensor,
    has_storage_type_of,
    MemoryConfig,
    DRAM_MEMORY_CONFIG,
    DEVICE_STORAGE_TYPE,
)


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
        f"""{name}(input_tensor: Tensor) -> Tensor

        Applies {name} to :attr:`input_tensor` element-wise.

        .. math::
            {name}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor`

        Example::

            >>> tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor)
            >>> print(output)
            Tensor([ 0, 2], dtype=bfloat16 )

        """

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

    return unary_function


TTL_UNARY_FUNCTIONS = [
    ("exp", ttl.tensor.exp),
    ("tanh", ttl.tensor.tanh),
    ("gelu", ttl.tensor.gelu),
    ("relu", ttl.tensor.relu),
    ("rsqrt", ttl.tensor.rsqrt),
]

# register functions
exp = register_ttl_unary_function("exp", ttl.tensor.exp)
tanh = register_ttl_unary_function("tanh", ttl.tensor.tanh)
gelu = register_ttl_unary_function("gelu", ttl.tensor.gelu)
relu = register_ttl_unary_function("relu", ttl.tensor.relu)
rsqrt = register_ttl_unary_function("rsqrt", ttl.tensor.rsqrt)


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
        f"""{name}(input_tensor: Tensor) -> Tensor

        Applies {name} to :attr:`input_tensor` element-wise.

        .. math::
            {name}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor`

        Example::

            >>> tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor, 2)
            >>> print(output)
            Tensor([ 1, 4], dtype=bfloat16 )

        """

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

    return unary_function


# register functions
pow = register_ttl_unary_function_with_float_parameter("pow", ttl.tensor.power)

__all__ = [
    "exp",
    "tanh",
    "gelu",
    "relu",
    "rsqrt",
    "pow",
]
