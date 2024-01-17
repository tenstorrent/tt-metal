# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import tt_lib as ttl

import ttnn.tensor as ttnn
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
    __all__.append(name)


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
