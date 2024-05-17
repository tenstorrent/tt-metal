# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import math

from typing import Union

import tt_lib as ttl

import ttnn


THIS_MODULE = sys.modules[__name__]

__all__ = []


def register_ttl_relational_function_zero(name, ttl_relational_function, op_function):
    def _golden_function(input_tensor: ttnn.Tensor, **_):
        import torch

        name_to_golden_function_function = {
            "gtz": torch.gt,
            "ltz": torch.lt,
            "gez": torch.ge,
            "lez": torch.le,
            "nez": torch.ne,
            "eqz": torch.eq,
        }
        torch_function = name_to_golden_function_function[name]
        return torch_function(input_tensor, 0)

    def _relational_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_relational_validate_input_tensors,
        golden_function=_golden_function,
    )
    def relational_function(
        input_tensor: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        original_shape = input_tensor.shape
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not ttnn.is_tensor_storage_on_device(input_tensor):
            raise RuntimeError("input_tensor must be on device!")

        output_tensor = ttl_relational_function(input_tensor, output_mem_config=memory_config)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    if isinstance(relational_function, ttnn.decorators.Operation):
        relational_function.__name__ = f"ttnn.{name}"
        relational_function.decorated_function.__doc__ = f"""{name}(input_tensor: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

            Returns tensor with the {op_function} of all of the elements of the input tensor :attr:`input_tensor` element-wise.

            .. math::
                {name}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor`

            Example::

                >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> output = ttnn.{name}(tensor)

            """
    setattr(THIS_MODULE, name, relational_function)


def _is_scalar(value):
    return isinstance(value, (int, float))


TTL_RELATIONAL_FUNCTIONS_ZERO = [
    ("gtz", ttl.tensor.gtz, "greater than zero"),
    ("ltz", ttl.tensor.ltz, "less than zero"),
    ("gez", ttl.tensor.gez, "greater than or equal to zero"),
    ("lez", ttl.tensor.lez, "less than or equal to zero"),
    ("nez", ttl.tensor.nez, "not equal to zero"),
    ("eqz", ttl.tensor.eqz, "equal to zero"),
]


for relational_function_name, ttl_relational_function, name in TTL_RELATIONAL_FUNCTIONS_ZERO:
    register_ttl_relational_function_zero(relational_function_name, ttl_relational_function, name)
