# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import tt_lib as ttl

from ttnn.tensor import (
    Tensor,
    MemoryConfig,
    DRAM_MEMORY_CONFIG,
)
from ttnn.core import reshape, _reshape_to_4D


THIS_MODULE = sys.modules[__name__]

__all__ = []


def register_ttl_unary_function(name):
    ttl_unary_function = getattr(ttl.tensor, name)

    def unary_function(input_tensor: Tensor, *, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG) -> Tensor:
        f"""{name}(input_tensor: Tensor) -> Tensor

        Applies {name} to :attr:`input_tensor` element-wise.

        .. math::
            {name}(\mathrm{{input\_tensor}}_i)

        Args:
            * :attr:`input_tensor`

        Example::

            >>> tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor)
            >>> print(output)
            Tensor([ 0, 2], dtype=bfloat16 )

        """

        original_shape = tuple(input_tensor.shape)
        input_tensor = _reshape_to_4D(input_tensor)
        ttl_input_tensor = input_tensor._tensor

        if not isinstance(input_tensor, Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not input_tensor.is_on_device:
            raise RuntimeError("input_tensor must be on device!")
        ttl_input_tensor = input_tensor._tensor

        ttl_output_tensor = ttl_unary_function(ttl_input_tensor)

        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = reshape(output_tensor, original_shape)
        return output_tensor

    setattr(THIS_MODULE, name, unary_function)
    __all__.append(name)


TTL_UNARY_FUNCTIONS = [
    "exp",
    "tanh",
    "gelu",
]


for unary_function_name in TTL_UNARY_FUNCTIONS:
    register_ttl_unary_function(unary_function_name)
