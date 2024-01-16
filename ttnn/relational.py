# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import sys
import tt_lib as ttl
from ttnn.common import rst_escape
from ttnn.core import reshape, _reshape_to_4D
from ttnn.decorators import decorate_operation
from typing import Union
from ttnn.tensor import (
    Tensor,
    has_storage_type_of,
    MemoryConfig,
    DRAM_MEMORY_CONFIG,
    DEVICE_STORAGE_TYPE,
    from_torch,
    to_torch,
    to_device,
)
from ttnn.core import reshape, _reshape_to_4D
from ttnn.decorators import decorate_operation
from ttnn.unary import full_like
import torch
import torch.nn.functional as F

THIS_MODULE = sys.modules[__name__]

__all__ = []


# reference functions
def torch_gte(x, y, *args, **kwargs):
    return (x >= y).to(x.dtype)


def torch_lte(x, y, *args, **kwargs):
    return (x <= y).to(x.dtype)


def torch_gtz(x, *args, **kwargs):
    return (x > 0.0).to(x.dtype)


def torch_gez(x, *args, **kwargs):
    return (x >= 0.0).to(x.dtype)


def torch_eqz(x, *args, **kwargs):
    return (x == 0.0).to(x.dtype)


def torch_ltz(x, *args, **kwargs):
    return (x < 0.0).to(x.dtype)


def torch_nez(x, *args, **kwargs):
    return (x != 0.0).to(x.dtype)


def __get_converse_operator(ttl_relational_function):
    if ttl_relational_function == ttl.tensor.gt:
        return ttl.tensor.lt, torch.lt
    if ttl_relational_function == ttl.tensor.lt:
        return ttl.tensor.gt, torch.gt
    if ttl_relational_function == ttl.tensor.gte:
        return ttl.tensor.lte, torch_lte
    if ttl_relational_function == ttl.tensor.lte:
        return ttl.tensor.gte, torch_gte
    if ttl_relational_function == ttl.tensor.eq:
        return ttl.tensor.ne, torch.ne
    if ttl_relational_function == ttl.tensor.ne:
        return ttl.tensor.eq, torch.eq

    raise ValueError(f"Cannot find converse for function {ttl_relational_function}")


def register_ttl_relational_function(name, ttl_relational_function, torch_function):
    # ttl_converse_relational_function, torch_converse_function = __get_converse_operator( ttl_relational_function )
    def _torch_relational(input_tensor_a: Tensor, input_tensor_b: Tensor, **_):
        import ttnn

        input_tensor_a = ttnn.from_device(input_tensor_a)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a = ttnn.to_torch(input_tensor_a)

        input_tensor_b = ttnn.from_device(input_tensor_b)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b = ttnn.to_torch(input_tensor_b)
        assert torch_function, f"Torch function not implemented for {str(ttl_relational_function)}"
        return torch_function(input_tensor_a, input_tensor_b)

    @decorate_operation(torch_function=_torch_relational, name=name)
    def relational_function(
        input_tensor_a: Tensor, input_tensor_b: Tensor, *, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG
    ) -> Tensor:
        assert not all(
            [isinstance(input_tensor_a, (float, int)), isinstance(input_tensor_b, (float, int))]
        ), "both operands cannot be float"
        if isinstance(input_tensor_a, (float, int)):
            # breakpoint()
            input_tensor_a = full_like(input_tensor_b, input_tensor_a)
            # ttl_relational_function, torch_function = ttl_converse_relational_function, torch_converse_function
            # input_tensor_a, input_tensor_b = input_tensor_b, input_tensor_a
        elif isinstance(input_tensor_b, (float, int)):
            input_tensor_b = full_like(input_tensor_a, input_tensor_b)

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

        # breakpoint()
        ttl_output_tensor = ttl_relational_function(
            ttl_input_tensor_a, ttl_input_tensor_b, output_mem_config=memory_config
        )

        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = reshape(output_tensor, original_shape)
        return output_tensor

    relational_function.__name__ = f"ttnn.{rst_escape(name)}"
    relational_function.__doc__ = f"""{rst_escape(name)}(input_tensor_a: Tensor, input_tensor_b: Tensor) -> Tensor

        Applies {rst_escape(name)} to :attr:`input_tensor_a` and  :attr:`input_tensor_b` element-wise.

        .. math::
            {rst_escape(name)}(\\mathrm{{input\\_tensor}}_i)

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
    setattr(THIS_MODULE, name, relational_function)
    __all__.append(name)
    return relational_function


TTL_RELATIONAL_FUNCTIONS = [
    ("gt", ttl.tensor.gt, torch.gt),
    ("gte", ttl.tensor.gte, torch_gte),
    ("eq", ttl.tensor.eq, torch.eq),
    ("lt", ttl.tensor.lt, torch.lt),
    ("lte", ttl.tensor.lte, torch_lte),
    ("ne", ttl.tensor.ne, torch.ne),
]


for relational_function_name, ttl_relational_function, torch_function in TTL_RELATIONAL_FUNCTIONS:
    register_ttl_relational_function(relational_function_name, ttl_relational_function, torch_function)


def register_ttl_relational_function_with_zero(name, ttl_relational_function, torch_function):
    def _torch_relational(input_tensor_a: Tensor, **_):
        import torch
        import ttnn

        input_tensor_a = ttnn.from_device(input_tensor_a)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a = ttnn.to_torch(input_tensor_a)

        assert torch_function, f"Torch function not implemented for {str(ttl_relational_function)}"
        return torch_function(input_tensor_a)

    @decorate_operation(torch_function=_torch_relational, name=name)
    def relational_function(input_tensor_a: Tensor, *, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG) -> Tensor:
        original_shape = input_tensor_a.shape
        input_tensor_a = _reshape_to_4D(input_tensor_a)

        if not isinstance(input_tensor_a, Tensor):
            raise TypeError("Expected both arguments to be a ttnn.Tensor")

        if not has_storage_type_of(input_tensor_a, DEVICE_STORAGE_TYPE):
            raise RuntimeError("input_tensors must be on device!")

        ttl_input_tensor_a = input_tensor_a.value

        ttl_output_tensor = ttl_relational_function(ttl_input_tensor_a, output_mem_config=memory_config)

        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = reshape(output_tensor, original_shape)
        return output_tensor

    relational_function.__name__ = f"ttnn.{rst_escape(name)}"
    relational_function.__doc__ = f"""{rst_escape(name)}(input_tensor_a: Tensor) -> Tensor

        Applies {rst_escape(name)} to :attr:`input_tensor_a`  element-wise.

        .. math::
            {rst_escape(name)}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor_a`

        Example::

            >>> tensor_a = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor_a, tensor_b)
            >>> print(output)
            Tensor([ 1, 0], dtype=bfloat16 )

        """
    setattr(THIS_MODULE, name, relational_function)
    __all__.append(name)
    return relational_function


TTL_RELATIONAL_FUNCTIONS_WITH_ZERO = [
    ("gtz", ttl.tensor.gtz, torch_gtz),
    ("gez", ttl.tensor.gez, torch_gez),
    ("eqz", ttl.tensor.eqz, torch_eqz),
    ("ltz", ttl.tensor.ltz, torch_ltz),
    ("nez", ttl.tensor.nez, torch_nez),
]


for relational_function_name, ttl_relational_function, torch_function in TTL_RELATIONAL_FUNCTIONS_WITH_ZERO:
    register_ttl_relational_function_with_zero(relational_function_name, ttl_relational_function, torch_function)
