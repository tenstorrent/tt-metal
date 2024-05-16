# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import math

from typing import Union

import tt_lib as ttl

import ttnn


THIS_MODULE = sys.modules[__name__]

__all__ = []


def _is_scalar(value):
    return isinstance(value, (int, float))


def register_ttl_relational_function(name, ttl_relational_function, op_name):
    def _golden_function(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, **_):
        import torch

        name_to_golden_function_function = {
            "gt": torch.gt,
            "gte": torch.ge,
            "lt": torch.lt,
            "lte": torch.le,
            "eq": torch.eq,
            "ne": torch.ne,
        }
        torch_function = name_to_golden_function_function[name]
        return torch_function(input_tensor_a, input_tensor_b)

    def _relational_validate_input_tensors(operation_name, input_tensor_a, input_tensor_b, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor_a,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor_b,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
            can_be_a_scalar=True,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_relational_validate_input_tensors,
        golden_function=_golden_function,
    )
    def relational_function(
        input_tensor_a: ttnn.Tensor,
        input_tensor_b: Union[ttnn.Tensor, int, float],
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        if not isinstance(input_tensor_a, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not isinstance(input_tensor_b, ttnn.Tensor) and not _is_scalar(input_tensor_b):
            raise TypeError("Expected second argument to be a ttnn.Tensor or a scalar")

        if isinstance(input_tensor_a, ttnn.Tensor) and not ttnn.is_tensor_storage_on_device(input_tensor_a):
            raise RuntimeError("input_tensor_a must be on device!")

        if isinstance(input_tensor_b, ttnn.Tensor) and not ttnn.is_tensor_storage_on_device(input_tensor_b):
            raise RuntimeError("input_tensor_b must be on device!")

        original_shape = input_tensor_a.shape

        if _is_scalar(input_tensor_b):
            original_shape = input_tensor_a.shape
            input_tensor_b = ttnn.Tensor(
                ttl.tensor.full(
                    original_shape,
                    input_tensor_b,
                    output_mem_config=memory_config,
                )
            )
        elif (
            isinstance(input_tensor_a, ttnn.Tensor)
            and isinstance(input_tensor_b, ttnn.Tensor)
            and len(input_tensor_a.shape) != len(input_tensor_b.shape)
        ):
            if len(input_tensor_a.shape) > len(input_tensor_b.shape):
                original_shape = input_tensor_a.shape
            else:
                original_shape = input_tensor_b.shape

        input_tensor_a = ttnn.unsqueeze_to_4D(input_tensor_a)
        input_tensor_b = ttnn.unsqueeze_to_4D(input_tensor_b)

        output_tensor = ttl_relational_function(input_tensor_a, input_tensor_b, output_mem_config=memory_config)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    if isinstance(relational_function, ttnn.decorators.Operation):
        relational_function.__name__ = f"ttnn.{name}"
        relational_function.decorated_function.__doc__ = f"""{name}(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

            Performs relational {op_name} operation on two tensors :attr:`input_a` and :attr:`input_b` or one tensor :attr:`input_a` and one :attr:`scalar` element-wise.

            .. math::
                {name}(\\mathrm{{input\\_tensor\\_a}}_i \\; , \\; \\mathrm{{input\\_tensor\\_b}}_i  \\; \\; or \\; \\; \\mathrm{{scalar}})

            Args:
                * :attr:`input_tensor_a`
                * :attr:`input_tensor_b` or :attr:`scalar`

            Example::
                >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 2], [3, 4]]), dtype=torch.bfloat16)), device)
                >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 1], [4, 4]]), dtype=torch.bfloat16)), device)
                >>> output = ttnn.{name}(tensor1, tensor2)
            """

    setattr(THIS_MODULE, name, relational_function)


TTL_RELATIONAL_FUNCTIONS = [
    ("gt", ttl.tensor.gt, "greater than"),
    ("gte", ttl.tensor.gte, "greater than or equal to"),
    ("lt", ttl.tensor.lt, "less than"),
    ("lte", ttl.tensor.lte, "less than or equal to"),
    ("eq", ttl.tensor.eq, "equal to"),
    ("ne", ttl.tensor.ne, "not equal to"),
]

for relational_function_name, ttl_relational_function, name in TTL_RELATIONAL_FUNCTIONS:
    register_ttl_relational_function(relational_function_name, ttl_relational_function, name)

ttnn.Tensor.__eq__ = lambda self, *args, **kwargs: getattr(THIS_MODULE, "eq")(self, *args, **kwargs)
ttnn.Tensor.__ne__ = lambda self, *args, **kwargs: getattr(THIS_MODULE, "ne")(self, *args, **kwargs)
ttnn.Tensor.__gt__ = lambda self, *args, **kwargs: getattr(THIS_MODULE, "gt")(self, *args, **kwargs)
ttnn.Tensor.__ge__ = lambda self, *args, **kwargs: getattr(THIS_MODULE, "ge")(self, *args, **kwargs)
ttnn.Tensor.__lt__ = lambda self, *args, **kwargs: getattr(THIS_MODULE, "lt")(self, *args, **kwargs)
ttnn.Tensor.__le__ = lambda self, *args, **kwargs: getattr(THIS_MODULE, "le")(self, *args, **kwargs)


def _golden_function(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    param1: float = 1e-05,
    param2: float = 1e-08,
    equal_nan: bool = False,
    **_,
):
    import torch

    return torch.isclose(input_tensor_a, input_tensor_b, rtol=param1, atol=param2, equal_nan=equal_nan)


@ttnn.register_operation(
    name=f"ttnn.isclose",
    golden_function=_golden_function,
)
def isclose(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """isclose(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Applies the isclose function to the elements of the input tensor :attr:`input_a` and :attr:`input_b`.

    isclose(input_a, input_b, rtol, atol) = ∣input_a−input_B∣ ≤ atol+rtol×∣input_b∣.

    .. math::
        ttnn.isclose(\\mathrm{{input\\_tensor\\_a}}_i \\; , \\; \\mathrm{{input\\_tensor\\_b}}_i  \\; , \\; \\mathrm{{atol}}\\; , \\; \\mathrm{{rtol}})

    Args:
        * :attr:`input_tensor_a`
        * :attr:`input_tensor_b`



    Example::
        >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 2], [3, 4]]), dtype=torch.bfloat16)), device)
        >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1 + 1e-10, 1], [4, 4 + 1e-10]]), dtype=torch.bfloat16)), device)
        >>> output = ttnn.isclose(tensor1, tensor2, rtol, atol)
    """
    return ttl.tensor.isclose(input_tensor_a, input_tensor_b, rtol, atol, equal_nan, output_mem_config=memory_config)


__all__ = []
