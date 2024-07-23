# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Union

import ttnn
import ttnn._ttnn.deprecated as ttl

__all__ = []


def _is_scalar(value):
    return isinstance(value, (float))


def torch_mac(input, tensor1, tensor2):
    import torch

    return torch.add(torch.mul(input, tensor1), tensor2)


def register_ttl_ternary_function(name, ttl_ternary_function):
    def _golden_function(input_tensor: ttnn.Tensor, **_):
        import torch

        name_to_golden_function_function = {
            "mac": torch_mac,
        }
        torch_function = name_to_golden_function_function[name]
        return torch_function(input_tensor)

    doc = __doc__ = f"""{name}(input_tensor: ttnn.Tensor, input_tensor1: ttnn.Tensor, input_tensor2: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

            Returns tensor with the {name} of all of elements of the input tensors input, tensor1, tensor2.

            .. math::
                {name.replace('_',' ')}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor`
                * :attr:`input_tensor1`
                * :attr:`input_tensor2`

            Example::

                >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> tensor1 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> tensor2 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> output = ttnn.{name}(tensor, tensor1, tensor2)

            """

    @ttnn.register_python_operation(
        name=f"ttnn.{name}",
        golden_function=_golden_function,
        doc=doc,
    )
    def ternary_function(
        input: ttnn.Tensor,
        input1: Union[ttnn.Tensor, float],
        input2: Union[ttnn.Tensor, float],
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        original_shape = input.shape
        input_tensor = ttnn.unsqueeze_to_4D(input)
        if not _is_scalar(input1) and not _is_scalar(input2):
            input_tensor1 = ttnn.unsqueeze_to_4D(input1)
            input_tensor2 = ttnn.unsqueeze_to_4D(input2)
        elif _is_scalar(input1) and _is_scalar(input2):
            input_tensor1 = input1
            input_tensor2 = input2

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected input argument to be a ttnn.Tensor")

        if (isinstance(input_tensor1, ttnn.Tensor) and not isinstance(input_tensor2, ttnn.Tensor)) or (
            not isinstance(input_tensor1, ttnn.Tensor) and isinstance(input_tensor2, ttnn.Tensor)
        ):
            raise TypeError("Expected other two inputs as either both tensor or scalar")

        if not ttnn.is_tensor_storage_on_device(input_tensor):
            raise RuntimeError("input_tensor must be on device!")

        output_tensor = ttl_ternary_function(input_tensor, input_tensor1, input_tensor2)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor


TTL_TERNARY_FUNCTIONS = [
    ("mac", ttl.tensor.mac),
]


for ternary_function_name, ttl_ternary_function in TTL_TERNARY_FUNCTIONS:
    register_ttl_ternary_function(ternary_function_name, ttl_ternary_function)


def _golden_function_addcmul(input_tensor_a, input_tensor_b, input_tensor_c, *args, value=1, **kwargs):
    import torch

    return torch.addcmul(input_tensor_a, input_tensor_b, input_tensor_c, value=value)


ttnn.attach_golden_function(ttnn.addcmul, golden_function=_golden_function_addcmul)


def _golden_function_addcdiv(input_tensor_a, input_tensor_b, input_tensor_c, *args, value=1, **kwargs):
    import torch

    return torch.addcdiv(input_tensor_a, input_tensor_b, input_tensor_c, value=value)


ttnn.attach_golden_function(ttnn.addcdiv, golden_function=_golden_function_addcdiv)


def _golden_function_where(input_tensor_a, input_tensor_b, input_tensor_c, *args, **kwargs):
    import torch

    return torch.where(input_tensor_a, input_tensor_b, input_tensor_c)


ttnn.attach_golden_function(ttnn.where, golden_function=_golden_function_where)


def _golden_function(
    input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, weight: Union[ttnn.Tensor, int, float], **_
):
    import torch

    return torch.lerp(input_tensor_a, input_tensor_b, weight)


@ttnn.register_python_operation(
    name=f"ttnn.lerp",
    golden_function=_golden_function,
)
def lerp(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    weight: Union[ttnn.Tensor, int, float],
    *,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    if not (input_tensor_a.shape == input_tensor_b.shape):
        raise RuntimeError("input_tensors must be of same size!")

    if not isinstance(input_tensor_a, ttnn.Tensor) or not isinstance(input_tensor_b, ttnn.Tensor):
        raise TypeError("Expected both arguments to be a ttnn.Tensor")

    if not ttnn.is_tensor_storage_on_device(input_tensor_a) or not ttnn.is_tensor_storage_on_device(input_tensor_b):
        raise RuntimeError("input_tensors must be on device!")

    if isinstance(weight, ttnn.Tensor) and not (input_tensor_a.shape == weight.shape):
        raise RuntimeError("weight tensor must be of same size!")

    if isinstance(weight, ttnn.Tensor) and not ttnn.is_tensor_storage_on_device(weight):
        raise RuntimeError("weight tensor must be on device!")

    original_shape = input_tensor_a.shape

    input_tensor_a = ttnn.unsqueeze_to_4D(input_tensor_a)
    input_tensor_b = ttnn.unsqueeze_to_4D(input_tensor_b)

    if isinstance(weight, ttnn.Tensor):
        weight = ttnn.unsqueeze_to_4D(weight)

    output_tensor = ttl.tensor.lerp(input_tensor_a, input_tensor_b, weight, output_mem_config=memory_config)

    output_tensor = ttnn.reshape(output_tensor, original_shape)
    return output_tensor


__all__ = []
