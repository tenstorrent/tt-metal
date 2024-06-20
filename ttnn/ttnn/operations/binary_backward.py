# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional

import sys

import ttnn

import tt_lib as ttl

THIS_MODULE = sys.modules[__name__]

__all__ = []


# def register_ttl_binary_backward_function(name, ttl_binary_backward_function, doc):
#     def _golden_function(grad_tensor: ttnn.Tensor, input_tensor_1: ttnn.Tensor, input_tensor_2: ttnn.Tensor, **_):
#         import torch

#         pyt_y = torch.atan2(input_tensor_1, input_tensor_2)
#         input_tensor_1.retain_grad()
#         input_tensor_2.retain_grad()

#         pyt_y.backward(gradient=grad_tensor)
#         name_to_torch_function = {"atan2_bw": pyt_y}
#         torch_function = name_to_torch_function[name]
#         return torch_function(input_tensor_1, input_tensor_2)

#     def _binary_backward_validate_input_tensors(operation_name, grad_tensor, input_tensor_1, input_tensor_2, *args, **kwargs):
#         ttnn.validate_input_tensor(
#             operation_name,
#             grad_tensor,
#             input_tensor_1,
#             input_tensor_2,
#             ranks=(2, 3, 4),
#             dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
#             layouts=(ttnn.TILE_LAYOUT,),
#             can_be_on_device=True,
#             can_be_on_cpu=False,
#         )

#     @ttnn.register_operation(
#         name=f"ttnn.{name}",
#         validate_input_tensors=_binary_backward_validate_input_tensors,
#         golden_function=_golden_function,
#     )
#     def binary_backward_function(
#         grad_tensor: ttnn.Tensor,
#         input_tensor_1: ttnn.Tensor,
#         input_tensor_2: ttnn.Tensor,
#         *,
#         memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
#     ) -> ttnn.Tensor:
#         original_shape = grad_tensor.shape
#         grad_tensor = ttnn.unsqueeze_to_4D(grad_tensor)
#         original_shape_1 = input_tensor_1.shape
#         input_tensor_1 = ttnn.unsqueeze_to_4D(input_tensor_1)
#         original_shape_2 = input_tensor.shape_2
#         input_tensor_2 = ttnn.unsqueeze_to_4D(input_tensor_2)
#         output_tensor = ttl_binary_backward_function(grad_tensor, input_tensor_1, input_tensor_2, memory_config)
#         # output_tensor = ttnn.reshape(output_tensor, original_shape)
#         return output_tensor

#     if isinstance(binary_backward_function, ttnn.decorators.Operation):
#         binary_backward_function.__name__ = f"ttnn.{name}"
#         binary_backward_function.decorated_function.__doc__ = doc + (
#             binary_backward_function.__doc__ if binary_backward_function.__doc__ is not None else ""
#         )

#     setattr(THIS_MODULE, name, binary_backward_function)


# TTL_BINARY_BACKWARD_FUNCTIONS = [
#     (
#         "atan2_bw",
#         ttnn.operations.binary_backward,
#         r"""atan2_bw(input_tensor_1: ttnn.Tensor, input_tensor_2: ttnn.Tensor) -> ttnn.Tensor

#         Takes the atan2 backward of each element in input and returns a tensor with the result.

#         .. math::
#             pow(\mathrm{{input\_tensor}}_i, \mathrm{{exponent}})

#         Args:
#             * :attr:`input_tensor`
#             * :attr:`exponent`

#         Example::

#             >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
#             >>> output = ttnn.pow(tensor, 2)

#         """,
#     ),
# ]


# for binary_backward_function_name, ttl_binary_backward_function, doc in TTL_BINARY_BACKWARD_FUNCTIONS:
#     register_ttl_binary_backward_function(binary_backward_function_name, ttl_binary_backward_function, doc)


def _golden_function(grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    pyt_y = torch.atan2(input_tensor_a, input_tensor_b)
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()

    pyt_y.backward(gradient=grad_tensor)
    return


atan2_bw = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.binary_backward.atan2_bw)


def _golden_function(grad_tensor, input_tensor, weight_tensor, *args, **kwargs):
    import torch

    return


embedding_bw = ttnn.register_operation(golden_function=_golden_function)(
    ttnn._ttnn.operations.binary_backward.embedding_bw
)

__all__ = []
