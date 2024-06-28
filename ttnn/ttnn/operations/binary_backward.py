# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional

import sys

import ttnn

import torch

THIS_MODULE = sys.modules[__name__]

__all__ = []


def _golden_function(grad_tensor, input_tensor, weight_tensor, *args, **kwargs):
    import torch

    batch_size = input_tensor.shape[0]
    no_of_embeddings = input_tensor.shape[1] * input_tensor.shape[2]
    embedding_dim = input_tensor.shape[3]

    input_shape = [batch_size, 1, 1, no_of_embeddings]
    input_index = torch.reshape(torch.arange(0, batch_size * no_of_embeddings), shape=input_shape)
    weights_shape = [batch_size, 1, no_of_embeddings, embedding_dim]
    weights = torch.randn(weights_shape, requires_grad=True)
    grad_shape = [1, 1, batch_size * no_of_embeddings, embedding_dim]
    grad_data = torch.randn(grad_shape, requires_grad=True)

    weights.retain_grad()

    pyt_y = torch.nn.functional.embedding(
        input_index.reshape((batch_size, no_of_embeddings)),
        weights.reshape((batch_size * no_of_embeddings, embedding_dim)),
    ).reshape((1, 1, batch_size * no_of_embeddings, embedding_dim))

    pyt_y.backward(gradient=grad_data)

    golden_output_tensor_a = weights.grad

    return golden_output_tensor_a


embedding_bw = ttnn.register_operation(golden_function=_golden_function)(
    ttnn._ttnn.operations.binary_backward.embedding_bw
)


def _golden_function_backward(torch_op, grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    if torch_op == torch.squared_difference:
        pyt_y = torch.square(torch.sub(input_tensor_a, input_tensor_b))
    elif torch_op == torch.clone:
        pyt_y = torch.clone(input_tensor_a)
        input_tensor_a.retain_grad()
        pyt_y.backward(gradient=grad_tensor)
        golden_tensor = [input_tensor_a.grad]
        return golden_tensor
    else:
        pyt_y = torch_op(input_tensor_a, input_tensor_b)
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


def _golden_function_backward_with_float(torch_op, grad_tensor, input_tensor_a, input_tensor_b, alpha, *args, **kwargs):
    pyt_y = torch_op(input_tensor_a, input_tensor_b, alpha=alpha)
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


def _golden_function_comparison_ops(torch_op, grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    golden_tensor = [torch.zeros_like(input_tensor_a), torch.zeros_like(input_tensor_b)]
    return golden_tensor


sub_bw = ttnn.register_operation(
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.sub, grad, a, b, *args, **kwargs
    )
)(ttnn._ttnn.operations.binary_backward.sub_bw)

add_bw = ttnn.register_operation(
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.add, grad, a, b, *args, **kwargs
    )
)(ttnn._ttnn.operations.binary_backward.add_bw)

atan2_bw = ttnn.register_operation(
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.atan2, grad, a, b, *args, **kwargs
    )
)(ttnn._ttnn.operations.binary_backward.atan2_bw)

xlogy_bw = ttnn.register_operation(
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.xlogy, grad, a, b, *args, **kwargs
    )
)(ttnn._ttnn.operations.binary_backward.xlogy_bw)

hypot_bw = ttnn.register_operation(
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.hypot, grad, a, b, *args, **kwargs
    )
)(ttnn._ttnn.operations.binary_backward.hypot_bw)

ldexp_bw = ttnn.register_operation(
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.ldexp, grad, a, b, *args, **kwargs
    )
)(ttnn._ttnn.operations.binary_backward.ldexp_bw)

logaddexp_bw = ttnn.register_operation(
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.logaddexp, grad, a, b, *args, **kwargs
    )
)(ttnn._ttnn.operations.binary_backward.logaddexp_bw)

logaddexp2_bw = ttnn.register_operation(
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.logaddexp2, grad, a, b, *args, **kwargs
    )
)(ttnn._ttnn.operations.binary_backward.logaddexp2_bw)

squared_difference_bw = ttnn.register_operation(
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.squared_difference, grad, a, b, *args, **kwargs
    )
)(ttnn._ttnn.operations.binary_backward.squared_difference_bw)

subalpha_bw = ttnn.register_operation(
    golden_function=lambda grad, a, b, alpha, *args, **kwargs: _golden_function_backward_with_float(
        torch.sub, grad, a, b, alpha, *args, **kwargs
    )
)(ttnn._ttnn.operations.binary_backward.subalpha_bw)

addalpha_bw = ttnn.register_operation(
    golden_function=lambda grad, a, b, alpha, *args, **kwargs: _golden_function_backward_with_float(
        torch.add, grad, a, b, alpha, *args, **kwargs
    )
)(ttnn._ttnn.operations.binary_backward.addalpha_bw)

binary_eq_bw = ttnn.register_operation(
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_comparison_ops(
        torch.eq, grad, a, b, *args, **kwargs
    )
)(ttnn._ttnn.operations.binary_backward.binary_eq_bw)

binary_assign_bw = ttnn.register_operation(
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.clone, grad, a, b, *args, **kwargs
    )
)(ttnn._ttnn.operations.binary_backward.binary_assign_bw)

concat_bw = ttnn.register_operation(
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.clone, grad, a, b, *args, **kwargs
    )
)(ttnn._ttnn.operations.binary_backward.concat_bw)

binary_le_bw = ttnn.register_operation(
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.clone, grad, a, b, *args, **kwargs
    )
)(ttnn._ttnn.operations.binary_backward.binary_le_bw)

__all__ = []
