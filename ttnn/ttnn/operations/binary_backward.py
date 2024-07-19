# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional

import sys

import ttnn

import torch

THIS_MODULE = sys.modules[__name__]

__all__ = []


def _golden_function(grad_tensor, input_index, weights, input_shapes, *args, **kwargs):
    import torch

    batch_size = input_shapes[0]
    no_of_embeddings = input_shapes[1] * input_shapes[2]
    embedding_dim = input_shapes[3]

    weights.retain_grad()

    pyt_y = torch.nn.functional.embedding(
        input_index.reshape((batch_size, no_of_embeddings)),
        weights.reshape((batch_size * no_of_embeddings, embedding_dim)),
    ).reshape((1, 1, batch_size * no_of_embeddings, embedding_dim))

    pyt_y.backward(gradient=grad_tensor)

    golden_output_tensor_a = weights.grad

    return golden_output_tensor_a


ttnn.attach_golden_function(ttnn.embedding_bw, golden_function=_golden_function)


def _golden_function_backward(torch_op, grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    if torch_op == "squared_difference_bw":
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


def _golden_function_backward_with_dim(
    torch_op, grad_tensor, input_tensor_a, input_tensor_b, dimension, *args, **kwargs
):
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()
    pyt_y = torch.cat((input_tensor_a, input_tensor_b), dim=dimension)
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


def _golden_function_backward_with_string(
    torch_op, grad_tensor, input_tensor_a, input_tensor_b, value, *args, **kwargs
):
    if torch_op == "bias_gelu":
        sum_result = torch.add(input_tensor_a, input_tensor_b)
        pyt_y = torch.nn.functional.gelu(sum_result)
        sum_result.retain_grad()
        pyt_y.backward(gradient=grad_tensor)
        golden_tensor = [sum_result.grad, sum_result.grad]
        return golden_tensor

    if torch_op == torch.div:
        pyt_y = torch_op(input_tensor_a, input_tensor_b, rounding_mode=value)
    else:
        pyt_y = torch_op(input_tensor_a, input_tensor_b, value=value)
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


def _golden_function_comparison_ops(torch_op, grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    golden_tensor = [torch.zeros_like(input_tensor_a), torch.zeros_like(input_tensor_b)]
    return golden_tensor


ttnn.attach_golden_function(
    ttnn.sub_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.sub, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.add_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.add, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.atan2_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.atan2, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.xlogy_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.xlogy, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.hypot_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.hypot, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.ldexp_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.ldexp, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.logaddexp_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.logaddexp, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.logaddexp2_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.logaddexp2, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.squared_difference_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        "squared_difference_bw", grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.subalpha_bw,
    golden_function=lambda grad, a, b, alpha, *args, **kwargs: _golden_function_backward_with_float(
        torch.sub, grad, a, b, alpha, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.addalpha_bw,
    golden_function=lambda grad, a, b, alpha, *args, **kwargs: _golden_function_backward_with_float(
        torch.add, grad, a, b, alpha, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.eq_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_comparison_ops(
        torch.eq, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.assign_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.clone, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.concat_bw,
    golden_function=lambda grad, a, b, dimension, *args, **kwargs: _golden_function_backward_with_dim(
        torch.cat, grad, a, b, dimension, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.le_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_comparison_ops(
        torch.le, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.rsub_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.rsub, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.bias_gelu_bw,
    golden_function=lambda grad, a, b, value, *args, **kwargs: _golden_function_backward_with_string(
        torch.gelu, grad, a, b, value, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.gt_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_comparison_ops(
        torch.gt, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.lt_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_comparison_ops(
        torch.gt, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.ne_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_comparison_ops(
        torch.ne, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.ge_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_comparison_ops(
        torch.ge, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.min_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.min, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.max_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.max, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.div_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward_with_string(
        torch.div, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.lerp_bw,
    golden_function=lambda grad, a, b, weight, *args, **kwargs: _golden_function_backward_with_float(
        torch.add, grad, a, b, weight, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.mul_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.mul, grad, a, b, *args, **kwargs
    ),
)


__all__ = []
