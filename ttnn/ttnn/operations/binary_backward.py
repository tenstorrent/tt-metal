# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import ttnn
from ttnn.operations.complex_binary_backward import (
    _golden_function_complex_add,
    _golden_function_complex_sub,
    _golden_function_complex_mul,
    _golden_function_complex_div,
)

import torch

THIS_MODULE = sys.modules[__name__]

__all__ = []


def _golden_function_backward(torch_op, grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if torch.is_complex(input_tensor_a):
        if torch_op == torch.add:
            alpha = kwargs.pop("alpha")
            return _golden_function_complex_add(grad_tensor, input_tensor_a, input_tensor_b, alpha)
        elif torch_op == torch.sub:
            alpha = kwargs.pop("alpha")
            return _golden_function_complex_sub(grad_tensor, input_tensor_a, input_tensor_b, alpha)
        elif torch_op == torch.mul:
            return _golden_function_complex_mul(grad_tensor, input_tensor_a, input_tensor_b)
    elif torch_op == torch.add or torch_op == torch.sub or torch_op == torch.mul:
        return _golden_function_backward_overload(torch_op, grad_tensor, input_tensor_a, input_tensor_b)
    if torch_op == "torch.squared_difference":
        pyt_y = torch.square(torch.sub(input_tensor_a, input_tensor_b))
    else:
        pyt_y = torch_op(input_tensor_a, input_tensor_b)
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


def _golden_function_backward_overload(torch_op, grad_tensor, input_tensor_a, input_tensor_b=None, *args, **kwargs):
    import torch

    if torch_op == torch.clone:
        pyt_y = torch.clone(input_tensor_a)
        input_tensor_a.retain_grad()
        pyt_y.backward(gradient=grad_tensor)
        if input_tensor_b == None:
            golden_tensor = [input_tensor_a.grad]
            return golden_tensor
        else:
            golden_tensor = [input_tensor_a.grad, input_tensor_a.grad]
            return golden_tensor
    pyt_y = torch_op(input_tensor_a, input_tensor_b)
    if isinstance(input_tensor_b, (float, int)):
        input_tensor_a.retain_grad()
        pyt_y.backward(gradient=grad_tensor)
        golden_tensor = [input_tensor_a.grad]
        return golden_tensor
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


def _golden_function_backward_with_dim(
    torch_op, grad_tensor, input_tensor_a, input_tensor_b, dimension=None, *args, **kwargs
):
    import torch

    if input_tensor_a.requires_grad is False:
        input_tensor_a.requires_grad = True
    if input_tensor_b.requires_grad is False:
        input_tensor_b.requires_grad = True

    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()

    if dimension == None:
        pyt_y = torch.concat((input_tensor_a, input_tensor_b))
    else:
        pyt_y = torch.concat((input_tensor_a, input_tensor_b), dim=dimension)
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


def _golden_function_backward_with_float(
    torch_op, grad_tensor, input_tensor_a, input_tensor_b, alpha=None, *args, **kwargs
):
    if alpha == None:
        pyt_y = torch_op(input_tensor_a, input_tensor_b)
    else:
        pyt_y = torch_op(input_tensor_a, input_tensor_b, alpha=alpha)
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


def _golden_function_backward_with_string(
    torch_op, grad_tensor, input_tensor_a, input_tensor_b, value=None, *args, **kwargs
):
    import torch

    if torch.is_complex(input_tensor_a):
        if torch_op == torch.div:
            return _golden_function_complex_div(grad_tensor, input_tensor_a, input_tensor_b)
    if torch_op == "bias_gelu_bw":
        sum_result = torch.add(input_tensor_a, input_tensor_b)
        pyt_y = torch.nn.functional.gelu(sum_result, approximate=value)
        sum_result.retain_grad()
        pyt_y.backward(gradient=grad_tensor)
        if isinstance(input_tensor_b, (float, int)):
            golden_tensor = [sum_result.grad]
        else:
            golden_tensor = [sum_result.grad, sum_result.grad]
        return golden_tensor
    elif torch_op == torch.div:
        pyt_y = torch_op(input_tensor_a, input_tensor_b, rounding_mode=value)
    else:
        pyt_y = torch_op(input_tensor_a, input_tensor_b, value=value)
    if isinstance(input_tensor_b, (float, int)):
        input_tensor_a.retain_grad()
        pyt_y.backward(gradient=grad_tensor)
        golden_tensor = [input_tensor_a.grad]
        return golden_tensor
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


def _golden_function_comparison_ops(torch_op, grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if isinstance(input_tensor_b, (float, int)):
        golden_tensor = [torch.zeros_like(input_tensor_a)]
    else:
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
    ttnn.remainder_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward_overload(
        torch.remainder, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.fmod_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward_overload(
        torch.fmod, grad, a, b, *args, **kwargs
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
        "torch.squared_difference", grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.subalpha_bw,
    golden_function=lambda grad, a, b, alpha=None, *args, **kwargs: _golden_function_backward_with_float(
        torch.sub, grad, a, b, alpha, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.addalpha_bw,
    golden_function=lambda grad, a, b, alpha=None, *args, **kwargs: _golden_function_backward_with_float(
        torch.add, grad, a, b, alpha, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.assign_bw,
    golden_function=lambda grad, a, b=None, *args, **kwargs: _golden_function_backward_overload(
        torch.clone, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.concat_bw,
    golden_function=lambda grad, a, b, dim=None, *args, **kwargs: _golden_function_backward_with_dim(
        torch.concat, grad, a, b, dim, *args, **kwargs
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
    golden_function=lambda grad, a, b, value="none", *args, **kwargs: _golden_function_backward_with_string(
        "bias_gelu_bw", grad, a, b, value, *args, **kwargs
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
    golden_function=lambda grad, a, b, value=None, *args, **kwargs: _golden_function_backward_with_string(
        torch.div, grad, a, b, value, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.mul_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.mul, grad, a, b, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.fmod_bw,
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward_overload(
        torch.fmod, grad, a, b, *args, **kwargs
    ),
)


__all__ = []
