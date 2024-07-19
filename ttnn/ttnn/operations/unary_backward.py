# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import ttnn

from typing import List, Union, Optional

import sys

import ttnn

import torch

THIS_MODULE = sys.modules[__name__]

__all__ = []


def _golden_function_unary_backward(torch_op, grad_tensor, input_tensor, *args, **kwargs):
    if torch_op == "softsign":
        pyt_y = torch.nn.functional.softsign(input_tensor)
    else:
        pyt_y = torch_op(input_tensor)
    input_tensor.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor.grad]
    return golden_tensor


def _golden_function_unary_backward_with_float(torch_op, grad_tensor, input_tensor, alpha, *args, **kwargs):
    if torch_op == "leaky_relu":
        pyt_y = torch.nn.functional.leaky_relu(input_tensor, negative_slope=alpha, inplace=False)
    elif torch_op == "elu":
        pyt_y = torch.nn.functional.elu(input_tensor, alpha=alpha)
    elif torch_op == "celu":
        pyt_y = torch.nn.functional.celu(input_tensor, alpha)
    elif torch_op == "div_no_nan":
        pyt_y = torch.where(torch.tensor(alpha) == 0, torch.zeros_like(input_tensor), torch.div(input_tensor, alpha))
    else:
        pyt_y = torch_op(input_tensor, alpha)
    input_tensor.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor.grad]
    if torch_op == "div_no_nan":
        golden_tensor[0] = torch.where(torch.isnan(golden_tensor[0]), torch.zeros_like(input_tensor), golden_tensor[0])
    return golden_tensor


def _golden_function_unary_backward_with_two_float(torch_op, grad_tensor, input_tensor, a, b, *args, **kwargs):
    if torch_op == torch.clamp:
        pyt_y = torch.clamp(input_tensor, min=a, max=b)
    else:
        pyt_y = torch_op(input_tensor, a, b)
    input_tensor.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor.grad]
    return golden_tensor


ttnn.attach_golden_function(
    ttnn.atanh_bw,
    golden_function=lambda grad, input, *args, **kwargs: _golden_function_unary_backward(
        torch.atanh, grad, input, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.asin_bw,
    golden_function=lambda grad, input, *args, **kwargs: _golden_function_unary_backward(
        torch.asin, grad, input, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.asinh_bw,
    golden_function=lambda grad, input, *args, **kwargs: _golden_function_unary_backward(
        torch.asinh, grad, input, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.sin_bw,
    golden_function=lambda grad, input, *args, **kwargs: _golden_function_unary_backward(
        torch.sin, grad, input, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.sinh_bw,
    golden_function=lambda grad, input, *args, **kwargs: _golden_function_unary_backward(
        torch.sinh, grad, input, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.log10_bw,
    golden_function=lambda grad, input, *args, **kwargs: _golden_function_unary_backward(
        torch.log10, grad, input, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.log1p_bw,
    golden_function=lambda grad, input, *args, **kwargs: _golden_function_unary_backward(
        torch.log1p, grad, input, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.erfc_bw,
    golden_function=lambda grad, input, *args, **kwargs: _golden_function_unary_backward(
        torch.erfc, grad, input, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.ceil_bw,
    golden_function=lambda grad, input, *args, **kwargs: _golden_function_unary_backward(
        torch.ceil, grad, input, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.softsign_bw,
    golden_function=lambda grad, input, *args, **kwargs: _golden_function_unary_backward(
        "softsign", grad, input, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.hardshrink_bw,
    golden_function=lambda grad, input, *args, **kwargs: _golden_function_unary_backward(
        torch.hardshrink, grad, input, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.softshrink_bw,
    golden_function=lambda grad, input, *args, **kwargs: _golden_function_unary_backward(
        torch.softshrink, grad, input, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.leaky_relu_bw,
    golden_function=lambda grad, input, alpha, *args, **kwargs: _golden_function_unary_backward_with_float(
        "leaky_relu", grad, input, alpha, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.elu_bw,
    golden_function=lambda grad, input, alpha, *args, **kwargs: _golden_function_unary_backward_with_float(
        "elu", grad, input, alpha, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.celu_bw,
    golden_function=lambda grad, input, alpha, *args, **kwargs: _golden_function_unary_backward_with_float(
        "celu", grad, input, alpha, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.rpow_bw,
    golden_function=lambda grad, input, alpha, *args, **kwargs: _golden_function_unary_backward_with_float(
        torch.pow, grad, input, alpha, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.logiteps_bw,
    golden_function=lambda grad, input, alpha, *args, **kwargs: _golden_function_unary_backward_with_float(
        torch.logit, grad, input, alpha, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.cosh_bw,
    golden_function=lambda grad, input, *args, **kwargs: _golden_function_unary_backward(
        torch.cosh, grad, input, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.sign_bw,
    golden_function=lambda grad, input, *args, **kwargs: _golden_function_unary_backward(
        torch.sign, grad, input, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.log2_bw,
    golden_function=lambda grad, input, *args, **kwargs: _golden_function_unary_backward(
        torch.log2, grad, input, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.div_no_nan_bw,
    golden_function=lambda grad, input, alpha, *args, **kwargs: _golden_function_unary_backward_with_float(
        "div_no_nan", grad, input, alpha, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.clamp_bw,
    golden_function=lambda grad, input, a, b, *args, **kwargs: _golden_function_unary_backward_with_two_float(
        torch.clamp, grad, input, a, b, *args, **kwargs
    ),
)

__all__ = []
