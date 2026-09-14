# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import ttnn
from ttnn.operations.golden_common import (
    golden_compute_gradients,
    golden_pack_complex_gradient,
    golden_prepare_grad_inputs,
    golden_select_optional_outputs,
)


THIS_MODULE = sys.modules[__name__]

__all__ = []


def _complex_binary_backward(torch_op, grad_tensor, input_tensor_a, input_tensor_b, *, op_kwargs=None):
    """Compute a complex binary backward golden, packing each gradient as real|imag halves."""

    op_kwargs = op_kwargs or {}
    input_tensor_a, input_tensor_b = golden_prepare_grad_inputs(input_tensor_a, input_tensor_b)
    output = torch_op(input_tensor_a, input_tensor_b, **op_kwargs)
    gradients = golden_compute_gradients(output, (input_tensor_a, input_tensor_b), grad_tensor)
    return [golden_pack_complex_gradient(gradient) for gradient in gradients]


def _golden_function_backward(
    torch_op, grad_tensor, input_tensor_a, input_tensor_b, alpha=1.0, *args, are_required_outputs=None, **kwargs
):
    import torch

    if torch.is_complex(input_tensor_a):
        if torch_op == torch.add or torch_op == torch.sub:
            return _complex_binary_backward(
                torch_op, grad_tensor, input_tensor_a, input_tensor_b, op_kwargs={"alpha": alpha}
            )
        elif torch_op == torch.mul:
            return _complex_binary_backward(torch_op, grad_tensor, input_tensor_a, input_tensor_b)
    elif torch_op == torch.add or torch_op == torch.sub or torch_op == torch.mul:
        return _golden_function_backward_overload(
            torch_op, grad_tensor, input_tensor_a, input_tensor_b, are_required_outputs=are_required_outputs
        )
    if torch_op == "torch.squared_difference":
        pyt_y = torch.square(torch.sub(input_tensor_a, input_tensor_b))
    else:
        pyt_y = torch_op(input_tensor_a, input_tensor_b)
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


def _golden_function_backward_overload(
    torch_op, grad_tensor, input_tensor_a, input_tensor_b=None, *args, are_required_outputs=None, **kwargs
):
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
    if are_required_outputs is not None:
        return golden_select_optional_outputs(golden_tensor, are_required_outputs)
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
    import torch

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
    torch_op, grad_tensor, input_tensor_a, input_tensor_b, value=None, *args, are_required_outputs=None, **kwargs
):
    import torch

    if torch.is_complex(input_tensor_a):
        if torch_op == torch.div:
            return _complex_binary_backward(torch.div, grad_tensor, input_tensor_a, input_tensor_b)
    if torch_op == "bias_gelu_bw":
        sum_result = torch.add(input_tensor_a, input_tensor_b)
        pyt_y = torch.nn.functional.gelu(sum_result, approximate=value)
        sum_result.retain_grad()
        pyt_y.backward(gradient=grad_tensor)
        if isinstance(input_tensor_b, (float, int)):
            return [sum_result.grad]
        return [sum_result.grad, sum_result.grad]
    elif torch_op == torch.div:
        pyt_y = torch_op(input_tensor_a, input_tensor_b, rounding_mode=value)
    else:
        pyt_y = torch_op(input_tensor_a, input_tensor_b, value=value)
    if isinstance(input_tensor_b, (float, int)):
        input_tensor_a.retain_grad()
        pyt_y.backward(gradient=grad_tensor)
        return [input_tensor_a.grad]
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    if are_required_outputs is not None:
        return golden_select_optional_outputs(golden_tensor, are_required_outputs)
    return golden_tensor


def _golden_function_bw(grad, a, b, *args, **kwargs):
    import torch

    return _golden_function_backward(torch.sub, grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.sub_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, *args, **kwargs):
    import torch

    return _golden_function_backward(torch.add, grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.add_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, *args, **kwargs):
    import torch

    return _golden_function_backward_overload(torch.remainder, grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.remainder_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, *args, **kwargs):
    import torch

    return _golden_function_backward_overload(torch.fmod, grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.fmod_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, *args, **kwargs):
    import torch

    return _golden_function_backward(torch.atan2, grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.atan2_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, *args, **kwargs):
    import torch

    return _golden_function_backward(torch.xlogy, grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.xlogy_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, *args, **kwargs):
    import torch

    return _golden_function_backward(torch.hypot, grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.hypot_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, *args, **kwargs):
    import torch

    return _golden_function_backward(torch.ldexp, grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.ldexp_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, *args, **kwargs):
    import torch

    return _golden_function_backward(torch.logaddexp, grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.logaddexp_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, *args, **kwargs):
    import torch

    return _golden_function_backward(torch.logaddexp2, grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.logaddexp2_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, *args, **kwargs):
    return _golden_function_backward("torch.squared_difference", grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.squared_difference_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, alpha=None, *args, **kwargs):
    import torch

    return _golden_function_backward_with_float(torch.sub, grad, a, b, alpha, *args, **kwargs)


ttnn.attach_golden_function(ttnn.subalpha_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, alpha=None, *args, **kwargs):
    import torch

    return _golden_function_backward_with_float(torch.add, grad, a, b, alpha, *args, **kwargs)


ttnn.attach_golden_function(ttnn.addalpha_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b=None, *args, **kwargs):
    import torch

    return _golden_function_backward_overload(torch.clone, grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.assign_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, dim=None, *args, **kwargs):
    import torch

    return _golden_function_backward_with_dim(torch.concat, grad, a, b, dim, *args, **kwargs)


ttnn.attach_golden_function(ttnn.concat_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, *args, **kwargs):
    import torch

    return _golden_function_backward(torch.rsub, grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.rsub_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, variant=None, approximate=None, *args, **kwargs):
    import torch

    if approximate is None:
        approximate = kwargs.pop("value", variant)
    if approximate is None:
        approximate = "none"
    if isinstance(approximate, ttnn.GeluVariant):
        approximate = "tanh" if approximate == ttnn.GeluVariant.Tanh else "none"
    return _golden_function_backward_with_string("bias_gelu_bw", grad, a, b, approximate, *args, **kwargs)


ttnn.attach_golden_function(ttnn.bias_gelu_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, *args, **kwargs):
    import torch

    return _golden_function_backward(torch.min, grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.min_bw, golden_function=_golden_function_bw)


def _golden_function(grad, a, b, *args, **kwargs):
    import torch

    return _golden_function_backward(torch.max, grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.max_bw, golden_function=_golden_function)


def _golden_function_bw(grad, a, b, *args, rounding_mode=None, are_required_outputs=None, **kwargs):
    import torch

    return _golden_function_backward_with_string(
        torch.div, grad, a, b, rounding_mode, are_required_outputs=are_required_outputs
    )


ttnn.attach_golden_function(ttnn.div_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, *args, **kwargs):
    import torch

    return _golden_function_backward(torch.mul, grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.mul_bw, golden_function=_golden_function_bw)


__all__ = []
