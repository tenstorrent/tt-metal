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

_MISSING = object()


def _binary_backward_reference(
    torch_op,
    grad_tensor,
    input_tensor_a,
    input_tensor_b,
    *,
    op_kwargs=None,
    are_required_outputs=None,
):
    import torch

    op_kwargs = op_kwargs or {}
    if isinstance(input_tensor_b, (float, int)):
        (input_tensor_a,) = golden_prepare_grad_inputs(input_tensor_a)
        output = torch_op(input_tensor_a, input_tensor_b, **op_kwargs)
        return golden_compute_gradients(output, (input_tensor_a,), grad_tensor)

    input_tensor_a, input_tensor_b = golden_prepare_grad_inputs(input_tensor_a, input_tensor_b)
    output = torch_op(input_tensor_a, input_tensor_b, **op_kwargs)
    gradients = golden_compute_gradients(output, (input_tensor_a, input_tensor_b), grad_tensor)
    if torch.is_complex(input_tensor_a):
        return [golden_pack_complex_gradient(gradient) for gradient in gradients]
    if are_required_outputs is not None:
        return golden_select_optional_outputs(gradients, are_required_outputs)
    return gradients


def _resolve_alpha(input_tensor, args, alpha):
    import torch

    if torch.is_complex(input_tensor) and args:
        return args[0]
    return alpha


def _resolve_binary_backward_inputs(grad, input_tensor_a, input_tensor_b, kwargs):
    if grad is _MISSING:
        grad = kwargs.pop("grad_tensor")
    if input_tensor_a is _MISSING:
        for argument_name in ("input_tensor_a", "input_tensor"):
            if argument_name in kwargs:
                input_tensor_a = kwargs.pop(argument_name)
                break
    if input_tensor_b is _MISSING:
        for argument_name in ("input_tensor_b", "other_tensor", "scalar"):
            if argument_name in kwargs:
                input_tensor_b = kwargs.pop(argument_name)
                break
    if input_tensor_a is _MISSING or input_tensor_b is _MISSING:
        raise TypeError("Binary backward golden requires two input operands")
    return grad, input_tensor_a, input_tensor_b


def _golden_function_backward(torch_op, grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if torch_op == torch.add or torch_op == torch.sub or torch_op == torch.mul:
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
    torch_op, grad_tensor, input_tensor_a, input_tensor_b, value=None, *args, **kwargs
):
    import torch

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


def _golden_sub_bw(
    grad=_MISSING,
    a=_MISSING,
    b=_MISSING,
    *args,
    alpha=1.0,
    are_required_outputs=(True, True),
    **kwargs,
):
    import torch

    grad, a, b = _resolve_binary_backward_inputs(grad, a, b, kwargs)
    alpha = _resolve_alpha(a, args, alpha)
    op_kwargs = {"alpha": alpha} if torch.is_complex(a) else None
    return _binary_backward_reference(
        torch.sub,
        grad,
        a,
        b,
        op_kwargs=op_kwargs,
        are_required_outputs=are_required_outputs,
    )


ttnn.attach_golden_function(ttnn.sub_bw, golden_function=_golden_sub_bw)


def _golden_add_bw(
    grad=_MISSING,
    a=_MISSING,
    b=_MISSING,
    *args,
    alpha=1.0,
    are_required_outputs=(True, True),
    **kwargs,
):
    import torch

    grad, a, b = _resolve_binary_backward_inputs(grad, a, b, kwargs)
    alpha = _resolve_alpha(a, args, alpha)
    op_kwargs = {"alpha": alpha} if torch.is_complex(a) else None
    return _binary_backward_reference(
        torch.add,
        grad,
        a,
        b,
        op_kwargs=op_kwargs,
        are_required_outputs=are_required_outputs,
    )


ttnn.attach_golden_function(ttnn.add_bw, golden_function=_golden_add_bw)


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


def _golden_function_bw(grad, a, b, value="none", *args, **kwargs):
    import torch

    return _golden_function_backward_with_string("bias_gelu_bw", grad, a, b, value, *args, **kwargs)


ttnn.attach_golden_function(ttnn.bias_gelu_bw, golden_function=_golden_function_bw)


def _golden_function_bw(grad, a, b, *args, **kwargs):
    import torch

    return _golden_function_backward(torch.min, grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.min_bw, golden_function=_golden_function_bw)


def _golden_function(grad, a, b, *args, **kwargs):
    import torch

    return _golden_function_backward(torch.max, grad, a, b, *args, **kwargs)


ttnn.attach_golden_function(ttnn.max_bw, golden_function=_golden_function)


def _golden_div_bw(
    grad=_MISSING,
    a=_MISSING,
    b=_MISSING,
    *args,
    value=None,
    rounding_mode=None,
    are_required_outputs=(True, True),
    **kwargs,
):
    import torch

    grad, a, b = _resolve_binary_backward_inputs(grad, a, b, kwargs)
    if args:
        rounding_mode = args[0]
    if value is not None:
        rounding_mode = value
    op_kwargs = None if torch.is_complex(a) else {"rounding_mode": rounding_mode}
    return _binary_backward_reference(
        torch.div,
        grad,
        a,
        b,
        op_kwargs=op_kwargs,
        are_required_outputs=are_required_outputs,
    )


ttnn.attach_golden_function(ttnn.div_bw, golden_function=_golden_div_bw)


def _golden_mul_bw(
    grad=_MISSING,
    a=_MISSING,
    b=_MISSING,
    *args,
    are_required_outputs=(True, True),
    **kwargs,
):
    import torch

    grad, a, b = _resolve_binary_backward_inputs(grad, a, b, kwargs)
    return _binary_backward_reference(
        torch.mul,
        grad,
        a,
        b,
        are_required_outputs=are_required_outputs,
    )


ttnn.attach_golden_function(ttnn.mul_bw, golden_function=_golden_mul_bw)


__all__ = []
