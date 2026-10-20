# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import ttnn
from ttnn.operations.golden_common import (
    golden_compute_gradients,
    golden_pack_complex_gradient,
    golden_prepare_grad_inputs,
    golden_select_optional_outputs,
    make_aliasing_preprocess,
)


THIS_MODULE = sys.modules[__name__]

__all__ = []

# The tensor overloads name their operands input_tensor and other_tensor, the scalar overloads name the second operand
# scalar, the scalar-bias bias_gelu overloads name it bias, and the single-operand assign overload names its operand
# input_tensor; the goldens take them as input_tensor_a and input_tensor_b.
_preprocess_binary_backward_golden_inputs = make_aliasing_preprocess(
    {
        "input_tensor": "input_tensor_a",
        "other_tensor": "input_tensor_b",
        "scalar": "input_tensor_b",
        "bias": "input_tensor_b",
    }
)


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
    pyt_y = torch_op(input_tensor_a, input_tensor_b)
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    return golden_select_optional_outputs([input_tensor_a.grad, input_tensor_b.grad], are_required_outputs)


def _golden_function_backward_overload(
    torch_op, grad_tensor, input_tensor_a, input_tensor_b=None, *args, are_required_outputs=None, **kwargs
):
    import torch

    if torch_op == torch.clone:
        pyt_y = torch.clone(input_tensor_a)
        input_tensor_a.retain_grad()
        pyt_y.backward(gradient=grad_tensor)
        if input_tensor_b is None:
            golden_tensor = [input_tensor_a.grad]
            return golden_tensor
        return golden_select_optional_outputs([input_tensor_a.grad, input_tensor_a.grad], are_required_outputs)
    pyt_y = torch_op(input_tensor_a, input_tensor_b)
    if isinstance(input_tensor_b, (float, int)):
        input_tensor_a.retain_grad()
        pyt_y.backward(gradient=grad_tensor)
        golden_tensor = [input_tensor_a.grad]
        return golden_tensor
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    return golden_select_optional_outputs([input_tensor_a.grad, input_tensor_b.grad], are_required_outputs)


def _golden_function_backward_with_dim(
    torch_op, grad_tensor, input_tensor_a, input_tensor_b, dimension=None, *args, are_required_outputs=None, **kwargs
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
    return golden_select_optional_outputs([input_tensor_a.grad, input_tensor_b.grad], are_required_outputs)


def _golden_function_backward_with_float(
    torch_op, grad_tensor, input_tensor_a, input_tensor_b, alpha=None, *args, are_required_outputs=None, **kwargs
):
    import torch

    if alpha == None:
        pyt_y = torch_op(input_tensor_a, input_tensor_b)
    else:
        pyt_y = torch_op(input_tensor_a, input_tensor_b, alpha=alpha)
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    return golden_select_optional_outputs([input_tensor_a.grad, input_tensor_b.grad], are_required_outputs)


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
    return golden_select_optional_outputs([input_tensor_a.grad, input_tensor_b.grad], are_required_outputs)


def _torch_squared_difference(input_tensor_a, input_tensor_b):
    import torch

    return torch.square(torch.sub(input_tensor_a, input_tensor_b))


def _make_binary_bw_golden(torch_op, reference=_golden_function_backward):
    """Return a binary backward golden that differentiates torch_op through reference.
    torch_op is a callable or the name of a Torch function, resolved at call time because PyTorch is optional.
    """

    def golden_function(grad_tensor, input_tensor_a, input_tensor_b=None, *args, **kwargs):
        import torch

        torch_function = getattr(torch, torch_op) if isinstance(torch_op, str) else torch_op
        return reference(torch_function, grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs)

    return golden_function


for _operation, _torch_op, _reference in (
    (ttnn.add_bw, "add", _golden_function_backward),
    (ttnn.sub_bw, "sub", _golden_function_backward),
    (ttnn.mul_bw, "mul", _golden_function_backward),
    (ttnn.atan2_bw, "atan2", _golden_function_backward),
    (ttnn.xlogy_bw, "xlogy", _golden_function_backward),
    (ttnn.hypot_bw, "hypot", _golden_function_backward),
    (ttnn.ldexp_bw, "ldexp", _golden_function_backward),
    (ttnn.logaddexp_bw, "logaddexp", _golden_function_backward),
    (ttnn.logaddexp2_bw, "logaddexp2", _golden_function_backward),
    (ttnn.squared_difference_bw, _torch_squared_difference, _golden_function_backward),
    (ttnn.rsub_bw, "rsub", _golden_function_backward),
    (ttnn.min_bw, "min", _golden_function_backward),
    (ttnn.max_bw, "max", _golden_function_backward),
    (ttnn.remainder_bw, "remainder", _golden_function_backward_overload),
    (ttnn.fmod_bw, "fmod", _golden_function_backward_overload),
    (ttnn.assign_bw, "clone", _golden_function_backward_overload),
    (ttnn.subalpha_bw, "sub", _golden_function_backward_with_float),
    (ttnn.addalpha_bw, "add", _golden_function_backward_with_float),
):
    ttnn.attach_golden_function(
        _operation,
        golden_function=_make_binary_bw_golden(_torch_op, _reference),
        preprocess_golden_function_inputs=_preprocess_binary_backward_golden_inputs,
    )


def _golden_function_bw(grad_tensor, input_tensor_a, input_tensor_b, dim=None, *args, **kwargs):
    import torch

    return _golden_function_backward_with_dim(
        torch.concat, grad_tensor, input_tensor_a, input_tensor_b, dim, *args, **kwargs
    )


ttnn.attach_golden_function(
    ttnn.concat_bw,
    golden_function=_golden_function_bw,
    preprocess_golden_function_inputs=_preprocess_binary_backward_golden_inputs,
)


def _golden_function_bw(grad_tensor, input_tensor_a, input_tensor_b, variant=None, approximate=None, *args, **kwargs):
    if approximate is None:
        approximate = kwargs.pop("value", variant)
    if approximate is None:
        approximate = "none"
    if isinstance(approximate, ttnn.GeluVariant):
        approximate = "tanh" if approximate == ttnn.GeluVariant.Tanh else "none"
    return _golden_function_backward_with_string(
        "bias_gelu_bw", grad_tensor, input_tensor_a, input_tensor_b, approximate, *args, **kwargs
    )


ttnn.attach_golden_function(
    ttnn.bias_gelu_bw,
    golden_function=_golden_function_bw,
    preprocess_golden_function_inputs=_preprocess_binary_backward_golden_inputs,
)


def _golden_function_bw(
    grad_tensor, input_tensor_a, input_tensor_b, rounding_mode=None, *args, are_required_outputs=None, **kwargs
):
    import torch

    # The scalar overload accepts rounding_mode positionally after the scalar; the tensor overload only by keyword.
    return _golden_function_backward_with_string(
        torch.div,
        grad_tensor,
        input_tensor_a,
        input_tensor_b,
        rounding_mode,
        are_required_outputs=are_required_outputs,
    )


ttnn.attach_golden_function(
    ttnn.div_bw,
    golden_function=_golden_function_bw,
    preprocess_golden_function_inputs=_preprocess_binary_backward_golden_inputs,
)


__all__ = []
