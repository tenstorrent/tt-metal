# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import ttnn


def register_ttnn_cpp_unary_backward_function(unary_backward_function):
    import torch

    def unary_bw_with_float(torch_op, x, grad_data, *args, **kwargs):
        value = kwargs.pop("scalar")
        x.retain_grad()

        pyt_y = torch_op(x, value)

        pyt_y.backward(gradient=grad_data)

        golden_tensor = [x.grad, x.grad]

        return golden_tensor

    def unary_bw(torch_op, x, grad_data, *args, **kwargs):
        x.retain_grad()

        pyt_y = torch_op(x)

        pyt_y.backward(gradient=grad_data)

        golden_tensor = [x.grad, x.grad]

        return golden_tensor

    def unary_bw_prod(torch_op, x, grad_data, *args, **kwargs):
        all_dimensions = kwargs.pop("all_dimensions", True)
        dim = kwargs.pop("dim", 0)
        if all_dimensions:
            temp = torch.prod(x)
            result = temp.view(1, 1, 1, 1)
        else:
            result = torch.prod(x, dim, keepdim=True)
        x.retain_grad()
        pyt_y.backward(gradient=grad_data)
        golden_tensor = [in_data.grad]
        return golden_tensor

    name_to_golden_function = {
        "mul_bw": lambda x, grad_data: unary_bw_with_float(torch.mul, x, grad_data),
        "clamp_min_bw": lambda x, grad_data: unary_bw_with_float(torch.clamp_min, x, grad_data),
        "add_bw": lambda x, grad_data: unary_bw_with_float(torch.add, x, grad_data),
        "eq_bw": lambda x, grad_data: unary_bw_with_float(torch.eq, x, grad_data),
        "gt_bw": lambda x, grad_data: unary_bw_with_float(torch.gt, x, grad_data),
        "lt_bw": lambda x, grad_data: unary_bw_with_float(torch.lt, x, grad_data),
        "le_bw": lambda x, grad_data: unary_bw_with_float(torch.le, x, grad_data),
        "ge_bw": lambda x, grad_data: unary_bw_with_float(torch.ge, x, grad_data),
        "ne_bw": lambda x, grad_data: unary_bw_with_float(torch.ne, x, grad_data),
        "sub_bw": lambda x, grad_data: unary_bw_with_float(torch.sub, x, grad_data),
        "hardshrink_bw": lambda x, grad_data: unary_bw_with_float(torch.nn.functional.hardshrink, x, grad_data),
        "softshrink_bw": lambda x, grad_data: unary_bw_with_float(torch.nn.functional.softshrink, x, grad_data),
        "leaky_relu_bw": lambda x, grad_data: unary_bw_with_float(torch.nn.functional.leaky_relu, x, grad_data),
        "elu_bw": lambda x, grad_data: unary_bw_with_float(torch.nn.functional.elu, x, grad_data),
        "celu_bw": lambda x, grad_data: unary_bw_with_float(torch.nn.functional.celu, x, grad_data),
        "rpow_bw": lambda x, grad_data: unary_bw_with_float(torch.pow, x, grad_data),
        "logiteps_bw": lambda x, grad_data: unary_bw_with_float(torch.nn.functional.logsigmoid, x, grad_data),
        "fmod_bw": lambda x, grad_data: unary_bw_with_float(torch.fmod, x, grad_data),
        "remainder_bw": lambda x, grad_data: unary_bw_with_float(torch.remainder, x, grad_data),
        "div_no_nan_bw": lambda x, grad_data: unary_bw_with_float(torch.div, x, grad_data),
        "polygamma_bw": lambda x, grad_data: unary_bw_with_float(torch.polygamma, x, grad_data),
        "assign_bw": lambda x, grad_data: unary_bw(torch.assign, x, grad_data),
        "multigammaln_bw": lambda x, grad_data: unary_bw(torch.multigammaln, x, grad_data),
        "lgamma_bw": lambda x, grad_data: unary_bw(torch.lgamma, x, grad_data),
        "fill_bw": lambda x, grad_data: unary_bw(torch.fill, x, grad_data),
        "hardsigmoid_bw": lambda x, grad_data: unary_bw(torch.nn.functional.hardsigmoid, x, grad_data),
        "cos_bw": lambda x, grad_data: unary_bw(torch.cos, x, grad_data),
        "acosh_bw": lambda x, grad_data: unary_bw(torch.acosh, x, grad_data),
        "prod_bw": lambda x, grad_data: unary_bw_prod(torch.prod, x, grad_data),
        "acos_bw": lambda x, grad_data: unary_bw(torch.acos, x, grad_data),
        "atan_bw": lambda x, grad_data: unary_bw(torch.atan, x, grad_data),
        "rad2deg_bw": lambda x, grad_data: unary_bw(torch.rad2deg, x, grad_data),
        "frac_bw": lambda x, grad_data: unary_bw(torch.frac, x, grad_data),
        "trunc_bw": lambda x, grad_data: unary_bw(torch.trunc, x, grad_data),
        "log_sigmoid_bw": lambda x, grad_data: unary_bw(torch.nn.functional.logsigmoid, x, grad_data),
        "fill_zero_bw": lambda x, grad_data: unary_bw(torch.fill_zero, x, grad_data),
        "i0_bw": lambda x, grad_data: unary_bw(torch.i0, x, grad_data),
        "tan_bw": lambda x, grad_data: unary_bw(torch.tan, x, grad_data),
        "sigmoid_bw": lambda x, grad_data: unary_bw(torch.sigmoid, x, grad_data),
        "rsqrt_bw": lambda x, grad_data: unary_bw(torch.rsqrt, x, grad_data),
        "neg_bw": lambda x, grad_data: unary_bw(torch.neg, x, grad_data),
        "relu_bw": lambda x, grad_data: unary_bw(torch.relu, x, grad_data),
        "logit_bw": lambda x, grad_data: unary_bw(torch.nn.functional.logit, x, grad_data),
        "floor_bw": lambda x, grad_data: unary_bw(torch.floor, x, grad_data),
        "round_bw": lambda x, grad_data: unary_bw(torch.round, x, grad_data),
        "log_bw": lambda x, grad_data: unary_bw(torch.log, x, grad_data),
        "relu6_bw": lambda x, grad_data: unary_bw(torch.nn.functional.relu6, x, grad_data),
        "abs_bw": lambda x, grad_data: unary_bw(torch.abs, x, grad_data),
        "silu_bw": lambda x, grad_data: unary_bw(torch.nn.functional.silu, x, grad_data),
        "selu_bw": lambda x, grad_data: unary_bw(torch.nn.functional.selu, x, grad_data),
        "square_bw": lambda x, grad_data: unary_bw(torch.square, x, grad_data),
        "hardswish_bw": lambda x, grad_data: unary_bw(torch.nn.functional.hardswish, x, grad_data),
        "tanhshrink_bw": lambda x, grad_data: unary_bw(torch.nn.functional.tanhshrink, x, grad_data),
        "atanh_bw": lambda x, grad_data: unary_bw(torch.atanh, x, grad_data),
        "asin_bw": lambda x, grad_data: unary_bw(torch.asin, x, grad_data),
        "asinh_bw": lambda x, grad_data: unary_bw(torch.asinh, x, grad_data),
        "sin_bw": lambda x, grad_data: unary_bw(torch.sin, x, grad_data),
        "sinh_bw": lambda x, grad_data: unary_bw(torch.sinh, x, grad_data),
        "log10_bw": lambda x, grad_data: unary_bw(torch.log10, x, grad_data),
        "log1p_bw": lambda x, grad_data: unary_bw(torch.log1p, x, grad_data),
        "erfc_bw": lambda x, grad_data: unary_bw(torch.erfc, x, grad_data),
        "ceil_bw": lambda x, grad_data: unary_bw(torch.ceil, x, grad_data),
        "softsign_bw": lambda x, grad_data: unary_bw(torch.nn.functional.softsign, x, grad_data),
        "cosh_bw": lambda x, grad_data: unary_bw(torch.cosh, x, grad_data),
        "log2_bw": lambda x, grad_data: unary_bw(torch.log2, x, grad_data),
        "sign_bw": lambda x, grad_data: unary_bw(torch.sign, x, grad_data),
        "exp2_bw": lambda x, grad_data: unary_bw(torch.exp2, x, grad_data),
        "expm1_bw": lambda x, grad_data: unary_bw(torch.expm1, x, grad_data),
        "reciprocal_bw": lambda x, grad_data: unary_bw(torch.reciprocal, x, grad_data),
        "digamma_bw": lambda x, grad_data: unary_bw(torch.digamma, x, grad_data),
        "erfinv_bw": lambda x, grad_data: unary_bw(torch.erfinv, x, grad_data),
        "erf_bw": lambda x, grad_data: unary_bw(torch.erf, x, grad_data),
        "deg2rad_bw": lambda x, grad_data: unary_bw(torch.deg2rad, x, grad_data),
    }

    golden_keys = set(name_to_golden_function.keys())
    function_names = {function.__name__.split(".")[-1] for function in TTNN_ELTWISE_UNARY_BACKWARD_CPP_FUNCTIONS}
    if golden_keys != function_names:
        raise ImportError(f"Missing or extra golden functions:\n{golden_keys}\nshould be equal to\n{function_names}")

    def _golden_function(input_tensor: ttnn.Tensor, **_):
        torch_function = name_to_golden_function[unary_backward_function.__name__.split(".")[-1]]
        return torch_function(input_tensor)

    ttnn.attach_golden_function(unary_backward_function, golden_function=_golden_function)


TTNN_ELTWISE_UNARY_BACKWARD_CPP_FUNCTIONS = [
    ttnn.mul_bw,
    ttnn.clamp_min_bw,
    ttnn.add_bw,
    ttnn.eq_bw,
    ttnn.gt_bw,
    ttnn.lt_bw,
    ttnn.le_bw,
    ttnn.ge_bw,
    ttnn.ne_bw,
    ttnn.sub_bw,
    ttnn.hardshrink_bw,
    ttnn.softshrink_bw,
    ttnn.leaky_relu_bw,
    ttnn.elu_bw,
    ttnn.celu_bw,
    ttnn.rpow_bw,
    ttnn.logiteps_bw,
    ttnn.fmod_bw,
    ttnn.remainder_bw,
    ttnn.div_no_nan_bw,
    ttnn.polygamma_bw,
    ttnn.assign_bw,
    ttnn.multigammaln_bw,
    ttnn.lgamma_bw,
    ttnn.fill_bw,
    ttnn.hardsigmoid_bw,
    ttnn.cos_bw,
    ttnn.acosh_bw,
    ttnn.acos_bw,
    ttnn.atan_bw,
    ttnn.rad2deg_bw,
    ttnn.frac_bw,
    ttnn.trunc_bw,
    ttnn.log_sigmoid_bw,
    ttnn.fill_zero_bw,
    ttnn.i0_bw,
    ttnn.tan_bw,
    ttnn.sigmoid_bw,
    ttnn.rsqrt_bw,
    ttnn.neg_bw,
    ttnn.relu_bw,
    ttnn.logit_bw,
    ttnn.floor_bw,
    ttnn.round_bw,
    ttnn.log_bw,
    ttnn.relu6_bw,
    ttnn.abs_bw,
    ttnn.silu_bw,
    ttnn.selu_bw,
    ttnn.square_bw,
    ttnn.hardswish_bw,
    ttnn.tanhshrink_bw,
    ttnn.atanh_bw,
    ttnn.asin_bw,
    ttnn.asinh_bw,
    ttnn.sin_bw,
    ttnn.sinh_bw,
    ttnn.log10_bw,
    ttnn.log1p_bw,
    ttnn.erfc_bw,
    ttnn.ceil_bw,
    ttnn.softsign_bw,
    ttnn.cosh_bw,
    ttnn.log2_bw,
    ttnn.sign_bw,
    ttnn.exp2_bw,
    ttnn.expm1_bw,
    ttnn.reciprocal_bw,
    ttnn.digamma_bw,
    ttnn.erfinv_bw,
    ttnn.erf_bw,
    ttnn.deg2rad_bw,
    ttnn.prod_bw,
]
for unary_backward_function in TTNN_ELTWISE_UNARY_BACKWARD_CPP_FUNCTIONS:
    register_ttnn_cpp_unary_backward_function(unary_backward_function)

__all__ = []
