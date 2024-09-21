# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

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


def _golden_function_backward_with_reverse_string(
    torch_op, grad_tensor, input_tensor_a, input_tensor_b, value=None, *args, **kwargs
):
    if torch_op == torch.div:
        pyt_y = torch_op(input_tensor_b, input_tensor_a, rounding_mode=value)
    else:
        pyt_y = torch_op(input_tensor_b, input_tensor_a)
    input_tensor_a.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad]
    return golden_tensor


ttnn.attach_golden_function(
    ttnn.atanh_bw,
    golden_function=lambda grad, input, *args, **kwargs: _golden_function_unary_backward(
        torch.atanh, grad, input, *args, **kwargs
    ),
)

ttnn.attach_golden_function(
    ttnn.rdiv_bw,
    golden_function=lambda grad, input, value=None, *args, **kwargs: _golden_function_backward_with_reverse_string(
        torch.div, grad, input, value, *args, **kwargs
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


def _golden_function_abs_cmplx(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()

    pyt_y = torch.abs(input_tensor)

    pyt_y.backward(gradient=grad_tensor)

    grad_res_real = torch.real(input_tensor.grad)
    grad_res_imag = torch.imag(input_tensor.grad)

    golden_tensor = [torch.cat((grad_res_real, grad_res_imag), dim=-1)]

    return golden_tensor


def _golden_function(grad_tensor, input_tensor, min_val=None, max_val=None, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    if min_val != None and max_val != None:
        pyt_y = torch.nn.functional.hardtanh(input_tensor, min_val, max_val)
    else:
        pyt_y = torch.nn.functional.hardtanh(input_tensor)
    pyt_y.backward(gradient=grad_tensor)

    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.hardtanh_bw, golden_function=_golden_function)


def _golden_function(grad_tensor, input_tensor, beta=None, threshold=None, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    if beta != None and threshold != None:
        pyt_y = torch.nn.functional.softplus(input_tensor, beta, threshold)
    else:
        pyt_y = torch.nn.functional.softplus(input_tensor)
    pyt_y.backward(gradient=grad_tensor)

    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.softplus_bw, golden_function=_golden_function)


def _golden_function(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()

    pyt_y = torch.trunc(input_tensor)
    pyt_y.backward(gradient=grad_tensor)

    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.trunc_bw, golden_function=_golden_function)


def _golden_function(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()

    pyt_y = torch.acos(input_tensor)
    pyt_y.backward(gradient=grad_tensor)

    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.acos_bw, golden_function=_golden_function)


def _golden_function_acosh(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.acosh(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.acosh_bw, golden_function=_golden_function_acosh)


def _golden_function_atan(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.atan(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.atan_bw, golden_function=_golden_function_atan)


def _golden_function_cos(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.cos(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.cos_bw, golden_function=_golden_function_cos)


def _golden_function_deg2rad(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.deg2rad(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.deg2rad_bw, golden_function=_golden_function_deg2rad)


def _golden_function_digamma(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.digamma(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.digamma_bw, golden_function=_golden_function_digamma)


def _golden_function_erf(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.erf(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.erf_bw, golden_function=_golden_function_erf)


def _golden_function_erfinv(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.erfinv(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.erfinv_bw, golden_function=_golden_function_erfinv)


def _golden_function_exp(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.exp(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.exp_bw, golden_function=_golden_function_exp)


def _golden_function_exp2(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.exp2(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.exp2_bw, golden_function=_golden_function_exp2)


def _golden_function_expm1(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.expm1(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.expm1_bw, golden_function=_golden_function_expm1)


def _golden_function_fill_zero(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.zeros_like(grad_tensor)

    return [pyt_y]


ttnn.attach_golden_function(ttnn.fill_zero_bw, golden_function=_golden_function_fill_zero)


def _golden_function_frac(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.frac(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.frac_bw, golden_function=_golden_function_frac)


def _golden_function_gelu(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.nn.functional.gelu(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.gelu_bw, golden_function=_golden_function_gelu)


def _golden_function_hardsigmoid(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.nn.functional.hardsigmoid(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.hardsigmoid_bw, golden_function=_golden_function_hardsigmoid)


def _golden_function_i0(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.i0(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.i0_bw, golden_function=_golden_function_i0)


def _golden_function_lgamma(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.lgamma(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.lgamma_bw, golden_function=_golden_function_lgamma)


def _golden_function_log_sigmoid(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.nn.functional.logsigmoid(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.log_sigmoid_bw, golden_function=_golden_function_log_sigmoid)


def _golden_function_logit(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.logit(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.logit_bw, golden_function=_golden_function_logit)


def _golden_function_mvlgamma(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.mvlgamma(input_tensor, 4)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.multigammaln_bw, golden_function=_golden_function_mvlgamma)


def _golden_function_neg(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.neg(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.neg_bw, golden_function=_golden_function_neg)


def _golden_function_rad2deg(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.rad2deg(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.rad2deg_bw, golden_function=_golden_function_rad2deg)


def _golden_function_reciprocal(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.reciprocal(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.reciprocal_bw, golden_function=_golden_function_reciprocal)


def _golden_function_relu(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.nn.functional.relu(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.relu_bw, golden_function=_golden_function_relu)


def _golden_function_rsqrt(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.rsqrt(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.rsqrt_bw, golden_function=_golden_function_rsqrt)


def _golden_function_sigmoid(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.sigmoid(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.sigmoid_bw, golden_function=_golden_function_sigmoid)


def _golden_function_sqrt(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.sqrt(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.sqrt_bw, golden_function=_golden_function_sqrt)


def _golden_function_tan(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.tan(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.tan_bw, golden_function=_golden_function_tan)


def _golden_function_tanh(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.tanh(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.tanh_bw, golden_function=_golden_function_tanh)


def _golden_function_threshold(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.nn.functional.threshold(input_tensor, *args)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.threshold_bw, golden_function=_golden_function_threshold)


def _golden_function_trunc(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.trunc(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.trunc_bw, golden_function=_golden_function_trunc)


def _golden_function_abs(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    if torch.is_complex(input_tensor):
        return _golden_function_abs_cmplx(grad_tensor, input_tensor)

    input_tensor.retain_grad()
    pyt_y = torch.abs(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.abs_bw, golden_function=_golden_function_abs)


def _golden_function_floor(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.floor(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.floor_bw, golden_function=_golden_function_floor)


def _golden_function_hardswish(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.nn.functional.hardswish(input_tensor, inplace=False)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.hardswish_bw, golden_function=_golden_function_hardswish)


def _golden_function_log(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.log(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.log_bw, golden_function=_golden_function_log)


def _golden_function_relu6(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.nn.functional.relu6(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.relu6_bw, golden_function=_golden_function_relu6)


def _golden_function_round(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.round(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.round_bw, golden_function=_golden_function_round)


def _golden_function_selu(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.nn.functional.selu(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.selu_bw, golden_function=_golden_function_selu)


def _golden_function_silu(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.nn.functional.silu(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.silu_bw, golden_function=_golden_function_silu)


def _golden_function_square(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.square(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.square_bw, golden_function=_golden_function_square)


def _golden_function_tanhshrink(grad_tensor, input_tensor, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.nn.functional.tanhshrink(input_tensor)
    pyt_y.backward(gradient=grad_tensor)
    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.tanhshrink_bw, golden_function=_golden_function_tanhshrink)


def _golden_function(grad_tensor, input_tensor, exponent, *args, **kwargs):
    import torch

    input_tensor.retain_grad()

    pyt_y = torch.pow(input_tensor, exponent)
    pyt_y.backward(gradient=grad_tensor)

    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.pow_bw, golden_function=_golden_function)


def _golden_function(grad_tensor, input_tensor, n, *args, **kwargs):
    import torch

    input_tensor.retain_grad()

    pyt_y = torch.polygamma(n, input_tensor)
    pyt_y.backward(gradient=grad_tensor)

    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.polygamma_bw, golden_function=_golden_function)


def _golden_function(grad_tensor, input_tensor, sizes, *args, **kwargs):
    import torch

    input_tensor.retain_grad()

    pyt_y = input_tensor.repeat(sizes)
    pyt_y.backward(gradient=grad_tensor)

    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.repeat_bw, golden_function=_golden_function)


def _golden_function(grad_tensor, input_tensor, *args, value=2.0, **kwargs):
    import torch

    input_tensor.retain_grad()
    pyt_y = torch.fill(input_tensor, value)
    pyt_y.backward(gradient=grad_tensor)

    return [input_tensor.grad]


ttnn.attach_golden_function(ttnn.fill_bw, golden_function=_golden_function)

__all__ = []
