# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional

import sys

import ttnn

__all__ = []


def apply_activations(tensor, activations):
    import torch

    string_to_function = {
        "relu": torch.relu,
        "gelu": torch.nn.functional.gelu,
        "silu": torch.nn.functional.silu,
    }

    if activations is not None:
        for activation in activations:
            activation_function = string_to_function[activation]
            tensor = activation_function(tensor)
    return tensor


def _golden_function(input_tensor_a, input_tensor_b, *args, activations=None, **kwargs):
    output_tensor = input_tensor_a + input_tensor_b
    return apply_activations(output_tensor, activations)


ttnn.attach_golden_function(ttnn.add, golden_function=_golden_function)
ttnn.attach_golden_function(ttnn.add_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, activations=None, **kwargs):
    output_tensor = input_tensor_a - input_tensor_b
    return apply_activations(output_tensor, activations)


ttnn.attach_golden_function(ttnn.subtract, golden_function=_golden_function)
ttnn.attach_golden_function(ttnn.subtract_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, activations=None, **kwargs):
    output_tensor = input_tensor_a * input_tensor_b
    return apply_activations(output_tensor, activations)


ttnn.attach_golden_function(ttnn.multiply, golden_function=_golden_function)
ttnn.attach_golden_function(ttnn.multiply_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.eq(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.eq, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.ne(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.ne, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.gt(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.gt, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.ge(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.ge, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.lt(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.lt, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.le(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.le, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return input_tensor_a.logical_and_(input_tensor_b)


ttnn.attach_golden_function(ttnn.logical_and_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return input_tensor_a.logical_or_(input_tensor_b)


ttnn.attach_golden_function(ttnn.logical_or_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return input_tensor_a.logical_xor_(input_tensor_b)


ttnn.attach_golden_function(ttnn.logical_xor_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.ldexp(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.ldexp, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.logaddexp(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.logaddexp, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.logaddexp2(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.logaddexp2, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.divide(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.divide, golden_function=_golden_function)


def _golden_function(a, b, *args, **kwargs):
    import torch

    return torch.nn.functional.gelu(torch.add(a, b))


ttnn.attach_golden_function(ttnn.bias_gelu, golden_function=_golden_function)


def _golden_function_squared_difference(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch_squared_difference(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.squared_difference, golden_function=_golden_function_squared_difference)


def _golden_function_addalpha(input_tensor_a, input_tensor_b, alpha, *args, **kwargs):
    import torch

    return torch.add(input_tensor_a, input_tensor_b, alpha=alpha)


ttnn.attach_golden_function(ttnn.addalpha, golden_function=_golden_function_addalpha)


def _golden_function_subalpha(input_tensor_a, input_tensor_b, alpha, *args, **kwargs):
    import torch

    return torch.sub(input_tensor_a, input_tensor_b, alpha=alpha)


ttnn.attach_golden_function(ttnn.subalpha, golden_function=_golden_function_subalpha)


def _golden_function_xlogy(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.xlogy(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.xlogy, golden_function=_golden_function_xlogy)


def _golden_function_hypot(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.hypot(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.hypot, golden_function=_golden_function_hypot)


def _golden_function_maximum(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.maximum(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.maximum, golden_function=_golden_function_maximum)


def _golden_function_minimum(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.minimum(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.minimum, golden_function=_golden_function_minimum)


def _golden_function_logical_xor(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.logical_xor(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.logical_xor, golden_function=_golden_function_logical_xor)


def _golden_function_logical_and(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.logical_and(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.logical_and, golden_function=_golden_function_logical_and)


def _golden_function_logical_or(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.logical_or(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.logical_or, golden_function=_golden_function_logical_or)


def _golden_function_atan2(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.atan2(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.atan2, golden_function=_golden_function_atan2)


def _golden_function_nextafter(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.nextafter(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.nextafter, golden_function=_golden_function_nextafter)


def _golden_function_isclose(input_tensor_a, input_tensor_b, *args, rtol=1e-05, atol=1e-08, equal_nan=False, **kwargs):
    import torch

    return torch.isclose(input_tensor_a, input_tensor_b, rtol=rtol, atol=atol, equal_nan=equal_nan)


ttnn.attach_golden_function(ttnn.isclose, golden_function=_golden_function_isclose)


def _golden_function_div(input_tensor_a, input_tensor_b, round_mode, *args, **kwargs):
    import torch

    if round_mode == "None":
        return torch.div(input_tensor_a, input_tensor_b, rounding_mode=None)
    return torch.div(input_tensor_a, input_tensor_b, rounding_mode=round_mode)


ttnn.attach_golden_function(ttnn.div, golden_function=_golden_function_div)


def _golden_function_div_no_nan(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if isinstance(input_tensor_b, float):
        if input_tensor_b == 0:
            return torch.zeros_like(input_tensor_a)
        else:
            return input_tensor_a / input_tensor_b
    else:
        return torch.where(input_tensor_b == 0, 0, input_tensor_a / input_tensor_b)


ttnn.attach_golden_function(ttnn.div_no_nan, golden_function=_golden_function_div_no_nan)


def _golden_function_floor_div(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.floor_divide(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.floor_div, golden_function=_golden_function_floor_div)


def _golden_function_remainder(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.remainder(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.remainder, golden_function=_golden_function_remainder)


def _golden_function_fmod(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.fmod(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.fmod, golden_function=_golden_function_fmod)


def torch_squared_difference(x, y, *args, **kwargs):
    import torch

    return torch.square(torch.sub(x, y))


def _golden_function_scatter(input_tensor_a, input_tensor_b, *args, **kwargs):
    input_tensor_b[:, :, : input_tensor_a.shape[-2], : input_tensor_a.shape[-1]] = input_tensor_a
    return input_tensor_b


ttnn.attach_golden_function(ttnn.scatter, golden_function=_golden_function_scatter)


def _golden_function_outer(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.outer(input_tensor_a.squeeze(), input_tensor_b.squeeze())


ttnn.attach_golden_function(ttnn.outer, golden_function=_golden_function_outer)


def _golden_function_polyval(input_tensor_a, coeffs, *args, **kwargs):
    result = 0.0
    for coeff in coeffs:
        result = result * input_tensor_a + coeff
    return result


ttnn.attach_golden_function(ttnn.polyval, golden_function=_golden_function_polyval)


def _golden_function_gt_(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return input_tensor_a.gt_(input_tensor_b)


ttnn.attach_golden_function(ttnn.gt_, golden_function=_golden_function_gt_)


def _golden_function_le_(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return input_tensor_a.le_(input_tensor_b)


ttnn.attach_golden_function(ttnn.le_, golden_function=_golden_function_le_)


def _golden_function_lt_(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return input_tensor_a.lt_(input_tensor_b)


ttnn.attach_golden_function(ttnn.lt_, golden_function=_golden_function_lt_)


def _golden_function_ge_(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return input_tensor_a.ge_(input_tensor_b)


ttnn.attach_golden_function(ttnn.ge_, golden_function=_golden_function_ge_)


def _golden_function_eq_(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return input_tensor_a.eq_(input_tensor_b)


ttnn.attach_golden_function(ttnn.eq_, golden_function=_golden_function_eq_)


def _golden_function_ne_(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return input_tensor_a.ne_(input_tensor_b)


ttnn.attach_golden_function(ttnn.ne_, golden_function=_golden_function_ne_)


__all__ = []
