# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import os


def register_ttnn_cpp_unary_function(unary_function):
    def _golden_function(input_tensor: ttnn.Tensor, **_):
        import torch

        name_to_golden_function = {
            "floor": torch.floor,
            "ceil": torch.ceil,
            "trunc": torch.trunc,
        }

        golden_keys = set(name_to_golden_function.keys())
        function_names = {function.__name__.split(".")[-1] for function in TTNN_ELTWISE_UNARY_CPP_FUNCTIONS}
        if golden_keys != function_names:
            raise ImportError(
                f"Missing or extra golden functions:\n{golden_keys}\nshould be equal to\n{function_names}"
            )

        torch_function = name_to_golden_function[unary_function.__name__.split(".")[-1]]
        return torch_function(input_tensor)

    ttnn.attach_golden_function(unary_function, golden_function=_golden_function)


TTNN_ELTWISE_UNARY_CPP_FUNCTIONS = [
    ttnn.floor,
    ttnn.ceil,
    ttnn.trunc,
]
for unary_function in TTNN_ELTWISE_UNARY_CPP_FUNCTIONS:
    register_ttnn_cpp_unary_function(unary_function)


def _golden_function_cosh(input_tensor_a, *args, **kwargs):
    import torch

    return torch.cosh(input_tensor_a)


ttnn.attach_golden_function(ttnn.cosh, golden_function=_golden_function_cosh)


def _golden_function_cbrt(input_tensor_a, *args, **kwargs):
    import torch

    return torch.pow(torch.abs(input_tensor_a), 1.0 / 3.0) * torch.sign(input_tensor_a)


ttnn.attach_golden_function(ttnn.cbrt, golden_function=_golden_function_cbrt)


def _golden_function_hardtanh(input_tensor_a, *args, min_val=-1.0, max_val=1.0, **kwargs):
    import torch

    return torch.nn.functional.hardtanh(input_tensor_a, min_val=min_val, max_val=max_val)


ttnn.attach_golden_function(ttnn.hardtanh, golden_function=_golden_function_hardtanh)


def _golden_function_lgamma(input_tensor_a, *args, **kwargs):
    import torch

    return torch.lgamma(input_tensor_a)


ttnn.attach_golden_function(ttnn.lgamma, golden_function=_golden_function_lgamma)


def _golden_function_hardsigmoid(input_tensor_a, *args, **kwargs):
    import torch

    return torch.nn.functional.hardsigmoid(input_tensor_a)


ttnn.attach_golden_function(ttnn.hardsigmoid, golden_function=_golden_function_hardsigmoid)


def _golden_function_rpow(input_tensor_a, *args, base, **kwargs):
    import torch

    return torch.pow(torch.tensor(base), input_tensor_a)


ttnn.attach_golden_function(ttnn.rpow, golden_function=_golden_function_rpow)


def _golden_function_softsign(input_tensor_a, *args, **kwargs):
    import torch

    return input_tensor_a / (1 + torch.abs(input_tensor_a))


ttnn.attach_golden_function(ttnn.softsign, golden_function=_golden_function_softsign)


def _golden_function_selu(input_tensor_a, *args, **kwargs):
    import torch

    return torch.nn.functional.selu(input_tensor_a)


ttnn.attach_golden_function(ttnn.selu, golden_function=_golden_function_selu)


def _golden_function_hardswish(input_tensor_a, *args, **kwargs):
    import torch

    return torch.nn.functional.hardswish(input_tensor_a)


ttnn.attach_golden_function(ttnn.hardswish, golden_function=_golden_function_hardswish)


def _golden_function_softshrink(input_tensor_a, *args, lambd=0.5, **kwargs):
    import torch

    return torch.nn.functional.softshrink(input_tensor_a, lambd=lambd)


ttnn.attach_golden_function(ttnn.softshrink, golden_function=_golden_function_softshrink)


def _golden_function_swish(input_tensor_a, *args, **kwargs):
    import torch

    return torch.nn.functional.silu(input_tensor_a)


ttnn.attach_golden_function(ttnn.swish, golden_function=_golden_function_swish)


def _golden_function_frac(input_tensor_a, *args, **kwargs):
    import torch

    return torch.frac(input_tensor_a)


ttnn.attach_golden_function(ttnn.frac, golden_function=_golden_function_frac)


def _golden_function_atanh(input_tensor_a, *args, **kwargs):
    import torch

    return torch.atanh(input_tensor_a)


ttnn.attach_golden_function(ttnn.atanh, golden_function=_golden_function_atanh)


def _golden_function_sinh(input_tensor_a, *args, **kwargs):
    import torch

    return torch.sinh(input_tensor_a)


ttnn.attach_golden_function(ttnn.sinh, golden_function=_golden_function_sinh)


try:
    SigmoidMode = ttnn._ttnn.operations.unary.SigmoidMode
except AttributeError:
    pass

__all__ = []
