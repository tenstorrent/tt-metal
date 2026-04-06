# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import os


def register_ttnn_cpp_unary_function(unary_function):
    def _golden_function(input_tensor: ttnn.Tensor, **_):
        import torch

        def torch_cbrt(x, *args, **kwargs):
            return torch.sgn(x) * torch.pow(torch.abs(x), 1.0 / 3)

        def torch_multigammaln(x, *args, **kwargs):
            result = torch.lgamma(x)
            result += torch.lgamma(x - 0.5)
            result += torch.lgamma(x - 1.0)
            result += torch.lgamma(x - 1.5)
            result += 3.434189657547
            return result

        def torch_hardmish(x):
            x_f32 = x.to(torch.float32)
            result_f32 = x_f32 * torch.clamp(x_f32 + 2.8, min=0.0, max=5.0) / 5

            if x.dtype == torch.bfloat16:
                # Simulate SFPSTORE truncating
                result_int32 = result_f32.view(torch.int32)
                shifted_int32 = torch.bitwise_right_shift(result_int32, 16)
                truncated_int16 = shifted_int32.to(torch.int16)
                final_result = truncated_int16.view(torch.bfloat16)
            else:
                final_result = result_f32

            return final_result

        name_to_golden_function = {
            "hardsigmoid": torch.nn.functional.hardsigmoid,
            "floor": torch.floor,
            "ceil": torch.ceil,
            "trunc": torch.trunc,
            "frac": torch.frac,
            "sinh": torch.sinh,
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
    ttnn.hardsigmoid,
    ttnn.floor,
    ttnn.ceil,
    ttnn.trunc,
    ttnn.frac,
    ttnn.sinh,
]
for unary_function in TTNN_ELTWISE_UNARY_CPP_FUNCTIONS:
    register_ttnn_cpp_unary_function(unary_function)


def _golden_function_hardtanh(input_tensor_a, *args, min_val=-1.0, max_val=1.0, **kwargs):
    import torch

    return torch.nn.functional.hardtanh(input_tensor_a, min_val=min_val, max_val=max_val)


ttnn.attach_golden_function(ttnn.hardtanh, golden_function=_golden_function_hardtanh)


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


try:
    SigmoidMode = ttnn._ttnn.operations.unary.SigmoidMode
except AttributeError:
    pass

__all__ = []
