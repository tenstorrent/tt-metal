# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys


import ttnn


def register_ttnn_cpp_complex_binary_function(complex_binary_function):
    import torch

    def torch_complex_with_float(torch_op, x, y, grad_data, *args, **kwargs):
        alpha = kwargs.pop("alpha")
        pyt_y = torch_op(x, y, alpha=alpha)

        pyt_y.backward(gradient=grad_data)

        grad_in_real = torch.real(x.grad)
        grad_in_imag = torch.imag(x.grad)
        grad_other_real = torch.real(y.grad)
        grad_other_imag = torch.imag(y.grad)

        golden_tensor = [
            torch.cat((grad_in_real, grad_in_imag), dim=-1),
            torch.cat((grad_other_real, grad_other_imag), dim=-1),
        ]

        return golden_tensor

    def torch_complex(torch_op, x, y, grad_data, *args, **kwargs):
        pyt_y = torch_op(x, y)

        pyt_y.backward(gradient=grad_data)

        grad_in_real = torch.real(x.grad)
        grad_in_imag = torch.imag(x.grad)
        grad_other_real = torch.real(y.grad)
        grad_other_imag = torch.imag(y.grad)

        golden_tensor = [
            torch.cat((grad_in_real, grad_in_imag), dim=-1),
            torch.cat((grad_other_real, grad_other_imag), dim=-1),
        ]

        return golden_tensor

    name_to_golden_function = {
        "add_bw": lambda x, y, grad_data: torch_complex_with_float(torch.add, x, y, grad_data),
        "sub_bw": lambda x, y, grad_data: torch_complex_with_float(torch.sub, x, y, grad_data),
        "mul_bw": lambda x, y, grad_data: torch_complex(torch.mul, x, y, grad_data),
        "div_bw": lambda x, y, grad_data: torch_complex(torch.div, x, y, grad_data),
    }

    golden_keys = set(name_to_golden_function.keys())
    function_names = {function.__name__.split(".")[-1] for function in TTNN_ELTWISE_UNARY_CPP_FUNCTIONS}
    if golden_keys != function_names:
        raise ImportError(f"Missing or extra golden functions:\n{golden_keys}\nshould be equal to\n{function_names}")

    def _golden_function(grad_tensor: ttnn.Tensor, input_tensor: ttnn.Tensor, other_tensor: ttnn.Tensor, **_):
        torch_function = name_to_golden_function[complex_binary_function.__name__.split(".")[-1]]
        return torch_function(input_tensor, other_tensor)

    ttnn.attach_golden_function(complex_binary_function, golden_function=_golden_function)


TTNN_ELTWISE_UNARY_CPP_FUNCTIONS = [
    ttnn.add_bw,
    ttnn.sub_bw,
    ttnn.mul_bw,
    ttnn.div_bw,
]
for unary_function in TTNN_ELTWISE_UNARY_CPP_FUNCTIONS:
    register_ttnn_cpp_complex_binary_function(unary_function)

__all__ = []
