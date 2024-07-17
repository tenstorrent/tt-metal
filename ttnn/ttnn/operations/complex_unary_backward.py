# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import ttnn


def register_ttnn_cpp_complex_unary_function(complex_unary_function):
    import torch

    def torch_complex(torch_op, x, grad_data, *args, **kwargs):
        pyt_y = torch_op(x)

        pyt_y.backward(gradient=grad_data)

        grad_in_real = torch.real(x.grad)
        grad_in_imag = torch.imag(x.grad)

        golden_tensor = [
            torch.cat((grad_in_real, grad_in_imag), dim=-1),
        ]

        return golden_tensor

    name_to_golden_function = {
        "polar_bw": lambda x, grad_data: torch_complex(torch.polar, x, grad_data),
        "real_bw": lambda x, grad_data: torch_complex(torch.real, x, grad_data),
        "imag_bw": lambda x, grad_data: torch_complex(torch.imag, x, grad_data),
        "angle_bw": lambda x, grad_data: torch_complex(torch.angle, x, grad_data),
        "conj_bw": lambda x, grad_data: torch_complex(torch.conj, x, grad_data),
        "abs_bw": lambda x, grad_data: torch_complex(torch.abs, x, grad_data),
        "reciprocal_bw": lambda x, grad_data: torch_complex(torch.recip, x, grad_data),
    }

    golden_keys = set(name_to_golden_function.keys())
    function_names = {function.__name__.split(".")[-1] for function in TTNN_ELTWISE_UNARY_CPP_FUNCTIONS}
    if golden_keys != function_names:
        raise ImportError(f"Missing or extra golden functions:\n{golden_keys}\nshould be equal to\n{function_names}")

    def _golden_function(grad_tensor: ttnn.Tensor, input_tensor: ttnn.Tensor, other_tensor: ttnn.Tensor, **_):
        torch_function = name_to_golden_function[complex_unary_function.__name__.split(".")[-1]]
        return torch_function(input_tensor, other_tensor)

    ttnn.attach_golden_function(complex_unary_function, golden_function=_golden_function)


TTNN_ELTWISE_UNARY_CPP_FUNCTIONS = [
    ttnn.polar_bw,
    ttnn.real_bw,
    ttnn.imag_bw,
    ttnn.angle_bw,
    ttnn.conj_bw,
    ttnn.abs_bw,
    ttnn.reciprocal_bw,
]
for unary_function in TTNN_ELTWISE_UNARY_CPP_FUNCTIONS:
    register_ttnn_cpp_complex_unary_function(unary_function)

__all__ = []
