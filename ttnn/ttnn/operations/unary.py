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


try:
    SigmoidMode = ttnn._ttnn.operations.unary.SigmoidMode
except AttributeError:
    pass

__all__ = []
