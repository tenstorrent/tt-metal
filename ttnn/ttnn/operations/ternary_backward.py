# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional

import sys

import ttnn

import torch

THIS_MODULE = sys.modules[__name__]

__all__ = []


def _golden_function_backward(
    torch_op, grad_tensor, input_tensor_a, input_tensor_b, input_tensor_c, alpha, *args, **kwargs
):
    pyt_y = torch_op(input_tensor_a, input_tensor_b, input_tensor_c, value=alpha)

    pyt_y.backward(gradient=grad_tensor)

    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad, input_tensor_c.grad]
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


addcmul_bw = ttnn.register_operation(
    golden_function=lambda grad, a, b, c, alpha, *args, **kwargs: _golden_function_backward(
        torch.addcmul, grad, a, b, c, alpha, *args, **kwargs
    )
)(ttnn._ttnn.operations.ternary_backward.addcmul_bw)


__all__ = []
