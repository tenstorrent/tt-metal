# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional

import sys

import ttnn


__all__ = []


def _golden_function(grad_tensor, input_tensor, tensor1_tensor, tensor2_tensor, value, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    tensor1_tensor.retain_grad()
    tensor2_tensor.retain_grad()

    pyt_y = torch.addcmul(input_tensor, tensor1_tensor, tensor2_tensor, value=value)

    pyt_y.backward(gradient=grad_tensor)

    golden_tensor = [input_tensor.grad, tensor1_tensor.grad, tensor2_tensor.grad]
    return golden_tensor


ttnn.attach_golden_function(ttnn.addcmul_bw, golden_function=_golden_function)


def _golden_function(grad_tensor, input_tensor, tensor1_tensor, tensor2_tensor, value, *args, **kwargs):
    import torch

    input_tensor.retain_grad()
    tensor1_tensor.retain_grad()
    tensor2_tensor.retain_grad()

    pyt_y = torch.addcdiv(input_tensor, tensor1_tensor, tensor2_tensor, value=value)

    pyt_y.backward(gradient=grad_tensor)

    golden_tensor = [input_tensor.grad, tensor1_tensor.grad, tensor2_tensor.grad]
    return golden_tensor


ttnn.attach_golden_function(ttnn.addcdiv_bw, golden_function=_golden_function)

__all__ = []
