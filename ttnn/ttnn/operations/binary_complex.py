# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

__all__ = []


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.add(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.add, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.sub(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.subtract, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.mul(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.multiply, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.div(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.divide, golden_function=_golden_function)


__all__ = []
