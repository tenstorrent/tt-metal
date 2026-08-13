# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Union


import ttnn


def _golden_function(input_tensor: ttnn.Tensor, **_):
    import torch

    return torch.zeros_like(input_tensor)


ttnn.attach_golden_function(ttnn.zeros_like, golden_function=_golden_function)


def _golden_function(input_tensor: ttnn.Tensor, **_):
    import torch

    return torch.ones_like(input_tensor)


ttnn.attach_golden_function(ttnn.ones_like, golden_function=_golden_function)


def _golden_function(input_tensor: ttnn.Tensor, *args, fill_value: float = 0.0, **_):
    import torch

    # Build from shape as float32 so a negative/fractional fill_value cannot overflow an
    # integer input dtype, and accept extra positional args (dtype/layout/device/...).
    return torch.full(tuple(input_tensor.shape), fill_value, dtype=torch.float32)


ttnn.attach_golden_function(ttnn.full_like, golden_function=_golden_function)


def _golden_function(input_shape: ttnn.Shape, **_):
    import torch

    return torch.zeros(input_shape)


ttnn.attach_golden_function(ttnn.zeros, golden_function=_golden_function)


def _golden_function(input_shape: ttnn.Shape, **_):
    import torch

    return torch.ones(input_shape)


ttnn.attach_golden_function(ttnn.ones, golden_function=_golden_function)


def _golden_function_full(input_shape: ttnn.Shape, fill_value: float, **_):
    import torch

    return torch.full(input_shape, fill_value=fill_value)


ttnn.attach_golden_function(ttnn.full, golden_function=_golden_function_full)


# ttnn.empty / ttnn.empty_like allocate uninitialized memory, so a golden function would
# only ever compare device garbage against host garbage; intentionally leave them unattached.


def _golden_function(*args, **_):
    import torch

    # Mirror the two ttnn.arange pybind overloads: arange(end) and arange(start, end, step=1),
    # so the golden matches however many positional args the caller passed.
    if len(args) == 1:
        start, end, step = 0, args[0], 1
    elif len(args) == 2:
        start, end, step = args[0], args[1], 1
    else:
        start, end, step = args[0], args[1], args[2]
    return torch.arange(start, end, step)


ttnn.attach_golden_function(ttnn.arange, golden_function=_golden_function)

__all__ = []
