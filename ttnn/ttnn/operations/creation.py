# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

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


def _golden_function(input_tensor: ttnn.Tensor, *, fill_value: float, **_):
    import torch

    return torch.full_like(input_tensor, fill_value)


ttnn.attach_golden_function(ttnn.full_like, golden_function=_golden_function)


def _golden_function(input_tensor: ttnn.Tensor, *, fill_value: float, **_):
    import torch

    return torch.empty_like(input_tensor, fill_value)


ttnn.attach_golden_function(ttnn.empty_like, golden_function=_golden_function)


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


def _golden_function(input_shape: ttnn.Shape, **_):
    import torch

    return torch.empty(input_shape)


ttnn.attach_golden_function(ttnn.empty, golden_function=_golden_function)


def _golden_function(start: int, end: int, step: int, **_):
    import torch

    return torch.arange(start, end, step)


ttnn.attach_golden_function(ttnn.arange, golden_function=_golden_function)

__all__ = []
