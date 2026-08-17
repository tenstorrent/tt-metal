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


def _golden_function(input_tensor: ttnn.Tensor, fill_value: float, dtype=None, *_, **__):
    import torch

    # Honor the output dtype override instead of always inheriting the input tensor dtype.
    torch_dtype = ttnn.ttnn_dtype_to_torch_dtype(dtype) if dtype is not None else None
    return torch.full_like(input_tensor, fill_value, dtype=torch_dtype)


ttnn.attach_golden_function(ttnn.full_like, golden_function=_golden_function)


# empty_like returns uninitialized storage, so comparison mode has no meaningful value golden.
ttnn.attach_golden_function(ttnn.empty_like, golden_function=None)


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


# empty returns uninitialized storage, so comparison mode has no meaningful value golden.
ttnn.attach_golden_function(ttnn.empty, golden_function=None)


def _golden_function(*args, dtype=ttnn.bfloat16, **kwargs):
    import torch

    kwargs.pop("device", None)
    kwargs.pop("memory_config", None)
    kwargs.pop("layout", None)
    # Forward all supported range overloads, then cast to the requested TTNN dtype.
    return torch.arange(*args, **kwargs).to(ttnn.ttnn_dtype_to_torch_dtype(dtype))


ttnn.attach_golden_function(ttnn.arange, golden_function=_golden_function)

__all__ = []
