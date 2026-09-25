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


def _golden_function(shape: ttnn.Shape, dtype=None, *_, **__):
    import torch

    # TTNN accepts Shape directly, while Torch creation functions require a tuple of dimensions.
    # Normalize it in this golden to keep shared comparison preprocessing unchanged.
    if isinstance(shape, ttnn.Shape):
        shape = tuple(shape)
    torch_dtype = ttnn.ttnn_dtype_to_torch_dtype(dtype) if dtype is not None else torch.bfloat16
    return torch.zeros(shape, dtype=torch_dtype)


ttnn.attach_golden_function(ttnn.zeros, golden_function=_golden_function)


def _golden_function(shape: ttnn.Shape, dtype=None, *_, **__):
    import torch

    # The TTNN API permits dtype, layout, device, and memory config as positional arguments.
    # Torch only needs the requested dtype, so absorb the remaining allocation-only arguments.
    if isinstance(shape, ttnn.Shape):
        shape = tuple(shape)
    torch_dtype = ttnn.ttnn_dtype_to_torch_dtype(dtype) if dtype is not None else torch.bfloat16
    return torch.ones(shape, dtype=torch_dtype)


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


def _skip_random_comparison(output):
    """Mark a random golden output as shape/dtype-only, since device and torch RNGs differ."""
    ttnn.decorators.set_golden_comparison_config(output, method="skip", scope="all")
    return output


def _golden_function_rand(shape, *_, dtype=ttnn.bfloat16, low=0.0, high=1.0, **__):
    import torch

    if isinstance(shape, ttnn.Shape):
        shape = tuple(shape)
    torch_dtype = ttnn.ttnn_dtype_to_torch_dtype(dtype)
    output = torch.rand(shape, dtype=torch.float32).mul_(high - low).add_(low).to(torch_dtype)
    return _skip_random_comparison(output)


ttnn.attach_golden_function(ttnn.rand, golden_function=_golden_function_rand)


def _golden_function_randn(shape, *_, dtype=ttnn.bfloat16, **__):
    import torch

    if isinstance(shape, ttnn.Shape):
        shape = tuple(shape)
    torch_dtype = ttnn.ttnn_dtype_to_torch_dtype(dtype)
    output = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
    return _skip_random_comparison(output)


ttnn.attach_golden_function(ttnn.randn, golden_function=_golden_function_randn)


def _golden_function_uniform(input_tensor, *args, _ttnn_global_golden=False, **kwargs):
    import torch

    # 'from' is a Python reserved word, so it can only arrive as a positional argument or a kwargs dict entry.
    from_value = kwargs["from"] if "from" in kwargs else (args[0] if len(args) > 0 else 0.0)
    to_value = kwargs["to"] if "to" in kwargs else (args[1] if len(args) > 1 else 1.0)
    output = torch.rand(input_tensor.shape, dtype=torch.float32).mul_(to_value - from_value).add_(from_value)
    output = output.to(input_tensor.dtype)
    if _ttnn_global_golden:
        input_tensor.copy_(output)
        return _skip_random_comparison(input_tensor)
    return _skip_random_comparison(output)


_golden_function_uniform._ttnn_mutates_global_inputs = True
ttnn.attach_golden_function(ttnn.uniform, golden_function=_golden_function_uniform)


def _golden_function_bernoulli(input_tensor, *_, dtype=None, **__):
    import torch

    # The input tensor holds per-element probabilities; sample in float32 then cast to the output dtype.
    output = torch.bernoulli(input_tensor.float())
    if dtype is not None:
        output = output.to(ttnn.ttnn_dtype_to_torch_dtype(dtype))
    else:
        output = output.to(input_tensor.dtype)
    return _skip_random_comparison(output)


ttnn.attach_golden_function(ttnn.bernoulli, golden_function=_golden_function_bernoulli)

__all__ = []
