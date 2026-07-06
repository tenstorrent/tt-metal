# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Union

import ttnn

__all__ = []


def torch_mac(input, tensor1, tensor2):
    import torch

    return torch.add(torch.mul(input, tensor1), tensor2)


def _golden_function_addcmul(input_tensor_a, input_tensor_b, input_tensor_c, *args, value=1, **kwargs):
    import torch

    return torch.addcmul(input_tensor_a, input_tensor_b, input_tensor_c, value=value)


ttnn.attach_golden_function(ttnn.addcmul, golden_function=_golden_function_addcmul)


def _golden_function_addcdiv(input_tensor_a, input_tensor_b, input_tensor_c, *args, value=1, **kwargs):
    import torch

    return torch.addcdiv(input_tensor_a, input_tensor_b, input_tensor_c, value=value)


ttnn.attach_golden_function(ttnn.addcdiv, golden_function=_golden_function_addcdiv)


def _golden_function_lerp(input_tensor_a, input_tensor_b, input_tensor_c, *args, **kwargs):
    import torch

    if torch.is_tensor(input_tensor_c):
        input_tensor_c = input_tensor_c.to(input_tensor_a.dtype)

    return torch.lerp(input_tensor_a, input_tensor_b.to(input_tensor_a.dtype), input_tensor_c)


ttnn.attach_golden_function(ttnn.lerp, golden_function=_golden_function_lerp)


def _golden_function_mac(input_tensor_a, input_tensor_b, input_tensor_c, *args, **kwargs):
    import torch

    return torch.add(torch.mul(input_tensor_a, input_tensor_b), input_tensor_c)


ttnn.attach_golden_function(ttnn.mac, golden_function=_golden_function_mac)


def _golden_function_where(input_tensor_a, input_tensor_b, input_tensor_c, *args, **kwargs):
    import torch

    return torch.where(input_tensor_a, input_tensor_b, input_tensor_c)


ttnn.attach_golden_function(ttnn.where, golden_function=_golden_function_where)


# Single-argument overload: ttnn.where(condition) ≡ torch.where(condition) ≡
# torch.nonzero(condition, as_tuple=True). Returns a tuple of D 1-D int64
# torch tensors — one per input dim — giving the indices of nonzero elements.
# The 3-arg elementwise-select overload is unchanged.
_where_ternary_impl = ttnn.where


def _where_single_tensor_indices(condition):
    import torch

    orig_shape = tuple(condition.shape)
    n = 1
    for d in orig_shape:
        n *= d
    ndim = len(orig_shape)
    if n == 0:
        return tuple(torch.empty(0, dtype=torch.int64) for _ in range(ndim))
    flat = ttnn.reshape(condition, ttnn.Shape([1, 1, 1, n]))
    count_t, idx_t = ttnn.nonzero(flat)
    count = int(ttnn.to_torch(ttnn.from_device(count_t)).flatten()[0].item())
    if count == 0:
        return tuple(torch.empty(0, dtype=torch.int64) for _ in range(ndim))
    # ttnn.nonzero returns a flat buffer of (b, n, h, c) 4-tuples matching
    # torch.nonzero(x, as_tuple=False) semantics. For our [1,1,1,n] reshape
    # only the `c` slot varies — take stride-4 offset 3 and slice to `count`.
    raw = ttnn.to_torch(ttnn.from_device(idx_t)).flatten()
    linear = raw[: count * 4].reshape(count, 4)[:, 3].to(torch.int64)
    strides = [0] * ndim
    s = 1
    for i in range(ndim - 1, -1, -1):
        strides[i] = s
        s *= orig_shape[i]
    return tuple((linear // strides[i]) % orig_shape[i] for i in range(ndim))


def where(*args, **kwargs):
    if len(args) == 1 and not kwargs:
        return _where_single_tensor_indices(args[0])
    return _where_ternary_impl(*args, **kwargs)


where.__doc__ = (
    "ttnn.where(condition) → tuple of int64 index tensors, matching "
    "torch.where(condition) / torch.nonzero(condition, as_tuple=True).\n"
    "ttnn.where(condition, x, y) → elementwise select (C++ ternary op)."
)

ttnn.where = where


__all__ = []
