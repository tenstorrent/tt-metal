# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Union

import ttnn
from ttnn.operations import integer_golden

__all__ = []


def torch_mac(input, tensor1, tensor2):
    import torch

    return torch.add(torch.mul(input, tensor1), tensor2)


def _golden_function_addcmul(input_tensor_a, input_tensor_b, input_tensor_c, *args, value=1, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # PyTorch lacks UInt32 addcmul; widen and restore the hardware wraparound.
        return integer_golden.addcmul(input_tensor_a, input_tensor_b, input_tensor_c, value)
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

    # TT where selects the true branch for any nonzero predicate; torch.where requires bool.
    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        condition = integer_golden.compare(input_tensor_a, 0, torch.gt)
        # Widen unsigned branches because Torch has no UInt16/UInt32 where kernel.
        output_dtype = (
            input_tensor_b.dtype
            if torch.is_tensor(input_tensor_b)
            else input_tensor_c.dtype
            if torch.is_tensor(input_tensor_c)
            else input_tensor_a.dtype
        )
        input_tensor_b = input_tensor_b.to(torch.int64) if torch.is_tensor(input_tensor_b) else input_tensor_b
        input_tensor_c = input_tensor_c.to(torch.int64) if torch.is_tensor(input_tensor_c) else input_tensor_c
        result = torch.where(condition, input_tensor_b, input_tensor_c)
        return result.to(output_dtype)
    else:
        condition = torch.ne(input_tensor_a, 0)
    return torch.where(condition, input_tensor_b, input_tensor_c)


ttnn.attach_golden_function(ttnn.where, golden_function=_golden_function_where)


__all__ = []
