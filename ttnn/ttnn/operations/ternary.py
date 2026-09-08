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


def _golden_function_addcmul(input_a, input_b, input_c, *args, value=1, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_a.dtype):
        # PyTorch lacks UInt32 addcmul; widen and restore the hardware wraparound.
        return integer_golden.addcmul(input_a, input_b, input_c, value)
    return torch.addcmul(input_a, input_b, input_c, value=value)


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

    # The device path is a fused multiply-add: SFPMAD accumulates in fp32 and rounds once on
    # store. Rounding the product to the input dtype before the add rounds twice and drifts up
    # to 2 ULP from the fused result, so accumulate in fp32 here and round once as well.
    def promote(x):
        return x.float() if torch.is_tensor(x) else x

    out_dtype = next(x.dtype for x in (input_tensor_a, input_tensor_b, input_tensor_c) if torch.is_tensor(x))
    result = torch.add(torch.mul(promote(input_tensor_a), promote(input_tensor_b)), promote(input_tensor_c))
    return result.to(out_dtype)


ttnn.attach_golden_function(ttnn.mac, golden_function=_golden_function_mac)


def _golden_function_where(predicate, true_value, false_value, *args, **kwargs):
    import torch

    # TT where selects the true branch for any nonzero predicate; torch.where requires bool.
    if integer_golden.is_unsigned_dtype(predicate.dtype):
        condition = integer_golden.compare(predicate, 0, torch.gt)
        # Widen unsigned branches because Torch has no UInt16/UInt32 where kernel.
        output_dtype = (
            true_value.dtype
            if torch.is_tensor(true_value)
            else false_value.dtype
            if torch.is_tensor(false_value)
            else predicate.dtype
        )
        true_value = true_value.to(torch.int64) if torch.is_tensor(true_value) else true_value
        false_value = false_value.to(torch.int64) if torch.is_tensor(false_value) else false_value
        result = torch.where(condition, true_value, false_value)
        return result.to(output_dtype)
    else:
        condition = torch.ne(predicate, 0)
    return torch.where(condition, true_value, false_value)


ttnn.attach_golden_function(ttnn.where, golden_function=_golden_function_where)


__all__ = []
