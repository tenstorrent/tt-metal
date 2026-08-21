# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional

import sys

import ttnn
from ttnn.operations import integer_golden

__all__ = []


def apply_activations(tensor, activations):
    import torch

    act_func_map = {
        ttnn.UnaryOpType.RELU: torch.nn.functional.relu,
        ttnn.UnaryOpType.SILU: torch.nn.functional.silu,
        ttnn.UnaryOpType.MISH: torch.nn.functional.mish,
        ttnn.UnaryOpType.SIGMOID: torch.nn.functional.sigmoid,
        ttnn.UnaryOpType.TANH: torch.nn.functional.tanh,
        ttnn.UnaryOpType.LOG: torch.log,
        ttnn.UnaryOpType.SOFTPLUS: torch.nn.functional.softplus,
        ttnn.UnaryOpType.GELU: torch.nn.functional.gelu,
        ttnn.UnaryOpType.GELU_TANH: lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
        ttnn.UnaryOpType.SQRT: torch.sqrt,
        ttnn.UnaryOpType.EQZ: lambda x: x == 0,
        ttnn.UnaryOpType.NEZ: lambda x: x != 0,
        ttnn.UnaryOpType.GTZ: lambda x: x > 0,
        ttnn.UnaryOpType.LTZ: lambda x: x < 0,
        ttnn.UnaryOpType.GEZ: lambda x: x >= 0,
        ttnn.UnaryOpType.LEZ: lambda x: x <= 0,
        ttnn.UnaryOpType.SQUARE: lambda x: (
            integer_golden.binary(x, x, torch.mul) if integer_golden.is_unsigned_dtype(x.dtype) else torch.square(x)
        ),
    }

    if activations is not None:
        for activation in activations:
            # The API accepts either a bare enum or a parameter descriptor for fused activations.
            # Normalize both forms before dispatching so comparison goldens match device-side overloads.
            activation_type = getattr(activation, "op_type", activation)
            if activation_type in (ttnn.UnaryOpType.POWER, ttnn.UnaryOpType.POWER_ITERATIVE):
                params = getattr(activation, "params", ())
                if not params:
                    raise ValueError(f"{activation_type} requires an exponent parameter")
                # Both device power variants have the same mathematical host reference; iterative
                # execution affects implementation accuracy, not the golden function's definition.
                tensor = torch.pow(tensor, params[0])
            else:
                activation_function = act_func_map[activation_type]
                tensor = activation_function(tensor)
    return tensor


def _golden_function(input_tensor_a, input_tensor_b, *args, activations=None, **kwargs):
    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # PyTorch lacks unsigned arithmetic kernels; widen and restore TT wraparound.
        output_tensor = integer_golden.binary(input_tensor_a, input_tensor_b, lambda a, b: a + b)
    else:
        output_tensor = input_tensor_a + input_tensor_b
    return apply_activations(output_tensor, activations)


ttnn.attach_golden_function(ttnn.add, golden_function=_golden_function)
ttnn.attach_golden_function(ttnn.add_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, activations=None, **kwargs):
    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # PyTorch lacks unsigned arithmetic kernels; widen and restore TT wraparound.
        output_tensor = integer_golden.binary(input_tensor_a, input_tensor_b, lambda a, b: a - b)
    else:
        output_tensor = input_tensor_a - input_tensor_b
    return apply_activations(output_tensor, activations)


ttnn.attach_golden_function(ttnn.subtract, golden_function=_golden_function)
ttnn.attach_golden_function(ttnn.subtract_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, activations=None, **kwargs):
    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # PyTorch lacks unsigned arithmetic kernels; widen and restore TT wraparound.
        output_tensor = integer_golden.binary(input_tensor_a, input_tensor_b, lambda a, b: b - a)
    else:
        output_tensor = input_tensor_b - input_tensor_a
    return apply_activations(output_tensor, activations)


ttnn.attach_golden_function(ttnn.rsub, golden_function=_golden_function)
ttnn.attach_golden_function(ttnn.rsub_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, activations=None, **kwargs):
    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # PyTorch lacks unsigned arithmetic kernels; widen and restore TT wraparound.
        output_tensor = integer_golden.binary(input_tensor_a, input_tensor_b, lambda a, b: a * b)
    else:
        output_tensor = input_tensor_a * input_tensor_b
    return apply_activations(output_tensor, activations)


ttnn.attach_golden_function(ttnn.multiply, golden_function=_golden_function)
ttnn.attach_golden_function(ttnn.multiply_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.eq(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.eq, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.ne(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.ne, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Compare widened unsigned values because PyTorch has no UInt16/UInt32 kernel.
        return integer_golden.compare(input_tensor_a, input_tensor_b, torch.gt)
    return torch.gt(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.gt, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Compare widened unsigned values because PyTorch has no UInt16/UInt32 kernel.
        return integer_golden.compare(input_tensor_a, input_tensor_b, torch.ge)
    return torch.ge(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.ge, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Compare widened unsigned values because PyTorch has no UInt16/UInt32 kernel.
        return integer_golden.compare(input_tensor_a, input_tensor_b, torch.lt)
    return torch.lt(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.lt, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Compare widened unsigned values because PyTorch has no UInt16/UInt32 kernel.
        return integer_golden.compare(input_tensor_a, input_tensor_b, torch.le)
    return torch.le(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.le, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Preserve in-place golden state while evaluating unsupported unsigned logic.
        input_tensor_a.copy_(integer_golden.logical(input_tensor_a, input_tensor_b, torch.logical_and))
        return input_tensor_a
    return input_tensor_a.logical_and_(input_tensor_b)


ttnn.attach_golden_function(ttnn.logical_and_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Preserve in-place golden state while evaluating unsupported unsigned logic.
        input_tensor_a.copy_(integer_golden.logical(input_tensor_a, input_tensor_b, torch.logical_or))
        return input_tensor_a
    return input_tensor_a.logical_or_(input_tensor_b)


ttnn.attach_golden_function(ttnn.logical_or_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Preserve in-place golden state while evaluating unsupported unsigned logic.
        input_tensor_a.copy_(integer_golden.logical(input_tensor_a, input_tensor_b, torch.logical_xor))
        return input_tensor_a
    return input_tensor_a.logical_xor_(input_tensor_b)


ttnn.attach_golden_function(ttnn.logical_xor_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.ldexp(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.ldexp, golden_function=_golden_function)
ttnn.attach_golden_function(ttnn.ldexp_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.logaddexp(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.logaddexp, golden_function=_golden_function)
ttnn.attach_golden_function(ttnn.logaddexp_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.logaddexp2(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.logaddexp2, golden_function=_golden_function)
ttnn.attach_golden_function(ttnn.logaddexp2_, golden_function=_golden_function)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.divide(input_tensor_a, input_tensor_b)


# `divide` and `divide_` are separate registered operations and both need the same reference.
ttnn.attach_golden_function(ttnn.divide, golden_function=_golden_function)
ttnn.attach_golden_function(ttnn.divide_, golden_function=_golden_function)


def _golden_function_assign(input_tensor_a, input_tensor_b=None, *args, **kwargs):
    # Both assign overloads return the source values, independently of the destination storage.
    return input_tensor_a.clone()


ttnn.attach_golden_function(ttnn.assign, golden_function=_golden_function_assign)


def _preprocess_broadcast_golden_inputs(function_args, function_kwargs):
    input_tensor = function_args[0] if function_args else function_kwargs["input_tensor"]
    input_tensors = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(input_tensor)]

    function_args = list(function_args)
    function_kwargs = dict(function_kwargs)
    if function_args:
        function_args[0] = input_tensors
    else:
        function_kwargs["input_tensor"] = input_tensors
    function_kwargs["_golden_mesh_shape"] = tuple(input_tensor.device().shape)
    return tuple(function_args), function_kwargs


def _golden_function_broadcast(input_tensors, sender_coord, *args, _golden_mesh_shape=None, **kwargs):
    if _golden_mesh_shape is None:
        return None

    sender_index = 0
    for coordinate, dimension in zip(sender_coord, _golden_mesh_shape):
        sender_index = sender_index * dimension + int(coordinate)
    return input_tensors[sender_index]


ttnn.attach_golden_function(
    ttnn.broadcast,
    golden_function=_golden_function_broadcast,
    preprocess_golden_function_inputs=_preprocess_broadcast_golden_inputs,
)


def _golden_function(a, b, *args, **kwargs):
    import torch

    return torch.nn.functional.gelu(torch.add(a, b))


ttnn.attach_golden_function(ttnn.bias_gelu, golden_function=_golden_function)
ttnn.attach_golden_function(ttnn.bias_gelu_, golden_function=_golden_function)


def _golden_function_squared_difference(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Widen unsigned subtraction and square before restoring TTNN wraparound.
        return integer_golden.binary(input_tensor_a, input_tensor_b, lambda a, b: torch.square(torch.sub(a, b)))
    return torch_squared_difference(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.squared_difference, golden_function=_golden_function_squared_difference)
ttnn.attach_golden_function(ttnn.squared_difference_, golden_function=_golden_function_squared_difference)


def _golden_function_addalpha(input_tensor_a, input_tensor_b, alpha, *args, **kwargs):
    import torch

    return torch.add(input_tensor_a, input_tensor_b, alpha=alpha)


ttnn.attach_golden_function(ttnn.addalpha, golden_function=_golden_function_addalpha)


def _golden_function_subalpha(input_tensor_a, input_tensor_b, alpha, *args, **kwargs):
    import torch

    return torch.sub(input_tensor_a, input_tensor_b, alpha=alpha)


ttnn.attach_golden_function(ttnn.subalpha, golden_function=_golden_function_subalpha)


def _golden_function_xlogy(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.xlogy(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.xlogy, golden_function=_golden_function_xlogy)


def _golden_function_hypot(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.hypot(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.hypot, golden_function=_golden_function_hypot)


def _golden_function_situ_glu(gate, up, beta1, beta2, *args, **kwargs):
    import torch

    g = gate.to(torch.float32)
    u = up.to(torch.float32)
    situ_a = beta1 * torch.tanh(g / beta1) * torch.sigmoid(g)
    up_half = beta2 * torch.tanh(u / beta2)
    # Stays fp32, like the other goldens here: rounding it to the input dtype would make a ULP
    # comparison measure the golden's own rounding as much as the kernel's error.
    return situ_a * up_half


ttnn.attach_golden_function(ttnn.situ_glu, golden_function=_golden_function_situ_glu)


def _golden_function_maximum(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Evaluate unsupported unsigned min/max in int64 and restore the input dtype.
        return integer_golden.binary(input_tensor_a, input_tensor_b, torch.maximum)
    if not torch.is_tensor(input_tensor_b):
        # PyTorch maximum requires two tensors even though TTNN accepts a scalar operand.
        input_tensor_b = torch.tensor(input_tensor_b, dtype=input_tensor_a.dtype, device=input_tensor_a.device)
    return torch.maximum(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.maximum, golden_function=_golden_function_maximum)


def _golden_function_minimum(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Evaluate unsupported unsigned min/max in int64 and restore the input dtype.
        return integer_golden.binary(input_tensor_a, input_tensor_b, torch.minimum)
    if not torch.is_tensor(input_tensor_b):
        # PyTorch minimum requires two tensors even though TTNN accepts a scalar operand.
        input_tensor_b = torch.tensor(input_tensor_b, dtype=input_tensor_a.dtype, device=input_tensor_a.device)
    return torch.minimum(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.minimum, golden_function=_golden_function_minimum)


def _golden_function_logical_xor(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Convert unsigned operands to truth values before applying host logical kernels.
        return integer_golden.logical(input_tensor_a, input_tensor_b, torch.logical_xor)
    return torch.logical_xor(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.logical_xor, golden_function=_golden_function_logical_xor)


def _golden_function_logical_and(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Convert unsigned operands to truth values before applying host logical kernels.
        return integer_golden.logical(input_tensor_a, input_tensor_b, torch.logical_and)
    return torch.logical_and(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.logical_and, golden_function=_golden_function_logical_and)


def _golden_function_logical_or(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Convert unsigned operands to truth values before applying host logical kernels.
        return integer_golden.logical(input_tensor_a, input_tensor_b, torch.logical_or)
    return torch.logical_or(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.logical_or, golden_function=_golden_function_logical_or)


def _golden_function_atan2(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.atan2(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.atan2, golden_function=_golden_function_atan2)


def _golden_function_nextafter(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.nextafter(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.nextafter, golden_function=_golden_function_nextafter)


def _golden_function_isclose(input_tensor_a, input_tensor_b, *args, rtol=1e-05, atol=1e-08, equal_nan=False, **kwargs):
    import torch

    if torch.is_tensor(input_tensor_b) and input_tensor_a.dtype != input_tensor_b.dtype:
        common_dtype = torch.promote_types(input_tensor_a.dtype, input_tensor_b.dtype)
        input_tensor_a = input_tensor_a.to(common_dtype)
        input_tensor_b = input_tensor_b.to(common_dtype)
    return torch.isclose(input_tensor_a, input_tensor_b, rtol=rtol, atol=atol, equal_nan=equal_nan)


ttnn.attach_golden_function(ttnn.isclose, golden_function=_golden_function_isclose)


def _golden_function_div(input_tensor_a, input_tensor_b, rounding_mode=None, *args, **kwargs):
    import torch

    if input_tensor_a.dtype == torch.int32 and rounding_mode in ("trunc", "floor"):
        # Widen integer division so the golden does not take the float32 quotient path.
        wide_input_b = input_tensor_b.to(torch.int64) if torch.is_tensor(input_tensor_b) else input_tensor_b
        return torch.div(input_tensor_a.to(torch.int64), wide_input_b, rounding_mode=rounding_mode).to(torch.int32)
    return torch.div(input_tensor_a, input_tensor_b, rounding_mode=rounding_mode)


ttnn.attach_golden_function(ttnn.div, golden_function=_golden_function_div)


def _golden_function_div_no_nan(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if isinstance(input_tensor_b, float):
        if input_tensor_b == 0:
            return torch.zeros_like(input_tensor_a)
        else:
            return input_tensor_a / input_tensor_b
    else:
        return torch.where(input_tensor_b == 0, 0, input_tensor_a / input_tensor_b)


ttnn.attach_golden_function(ttnn.div_no_nan, golden_function=_golden_function_div_no_nan)


def _golden_function_floor_div(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.floor_divide(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.floor_div, golden_function=_golden_function_floor_div)


def _golden_function_remainder(input_tensor_a, input_tensor_b, *args, device=None, **kwargs):
    import torch

    # Comparison mode does not inject device, and this reference does not use it.
    input_dtype = input_tensor_a.dtype
    if integer_golden.is_unsigned_dtype(input_dtype):
        # PyTorch lacks unsigned remainder kernels; widen and restore the input dtype.
        return integer_golden.binary(input_tensor_a, input_tensor_b, torch.remainder)
    if not torch.is_tensor(input_tensor_b):
        if input_dtype == torch.bfloat16:
            input_tensor_a = input_tensor_a.float()

    result = torch.remainder(input_tensor_a, input_tensor_b)

    if input_dtype == torch.bfloat16:
        result = result.bfloat16()
    return result


ttnn.attach_golden_function(ttnn.remainder, golden_function=_golden_function_remainder)


def _golden_function_fmod(input_tensor_a, input_tensor_b, *args, device=None, **kwargs):
    import torch

    # Comparison-mode golden calls do not provide the unused device argument.
    if not torch.is_tensor(input_tensor_b):
        input_dtype = input_tensor_a.dtype
        if input_dtype == torch.bfloat16:
            input_tensor_a = input_tensor_a.float()
        result = torch.fmod(input_tensor_a, input_tensor_b)
        if input_dtype == torch.bfloat16:
            result = result.bfloat16()
    else:
        result = torch.fmod(input_tensor_a, input_tensor_b)

    return result


ttnn.attach_golden_function(ttnn.fmod, golden_function=_golden_function_fmod)


def torch_squared_difference(x, y, *args, **kwargs):
    import torch

    return torch.square(torch.sub(x, y))


def _golden_function_outer(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if input_tensor_a.dim() == 1 and input_tensor_b.dim() == 1:
        return torch.outer(input_tensor_a, input_tensor_b)
    return torch.einsum("...i,...j->...ij", input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.outer, golden_function=_golden_function_outer)


def _golden_function_polyval(input_tensor_a, coeffs, *args, **kwargs):
    result = 0.0
    for coeff in coeffs:
        result = result * input_tensor_a + coeff
    return result


ttnn.attach_golden_function(ttnn.polyval, golden_function=_golden_function_polyval)


def _golden_function_gt_(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Preserve in-place golden state while comparing widened unsigned values.
        input_tensor_a.copy_(integer_golden.compare(input_tensor_a, input_tensor_b, torch.gt))
        return input_tensor_a
    return input_tensor_a.gt_(input_tensor_b)


ttnn.attach_golden_function(ttnn.gt_, golden_function=_golden_function_gt_)


def _golden_function_le_(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Preserve in-place golden state while comparing widened unsigned values.
        input_tensor_a.copy_(integer_golden.compare(input_tensor_a, input_tensor_b, torch.le))
        return input_tensor_a
    return input_tensor_a.le_(input_tensor_b)


ttnn.attach_golden_function(ttnn.le_, golden_function=_golden_function_le_)


def _golden_function_lt_(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Preserve in-place golden state while comparing widened unsigned values.
        input_tensor_a.copy_(integer_golden.compare(input_tensor_a, input_tensor_b, torch.lt))
        return input_tensor_a
    return input_tensor_a.lt_(input_tensor_b)


ttnn.attach_golden_function(ttnn.lt_, golden_function=_golden_function_lt_)


def _golden_function_ge_(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Preserve in-place golden state while comparing widened unsigned values.
        input_tensor_a.copy_(integer_golden.compare(input_tensor_a, input_tensor_b, torch.ge))
        return input_tensor_a
    return input_tensor_a.ge_(input_tensor_b)


ttnn.attach_golden_function(ttnn.ge_, golden_function=_golden_function_ge_)


def _golden_function_eq_(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return input_tensor_a.eq_(input_tensor_b)


ttnn.attach_golden_function(ttnn.eq_, golden_function=_golden_function_eq_)


def _golden_function_ne_(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return input_tensor_a.ne_(input_tensor_b)


ttnn.attach_golden_function(ttnn.ne_, golden_function=_golden_function_ne_)


def _golden_function_gcd(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.gcd(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.gcd, golden_function=_golden_function_gcd)


def _golden_function_lcm(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    return torch.lcm(input_tensor_a, input_tensor_b)


ttnn.attach_golden_function(ttnn.lcm, golden_function=_golden_function_lcm)


def _golden_function_prelu(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if not torch.is_tensor(input_tensor_b):
        input_tensor_b = torch.tensor(input_tensor_b, dtype=input_tensor_a.dtype)

    return torch.nn.functional.prelu(input_tensor_a, weight=input_tensor_b)


ttnn.attach_golden_function(ttnn.prelu, golden_function=_golden_function_prelu)


def _golden_function_logical_right_shift(input_tensor_a, shift_amt, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Unsigned right shift uses widened non-negative values and dtype-width masking.
        return integer_golden.shift(input_tensor_a, shift_amt, torch.bitwise_right_shift)
    t1_uint = input_tensor_a.to(torch.int64) & 0xFFFFFFFF
    result = (t1_uint >> shift_amt).to(torch.int32)
    return result


ttnn.attach_golden_function(ttnn.logical_right_shift, golden_function=_golden_function_logical_right_shift)

__all__ = []
