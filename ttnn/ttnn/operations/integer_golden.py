# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


def is_unsigned_dtype(dtype):
    import torch

    return dtype in (torch.uint16, torch.uint32)


def _bit_width(dtype):
    import torch

    if dtype == torch.uint16:
        return 16
    if dtype == torch.uint32:
        return 32
    raise TypeError(f"Unsupported unsigned dtype: {dtype}")


def _mask(dtype):
    return (1 << _bit_width(dtype)) - 1


def _to_wide(value):
    import torch

    return value.to(torch.int64) if torch.is_tensor(value) else value


def _to_unsigned_scalar(value, dtype):
    import torch

    if isinstance(value, float):
        value = int(torch.tensor(value, dtype=torch.float32).item())
    return int(value) & _mask(dtype)


def _to_unsigned_operand(value, dtype):
    import torch

    if torch.is_tensor(value):
        return _to_wide(value)
    # Torch's maximum/minimum kernels require tensor operands, while TTNN accepts scalars.
    # Materialize scalar operands in the widened dtype so all unsigned golden operations share one path.
    return torch.tensor(_to_unsigned_scalar(value, dtype), dtype=torch.int64)


def restore_unsigned(result, dtype):
    import torch

    return torch.bitwise_and(result, _mask(dtype)).to(dtype)


def binary(input_tensor_a, input_tensor_b, torch_function):
    dtype = input_tensor_a.dtype
    result = torch_function(_to_wide(input_tensor_a), _to_unsigned_operand(input_tensor_b, dtype))
    return restore_unsigned(result, dtype)


def compare(input_tensor_a, input_tensor_b, torch_function):
    dtype = input_tensor_a.dtype
    return torch_function(_to_wide(input_tensor_a), _to_unsigned_operand(input_tensor_b, dtype))


def logical(input_tensor_a, input_tensor_b, torch_function):
    import torch

    lhs = _to_wide(input_tensor_a) != 0
    rhs = _to_wide(input_tensor_b) != 0 if torch.is_tensor(input_tensor_b) else input_tensor_b != 0
    return torch_function(lhs, rhs)


def logical_not(input_tensor):
    return _to_wide(input_tensor) == 0


def clamp(input_tensor, min_value=None, max_value=None):
    import torch

    dtype = input_tensor.dtype
    min_value = None if min_value is None else _to_unsigned_scalar(min_value, dtype)
    max_value = None if max_value is None else _to_unsigned_scalar(max_value, dtype)
    return restore_unsigned(torch.clamp(_to_wide(input_tensor), min_value, max_value), dtype)


def power(input_tensor, exponent, reverse=False):
    import torch

    dtype = input_tensor.dtype
    wide_input = _to_wide(input_tensor)
    if reverse:
        result = torch.pow(_to_unsigned_operand(exponent, dtype), wide_input)
    else:
        result = torch.pow(wide_input, _to_wide(exponent))
    if result.is_floating_point():
        # Integer outputs truncate a floating exponent result before width restoration.
        result = torch.trunc(result).to(torch.int64)
    return restore_unsigned(result, dtype)


def shift(input_tensor, shift_amount, torch_function):
    import torch

    dtype = input_tensor.dtype
    bit_width = _bit_width(dtype)
    wide_input = _to_wide(input_tensor)
    wide_shift = _to_wide(shift_amount)

    if torch.is_tensor(wide_shift):
        valid_shift = (wide_shift >= 0) & (wide_shift < bit_width)
        safe_shift = torch.where(valid_shift, wide_shift, torch.zeros_like(wide_shift))
        result = torch_function(wide_input, safe_shift)
        result = torch.where(valid_shift, result, torch.zeros_like(result))
    elif 0 <= wide_shift < bit_width:
        result = torch_function(wide_input, wide_shift)
    else:
        result = torch.zeros_like(wide_input)

    return restore_unsigned(result, dtype)


def right_shift(input_tensor, shift_amount):
    import torch

    dtype = input_tensor.dtype
    wide_input = _to_wide(input_tensor)
    wide_shift = _to_wide(shift_amount)

    # Unary SFPU right shift clamps counts at 31 and treats UInt32 lanes as signed
    # Int32 bit patterns. Left and binary shifts instead retain zero-on-invalid semantics.
    if dtype == torch.uint32:
        wide_input = torch.where(wide_input >= (1 << 31), wide_input - (1 << 32), wide_input)

    if torch.is_tensor(wide_shift):
        valid_shift = wide_shift >= 0
        effective_shift = torch.clamp(wide_shift, min=0, max=31)
        result = torch.bitwise_right_shift(wide_input, effective_shift)
        result = torch.where(valid_shift, result, torch.zeros_like(result))
    elif wide_shift >= 0:
        result = torch.bitwise_right_shift(wide_input, min(wide_shift, 31))
    else:
        result = torch.zeros_like(wide_input)

    return restore_unsigned(result, dtype)


def addcmul(input_tensor_a, input_tensor_b, input_tensor_c, value):
    dtype = input_tensor_a.dtype
    scalar = _to_unsigned_scalar(value, dtype)
    result = _to_wide(input_tensor_a) + scalar * _to_wide(input_tensor_b) * _to_wide(input_tensor_c)
    return restore_unsigned(result, dtype)
