# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional

import sys

import ttnn
from ttnn.operations import integer_golden

__all__ = []

# Scalar BF16 add follows the device test contract of at most one ULP.
_DEGENERATE_SCALAR_ADD_ULP_THRESHOLD = 1
# Other low-precision scalar arithmetic uses the empirically validated three-ULP
# degenerate-output bound shared with squared_difference.
_DEGENERATE_SCALAR_ARITHMETIC_ULP_THRESHOLD = 3


def _preprocess_binary_golden_function_inputs(function_args, function_kwargs):
    """Normalize binary golden arguments and retain TT dtype metadata.
    Records aliases so global comparison can reconstruct canonical keyword names.
    """

    function_args = tuple(function_args)
    original_kwargs = function_kwargs
    golden_args, golden_kwargs = ttnn.decorators.default_preprocess_golden_function_inputs(
        function_args, function_kwargs
    )

    argument_aliases = {
        "input_a": "input_tensor_a",
        "input_b": "input_tensor_b",
        "value": "input_tensor_b",
    }
    for alias, canonical_name in argument_aliases.items():
        if alias in golden_kwargs and canonical_name not in golden_kwargs:
            golden_kwargs[canonical_name] = golden_kwargs.pop(alias)
    golden_kwargs["_ttnn_golden_argument_aliases"] = argument_aliases

    input_tensor_a = (
        function_args[0] if function_args else original_kwargs.get("input_tensor_a", original_kwargs.get("input_a"))
    )
    input_tensor_b = (
        function_args[1]
        if len(function_args) > 1
        else original_kwargs.get(
            "input_tensor_b",
            original_kwargs.get("input_b", original_kwargs.get("value")),
        )
    )
    output_tensor = original_kwargs.get("output_tensor")
    golden_kwargs["_ttnn_input_tensor_a_dtype"] = getattr(input_tensor_a, "dtype", None)
    golden_kwargs["_ttnn_input_tensor_b_dtype"] = getattr(input_tensor_b, "dtype", None)
    golden_kwargs["_ttnn_output_tensor_dtype"] = getattr(output_tensor, "dtype", None)
    return golden_args, golden_kwargs


def _set_binary_scalar_comparison_config(
    output_tensor,
    input_tensor_b,
    *,
    ulp_threshold,
    _ttnn_input_tensor_a_dtype=None,
    _ttnn_output_tensor_dtype=None,
):
    """Configure scalar binary comparisons for low-precision output contracts.
    Selects BF8 allclose tolerances or a degenerate-case ULP threshold.
    """

    import torch

    if _ttnn_input_tensor_a_dtype == ttnn.bfloat8_b or _ttnn_output_tensor_dtype == ttnn.bfloat8_b:
        # BF8 scalar and output-buffer results use block quantization, so BF16 ULP is not meaningful.
        # Keep PCC for normal outputs and use the suite's direct tolerance only when PCC is degenerate.
        ttnn.decorators.set_golden_comparison_config(
            output_tensor, method="allclose", scope="degenerate", rtol=0.4, atol=0.35
        )
    elif (not hasattr(input_tensor_b, "shape") or getattr(input_tensor_b, "ndim", 1) == 0) and output_tensor.dtype in (
        torch.bfloat16,
        torch.float16,
    ):
        ttnn.decorators.set_golden_comparison_config(
            output_tensor, method="ulp", scope="degenerate", ulp_threshold=ulp_threshold
        )
    return output_tensor


def _copy_inplace_golden_result(input_tensor_a, output_tensor):
    """Copy a computed golden result back to its caller-visible input alias.
    Propagates or clears the attached comparison policy with the result.
    """

    # Positional in-place operations mutate the first operand even though their host arithmetic is out of place.
    # Copy the computed value and its comparison contract back so the global golden follows the caller-visible alias.
    input_tensor_a.copy_(output_tensor)
    comparison_config = getattr(output_tensor, "_ttnn_comparison_config", None)
    if comparison_config is not None:
        input_tensor_a._ttnn_comparison_config = comparison_config
    elif hasattr(input_tensor_a, "_ttnn_comparison_config"):
        del input_tensor_a._ttnn_comparison_config
    return input_tensor_a


def apply_activations(tensor, activations, reference_tensor=None):
    import torch

    if activations and not torch.is_tensor(tensor):
        tensor = torch.as_tensor(
            tensor,
            dtype=getattr(reference_tensor, "dtype", None),
            device=getattr(reference_tensor, "device", None),
        )

    def compare_zero(value, torch_function):
        """Compare tensor values with zero using dtype-compatible semantics.
        Widens unsupported unsigned Torch dtypes before applying the comparison.
        """

        if integer_golden.is_unsigned_dtype(value.dtype):
            return integer_golden.compare(value, 0, torch_function)
        return torch_function(value, 0)

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
        ttnn.UnaryOpType.GTZ: lambda x: compare_zero(x, torch.gt),
        ttnn.UnaryOpType.LTZ: lambda x: compare_zero(x, torch.lt),
        ttnn.UnaryOpType.GEZ: lambda x: compare_zero(x, torch.ge),
        ttnn.UnaryOpType.LEZ: lambda x: compare_zero(x, torch.le),
        ttnn.UnaryOpType.SQUARE: lambda x: (
            integer_golden.binary(x, x, torch.mul) if integer_golden.is_unsigned_dtype(x.dtype) else torch.square(x)
        ),
    }

    if activations is not None:
        for activation in activations:
            # The API accepts either a bare enum or a parameter descriptor for fused activations.
            # Normalize both forms before dispatching so comparison goldens match device-side overloads.
            activation_type = getattr(activation, "op_type", activation)
            if activation_type == ttnn.UnaryOpType.NEG:
                tensor = (
                    integer_golden.binary(tensor, 0, lambda value, _: -value)
                    if integer_golden.is_unsigned_dtype(tensor.dtype)
                    else torch.neg(tensor)
                )
            elif activation_type in (ttnn.UnaryOpType.POWER, ttnn.UnaryOpType.POWER_ITERATIVE):
                params = getattr(activation, "params", ())
                if not params:
                    raise ValueError(f"{activation_type} requires an exponent parameter")
                # Both device power variants have the same mathematical host reference; iterative
                # execution affects implementation accuracy, not the golden function's definition.
                tensor = (
                    integer_golden.power(tensor, params[0])
                    if integer_golden.is_unsigned_dtype(tensor.dtype)
                    else torch.pow(tensor, params[0])
                )
            elif activation_type == ttnn.UnaryOpType.RELU_MAX:
                params = getattr(activation, "params", ())
                if not params:
                    raise ValueError(f"{activation_type} requires a maximum parameter")
                tensor = (
                    integer_golden.clamp(tensor, min_value=0, max_value=params[0])
                    if integer_golden.is_unsigned_dtype(tensor.dtype)
                    else torch.clamp(tensor, min=0, max=params[0])
                )
            else:
                activation_function = act_func_map[activation_type]
                tensor = activation_function(tensor)
    return tensor


def _golden_function_add(
    input_tensor_a,
    input_tensor_b,
    *args,
    activations=None,
    input_tensor_a_activations=None,
    input_tensor_b_activations=None,
    _ttnn_input_tensor_a_dtype=None,
    _ttnn_output_tensor_dtype=None,
    **kwargs,
):
    # Binary kernels apply operand activations before the elementwise operation and
    # result activations afterward. Mirror that order in comparison-mode goldens.
    input_tensor_a = apply_activations(input_tensor_a, input_tensor_a_activations)
    input_tensor_b = apply_activations(input_tensor_b, input_tensor_b_activations, input_tensor_a)
    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # PyTorch lacks unsigned arithmetic kernels; widen and restore TT wraparound.
        output_tensor = integer_golden.binary(input_tensor_a, input_tensor_b, lambda a, b: a + b)
    else:
        output_tensor = input_tensor_a + input_tensor_b
    output_tensor = apply_activations(output_tensor, activations)
    return _set_binary_scalar_comparison_config(
        output_tensor,
        input_tensor_b,
        ulp_threshold=_DEGENERATE_SCALAR_ADD_ULP_THRESHOLD,
        _ttnn_input_tensor_a_dtype=_ttnn_input_tensor_a_dtype,
        _ttnn_output_tensor_dtype=_ttnn_output_tensor_dtype,
    )


def _golden_function_add_(input_tensor_a, input_tensor_b, *args, _ttnn_global_golden=False, **kwargs):
    output_tensor = _golden_function_add(input_tensor_a, input_tensor_b, *args, **kwargs)
    return _copy_inplace_golden_result(input_tensor_a, output_tensor) if _ttnn_global_golden else output_tensor


_golden_function_add_._ttnn_mutates_global_inputs = True


ttnn.attach_golden_function(
    ttnn.add,
    golden_function=_golden_function_add,
    preprocess_golden_function_inputs=_preprocess_binary_golden_function_inputs,
)
ttnn.attach_golden_function(
    ttnn.add_,
    golden_function=_golden_function_add_,
    preprocess_golden_function_inputs=_preprocess_binary_golden_function_inputs,
)


def _golden_function_subtract(
    input_tensor_a,
    input_tensor_b,
    *args,
    activations=None,
    input_tensor_a_activations=None,
    input_tensor_b_activations=None,
    _ttnn_input_tensor_a_dtype=None,
    _ttnn_output_tensor_dtype=None,
    **kwargs,
):
    input_tensor_a = apply_activations(input_tensor_a, input_tensor_a_activations)
    input_tensor_b = apply_activations(input_tensor_b, input_tensor_b_activations, input_tensor_a)
    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # PyTorch lacks unsigned arithmetic kernels; widen and restore TT wraparound.
        output_tensor = integer_golden.binary(input_tensor_a, input_tensor_b, lambda a, b: a - b)
    else:
        output_tensor = input_tensor_a - input_tensor_b
    output_tensor = apply_activations(output_tensor, activations)
    return _set_binary_scalar_comparison_config(
        output_tensor,
        input_tensor_b,
        ulp_threshold=_DEGENERATE_SCALAR_ARITHMETIC_ULP_THRESHOLD,
        _ttnn_input_tensor_a_dtype=_ttnn_input_tensor_a_dtype,
        _ttnn_output_tensor_dtype=_ttnn_output_tensor_dtype,
    )


def _golden_function_subtract_(input_tensor_a, input_tensor_b, *args, _ttnn_global_golden=False, **kwargs):
    output_tensor = _golden_function_subtract(input_tensor_a, input_tensor_b, *args, **kwargs)
    return _copy_inplace_golden_result(input_tensor_a, output_tensor) if _ttnn_global_golden else output_tensor


_golden_function_subtract_._ttnn_mutates_global_inputs = True


ttnn.attach_golden_function(
    ttnn.subtract,
    golden_function=_golden_function_subtract,
    preprocess_golden_function_inputs=_preprocess_binary_golden_function_inputs,
)
ttnn.attach_golden_function(
    ttnn.subtract_,
    golden_function=_golden_function_subtract_,
    preprocess_golden_function_inputs=_preprocess_binary_golden_function_inputs,
)


def _golden_function_rsub(
    input_tensor_a,
    input_tensor_b,
    *args,
    activations=None,
    input_tensor_a_activations=None,
    input_tensor_b_activations=None,
    _ttnn_input_tensor_a_dtype=None,
    _ttnn_output_tensor_dtype=None,
    **kwargs,
):
    input_tensor_a = apply_activations(input_tensor_a, input_tensor_a_activations)
    input_tensor_b = apply_activations(input_tensor_b, input_tensor_b_activations, input_tensor_a)
    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # PyTorch lacks unsigned arithmetic kernels; widen and restore TT wraparound.
        output_tensor = integer_golden.binary(input_tensor_a, input_tensor_b, lambda a, b: b - a)
    else:
        output_tensor = input_tensor_b - input_tensor_a
    output_tensor = apply_activations(output_tensor, activations)
    return _set_binary_scalar_comparison_config(
        output_tensor,
        input_tensor_b,
        ulp_threshold=_DEGENERATE_SCALAR_ARITHMETIC_ULP_THRESHOLD,
        _ttnn_input_tensor_a_dtype=_ttnn_input_tensor_a_dtype,
        _ttnn_output_tensor_dtype=_ttnn_output_tensor_dtype,
    )


def _golden_function_rsub_(input_tensor_a, input_tensor_b, *args, _ttnn_global_golden=False, **kwargs):
    output_tensor = _golden_function_rsub(input_tensor_a, input_tensor_b, *args, **kwargs)
    return _copy_inplace_golden_result(input_tensor_a, output_tensor) if _ttnn_global_golden else output_tensor


_golden_function_rsub_._ttnn_mutates_global_inputs = True


ttnn.attach_golden_function(
    ttnn.rsub,
    golden_function=_golden_function_rsub,
    preprocess_golden_function_inputs=_preprocess_binary_golden_function_inputs,
)
ttnn.attach_golden_function(
    ttnn.rsub_,
    golden_function=_golden_function_rsub_,
    preprocess_golden_function_inputs=_preprocess_binary_golden_function_inputs,
)


def _golden_function_multiply(
    input_tensor_a,
    input_tensor_b,
    *args,
    activations=None,
    input_tensor_a_activations=None,
    input_tensor_b_activations=None,
    _ttnn_input_tensor_a_dtype=None,
    _ttnn_output_tensor_dtype=None,
    **kwargs,
):
    input_tensor_a = apply_activations(input_tensor_a, input_tensor_a_activations)
    input_tensor_b = apply_activations(input_tensor_b, input_tensor_b_activations, input_tensor_a)
    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # PyTorch lacks unsigned arithmetic kernels; widen and restore TT wraparound.
        output_tensor = integer_golden.binary(input_tensor_a, input_tensor_b, lambda a, b: a * b)
    else:
        output_tensor = input_tensor_a * input_tensor_b
    output_tensor = apply_activations(output_tensor, activations)
    return _set_binary_scalar_comparison_config(
        output_tensor,
        input_tensor_b,
        ulp_threshold=_DEGENERATE_SCALAR_ARITHMETIC_ULP_THRESHOLD,
        _ttnn_input_tensor_a_dtype=_ttnn_input_tensor_a_dtype,
        _ttnn_output_tensor_dtype=_ttnn_output_tensor_dtype,
    )


def _golden_function_multiply_(input_tensor_a, input_tensor_b, *args, _ttnn_global_golden=False, **kwargs):
    output_tensor = _golden_function_multiply(input_tensor_a, input_tensor_b, *args, **kwargs)
    return _copy_inplace_golden_result(input_tensor_a, output_tensor) if _ttnn_global_golden else output_tensor


_golden_function_multiply_._ttnn_mutates_global_inputs = True


ttnn.attach_golden_function(
    ttnn.multiply,
    golden_function=_golden_function_multiply,
    preprocess_golden_function_inputs=_preprocess_binary_golden_function_inputs,
)
ttnn.attach_golden_function(
    ttnn.multiply_,
    golden_function=_golden_function_multiply_,
    preprocess_golden_function_inputs=_preprocess_binary_golden_function_inputs,
)


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


def _golden_function_divide(
    input_tensor_a,
    input_tensor_b,
    *args,
    rounding_mode=None,
    fast_and_approximate_mode=False,
    activations=None,
    input_tensor_a_activations=None,
    input_tensor_b_activations=None,
    _ttnn_input_tensor_a_dtype=None,
    _ttnn_output_tensor_dtype=None,
    **kwargs,
):
    import torch

    input_tensor_a = apply_activations(input_tensor_a, input_tensor_a_activations)
    input_tensor_b = apply_activations(input_tensor_b, input_tensor_b_activations, input_tensor_a)

    if args and rounding_mode is None:
        # Direct golden callers may pass rounding_mode as the third positional argument.
        # Normalize it here so their reference follows the same trunc/floor path as the device call.
        rounding_mode = args[0]

    if input_tensor_a.dtype == torch.int32 and rounding_mode in ("trunc", "floor"):
        # Widen integer division so the golden does not take the float32 quotient path.
        wide_input_b = input_tensor_b.to(torch.int64) if torch.is_tensor(input_tensor_b) else input_tensor_b
        output_tensor = torch.div(input_tensor_a.to(torch.int64), wide_input_b, rounding_mode=rounding_mode).to(
            torch.int32
        )
    else:
        output_tensor = torch.divide(input_tensor_a, input_tensor_b, rounding_mode=rounding_mode)
    if (
        input_tensor_a.dtype == torch.bfloat16
        and fast_and_approximate_mode
        and rounding_mode is None
        and bool(torch.any(input_tensor_b == 0) if torch.is_tensor(input_tensor_b) else input_tensor_b == 0)
    ):
        # Fast BF16 division by zero is intentionally outside the operation's numerical contract.
        # Preserve its global golden state, but do not fail before the caller's existing skip is reached.
        output_tensor = apply_activations(output_tensor, activations)
        ttnn.decorators.set_golden_comparison_config(output_tensor, method="skip", scope="all")
        return output_tensor
    output_tensor = apply_activations(output_tensor, activations)
    return _set_binary_scalar_comparison_config(
        output_tensor,
        input_tensor_b,
        ulp_threshold=_DEGENERATE_SCALAR_ARITHMETIC_ULP_THRESHOLD,
        _ttnn_input_tensor_a_dtype=_ttnn_input_tensor_a_dtype,
        _ttnn_output_tensor_dtype=_ttnn_output_tensor_dtype,
    )


def _golden_function_divide_(input_tensor_a, input_tensor_b, *args, _ttnn_global_golden=False, **kwargs):
    output_tensor = _golden_function_divide(input_tensor_a, input_tensor_b, *args, **kwargs)
    return _copy_inplace_golden_result(input_tensor_a, output_tensor) if _ttnn_global_golden else output_tensor


_golden_function_divide_._ttnn_mutates_global_inputs = True


ttnn.attach_golden_function(
    ttnn.divide,
    golden_function=_golden_function_divide,
    preprocess_golden_function_inputs=_preprocess_binary_golden_function_inputs,
)
ttnn.attach_golden_function(
    ttnn.divide_,
    golden_function=_golden_function_divide_,
    preprocess_golden_function_inputs=_preprocess_binary_golden_function_inputs,
)


def _golden_function_assign(
    input_tensor=None,
    *args,
    memory_config=None,
    dtype=None,
    output_tensor=None,
    input_a=None,
    input_b=None,
    **kwargs,
):
    # Accommodate both nanobind overloads while preserving their public argument names.
    source_tensor = input_tensor if input_tensor is not None else input_a
    if source_tensor is None:
        raise TypeError("ttnn.assign golden requires input_tensor or input_a")
    if args and input_b is None:
        input_b = args[0]

    if dtype is not None:
        from ttnn.operations.core import _typecast_golden_function

        return _typecast_golden_function(source_tensor, output_dtype=dtype)

    # The destination overload casts to input_b's storage dtype.
    return source_tensor.to(input_b.dtype) if input_b is not None else source_tensor.clone()


ttnn.attach_golden_function(ttnn.assign, golden_function=_golden_function_assign)


def _preprocess_broadcast_golden_inputs(function_args, function_kwargs):
    """Convert a mesh broadcast input into per-device Torch shards.
    Adds mesh metadata required by local and global golden comparison paths.
    """

    input_tensor = function_args[0] if function_args else function_kwargs["input_tensor"]
    input_tensors = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(input_tensor)]

    function_args = list(function_args)
    function_kwargs = dict(function_kwargs)
    if function_args:
        function_args[0] = input_tensors
    else:
        function_kwargs["input_tensor"] = input_tensors
    function_kwargs["_ttnn_golden_mesh_shape"] = tuple(input_tensor.device().shape)
    function_kwargs["_ttnn_global_golden_mesh_shards"] = True
    return tuple(function_args), function_kwargs


def _golden_function_broadcast(input_tensor, sender_coord, *args, _ttnn_golden_mesh_shape=None, **kwargs):
    if _ttnn_golden_mesh_shape is None:
        return None

    sender_index = 0
    for coordinate, dimension in zip(sender_coord, _ttnn_golden_mesh_shape):
        sender_index = sender_index * dimension + int(coordinate)
    return input_tensor[sender_index]


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


def _golden_function_squared_difference(
    input_tensor_a,
    input_tensor_b,
    *args,
    activations=None,
    input_tensor_a_activations=None,
    input_tensor_b_activations=None,
    _ttnn_input_tensor_a_dtype=None,
    _ttnn_output_tensor_dtype=None,
    **kwargs,
):
    import torch

    input_tensor_a = apply_activations(input_tensor_a, input_tensor_a_activations)
    input_tensor_b = apply_activations(input_tensor_b, input_tensor_b_activations, input_tensor_a)
    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Widen unsigned subtraction and square before restoring TTNN wraparound.
        output_tensor = integer_golden.binary(
            input_tensor_a, input_tensor_b, lambda a, b: torch.square(torch.sub(a, b))
        )
    else:
        output_tensor = torch_squared_difference(input_tensor_a, input_tensor_b)
    output_tensor = apply_activations(output_tensor, activations)
    # Singleton low-precision squared-difference results are validated to three ULP.
    # Mark this golden explicitly instead of weakening all constant-tensor comparisons.
    if _ttnn_input_tensor_a_dtype == ttnn.bfloat8_b or _ttnn_output_tensor_dtype == ttnn.bfloat8_b:
        ttnn.decorators.set_golden_comparison_config(
            output_tensor, method="allclose", scope="degenerate", rtol=0.4, atol=0.35
        )
    elif output_tensor.dtype in (torch.bfloat16, torch.float16):
        ttnn.decorators.set_golden_comparison_config(
            output_tensor,
            method="ulp",
            scope="degenerate",
            ulp_threshold=_DEGENERATE_SCALAR_ARITHMETIC_ULP_THRESHOLD,
        )
    return output_tensor


def _golden_function_squared_difference_(input_tensor_a, input_tensor_b, *args, _ttnn_global_golden=False, **kwargs):
    output_tensor = _golden_function_squared_difference(input_tensor_a, input_tensor_b, *args, **kwargs)
    return _copy_inplace_golden_result(input_tensor_a, output_tensor) if _ttnn_global_golden else output_tensor


_golden_function_squared_difference_._ttnn_mutates_global_inputs = True


ttnn.attach_golden_function(
    ttnn.squared_difference,
    golden_function=_golden_function_squared_difference,
    preprocess_golden_function_inputs=_preprocess_binary_golden_function_inputs,
)
ttnn.attach_golden_function(
    ttnn.squared_difference_,
    golden_function=_golden_function_squared_difference_,
    preprocess_golden_function_inputs=_preprocess_binary_golden_function_inputs,
)


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

    if torch.is_tensor(input_tensor_b) and torch.int32 in (input_tensor_a.dtype, input_tensor_b.dtype):
        # Binary-ng evaluates INT32 isclose through FLOAT32, including mixed INT32/BF16 inputs.
        # Match that promotion explicitly instead of relying on PyTorch's BF16 common dtype.
        input_tensor_a = input_tensor_a.to(torch.float32)
        input_tensor_b = input_tensor_b.to(torch.float32)
    elif torch.is_tensor(input_tensor_b) and input_tensor_a.dtype != input_tensor_b.dtype:
        common_dtype = torch.promote_types(input_tensor_a.dtype, input_tensor_b.dtype)
        input_tensor_a = input_tensor_a.to(common_dtype)
        input_tensor_b = input_tensor_b.to(common_dtype)
    return torch.isclose(input_tensor_a, input_tensor_b, rtol=rtol, atol=atol, equal_nan=equal_nan)


ttnn.attach_golden_function(ttnn.isclose, golden_function=_golden_function_isclose)


# Public ttnn.divide is aliased to ttnn.div after operation modules load.
# Reuse the same scalar, output-buffer, and special-value contract on the surviving operation object.
ttnn.attach_golden_function(
    ttnn.div,
    golden_function=_golden_function_divide,
    preprocess_golden_function_inputs=_preprocess_binary_golden_function_inputs,
)


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


def _golden_function_remainder(
    input_tensor_a,
    input_tensor_b,
    *args,
    device=None,
    activations=None,
    input_tensor_a_activations=None,
    input_tensor_b_activations=None,
    **kwargs,
):
    import torch

    # Remainder follows the same operand-before/result-after activation ordering as binary-ng.
    # Applying it here also preserves unsigned widening before restoring TTNN wraparound.
    input_tensor_a = apply_activations(input_tensor_a, input_tensor_a_activations)
    input_tensor_b = apply_activations(input_tensor_b, input_tensor_b_activations, input_tensor_a)
    input_dtype = input_tensor_a.dtype
    if integer_golden.is_unsigned_dtype(input_dtype):
        result = integer_golden.binary(input_tensor_a, input_tensor_b, torch.remainder)
        return apply_activations(result, activations)
    if not torch.is_tensor(input_tensor_b):
        if input_dtype == torch.bfloat16:
            input_tensor_a = input_tensor_a.float()

    result = torch.remainder(input_tensor_a, input_tensor_b)

    if input_dtype == torch.bfloat16:
        result = result.bfloat16()
    result = apply_activations(result, activations)
    if input_dtype == torch.bfloat16 and not torch.is_tensor(input_tensor_b):
        # Tiny scalar divisors are accepted by absolute error, while zero divisors may encode
        # the same undefined positions as NaN or infinity depending on the device path.
        ttnn.decorators.set_golden_comparison_config(
            result, method="allclose", scope="all", rtol=0.0, atol=0.001, nonfinite="mask"
        )
    return result


ttnn.attach_golden_function(ttnn.remainder, golden_function=_golden_function_remainder)


def _golden_function_fmod(input_tensor_a, input_tensor_b, *args, device=None, **kwargs):
    import torch

    # Comparison-mode golden calls do not provide the unused device argument.
    input_dtype = input_tensor_a.dtype
    if not torch.is_tensor(input_tensor_b):
        if input_dtype == torch.bfloat16:
            input_tensor_a = input_tensor_a.float()
        result = torch.fmod(input_tensor_a, input_tensor_b)
        if input_dtype == torch.bfloat16:
            result = result.bfloat16()
    else:
        result = torch.fmod(input_tensor_a, input_tensor_b)

    if input_dtype == torch.bfloat16 and not torch.is_tensor(input_tensor_b):
        # Scalar BF16 fmod has the same tiny-divisor and zero-divisor contract as remainder.
        # Use direct absolute error and compare only the placement of nonfinite results.
        ttnn.decorators.set_golden_comparison_config(
            result, method="allclose", scope="all", rtol=0.0, atol=0.001, nonfinite="mask"
        )
    return result


ttnn.attach_golden_function(ttnn.fmod, golden_function=_golden_function_fmod)


def torch_squared_difference(x, y, *args, **kwargs):
    import torch

    return torch.square(torch.sub(x, y))


def _golden_function_outer(input_tensor_a, input_tensor_b, *args, _ttnn_input_tensor_a_dtype=None, **kwargs):
    import torch

    if input_tensor_a.dim() == 1 and input_tensor_b.dim() == 1:
        result = torch.outer(input_tensor_a, input_tensor_b)
    else:
        result = torch.einsum("...i,...j->...ij", input_tensor_a, input_tensor_b)
    if _ttnn_input_tensor_a_dtype == ttnn.bfloat8_b:
        # A one-element BF8 outer product carries block-float quantization for which PCC is undefined.
        # Retain PCC elsewhere and use the operation's scalar-output tolerance only in that case.
        ttnn.decorators.set_golden_comparison_config(result, method="allclose", scope="degenerate", rtol=0.05, atol=2.0)
    return result


ttnn.attach_golden_function(
    ttnn.outer,
    golden_function=_golden_function_outer,
    preprocess_golden_function_inputs=_preprocess_binary_golden_function_inputs,
)


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
