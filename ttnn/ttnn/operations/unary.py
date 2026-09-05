# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import os
from ttnn.operations import integer_golden


def register_ttnn_cpp_unary_function(unary_function):
    def _golden_function(input_tensor: ttnn.Tensor, *args, **_):
        import torch

        def torch_cbrt(x, *args, **kwargs):
            return torch.sgn(x) * torch.pow(torch.abs(x), 1.0 / 3)

        def torch_multigammaln(x, *args, **kwargs):
            result = torch.lgamma(x)
            result += torch.lgamma(x - 0.5)
            result += torch.lgamma(x - 1.0)
            result += torch.lgamma(x - 1.5)
            result += 3.434189657547
            return result

        def torch_hardmish(x):
            x_f32 = x.to(torch.float32)
            result_f32 = x_f32 * torch.clamp(x_f32 * 0.5 + 1.0, min=0.0, max=1.0)

            if x.dtype == torch.bfloat16:
                # Simulate SFPSTORE truncating
                result_int32 = result_f32.view(torch.int32)
                shifted_int32 = torch.bitwise_right_shift(result_int32, 16)
                truncated_int16 = shifted_int32.to(torch.int16)
                final_result = truncated_int16.view(torch.bfloat16)
            else:
                final_result = result_f32

            return final_result

        def torch_logical_not(x):
            if integer_golden.is_unsigned_dtype(x.dtype):
                # PyTorch lacks unsigned logical-not; compare widened values with zero.
                return integer_golden.logical_not(x)
            return torch.logical_not(x)

        def torch_compare_zero(x, torch_function):
            if integer_golden.is_unsigned_dtype(x.dtype):
                # PyTorch lacks unsigned relational kernels; compare widened values with zero.
                return integer_golden.compare(x, 0, torch_function)
            return torch_function(x, 0)

        def torch_relu(x):
            if integer_golden.is_unsigned_dtype(x.dtype):
                # Unsigned values are non-negative, so ReLU is the identity.
                return x
            return torch.relu(x)

        def torch_relu6(x):
            if integer_golden.is_unsigned_dtype(x.dtype):
                # Evaluate unsigned ReLU6 as a widened clamp and restore its dtype.
                return integer_golden.clamp(x, 0, 6)
            return torch.nn.functional.relu6(x)

        def torch_bitcast(x, dtype):
            # Tensor.view requires the torch dtype corresponding to the TTNN output dtype.
            return x.view(ttnn.ttnn_dtype_to_torch_dtype(dtype))

        name_to_golden_function = {
            "abs": torch.abs,
            "atan": torch.atan,
            "bitcast": torch_bitcast,
            "cos": torch.cos,
            "erfinv": torch.erfinv,
            "exp2": torch.exp2,
            "expm1": torch.expm1,
            "eqz": lambda x: torch.eq(x, 0),
            "floor": torch.floor,
            "ceil": torch.ceil,
            "gez": lambda x: torch_compare_zero(x, torch.ge),
            "gtz": lambda x: torch_compare_zero(x, torch.gt),
            "i0": torch.i0,
            "identity": torch.clone,
            "isfinite": torch.isfinite,
            "isinf": torch.isinf,
            "isnan": torch.isnan,
            "isneginf": torch.isneginf,
            "isposinf": torch.isposinf,
            "lez": lambda x: torch_compare_zero(x, torch.le),
            "log": torch.log,
            "log10": torch.log10,
            "log2": torch.log2,
            "log_sigmoid": torch.nn.functional.logsigmoid,
            "logical_not": torch_logical_not,
            "ltz": lambda x: torch_compare_zero(x, torch.lt),
            "neg": torch.neg,
            "nez": lambda x: torch.ne(x, 0),
            "relu": torch_relu,
            "relu6": torch_relu6,
            "sigmoid": torch.sigmoid,
            "sign": torch.sign,
            "signbit": torch.signbit,
            "silu": torch.nn.functional.silu,
            "sin": torch.sin,
            "sqrt": torch.sqrt,
            # Torch lacks unsigned square kernels; widen through integer_golden to model TTNN wraparound.
            # Signed and floating-point inputs retain Torch's native square implementation.
            "square": lambda x: (
                integer_golden.binary(x, x, torch.mul) if integer_golden.is_unsigned_dtype(x.dtype) else torch.square(x)
            ),
            "tan": torch.tan,
            "tanh": torch.tanh,
            # Unaries with fast_and_approximate_mode
            "exp": torch.exp,
            "erf": torch.erf,
            "erfc": torch.erfc,
            "gelu": torch.nn.functional.gelu,
            "rsqrt": torch.rsqrt,
            # Unaries with float parameter
            # Other unaries (composite operations)
            "softplus": torch.nn.functional.softplus,
            "sigmoid_accurate": torch.sigmoid,
            "asinh": torch.asinh,
            "cbrt": torch_cbrt,
            "cosh": torch.cosh,
            "deg2rad": torch.deg2rad,
            "digamma": torch.digamma,
            "hardswish": torch.nn.functional.hardswish,
            "hardsigmoid": torch.nn.functional.hardsigmoid,
            "lgamma": torch.lgamma,
            "log1p": torch.log1p,
            "mish": lambda _x: torch.nn.functional.mish(_x.to(torch.float)),
            "hardmish": lambda _x: torch_hardmish(_x),
            "multigammaln": torch_multigammaln,
            "rad2deg": torch.rad2deg,
            "sinh": torch.sinh,
            "softsign": torch.nn.functional.softsign,
            "swish": torch.nn.functional.silu,
            "tril": torch.tril,
            "triu": torch.triu,
        }

        golden_keys = set(name_to_golden_function.keys())
        function_names = {function.__name__.split(".")[-1] for function in TTNN_ELTWISE_UNARY_CPP_FUNCTIONS}
        if golden_keys != function_names:
            raise ImportError(
                f"Missing or extra golden functions:\n{golden_keys}\nshould be equal to\n{function_names}"
            )

        torch_function = name_to_golden_function[unary_function.__name__.split(".")[-1]]
        # Preserve operation-specific positional parameters while discarding TTNN-only kwargs.
        return torch_function(input_tensor, *args)

    ttnn.attach_golden_function(unary_function, golden_function=_golden_function)


TTNN_ELTWISE_UNARY_CPP_FUNCTIONS = [
    ttnn.abs,
    ttnn.atan,
    ttnn.bitcast,
    ttnn.cos,
    ttnn.erfinv,
    ttnn.exp2,
    ttnn.expm1,
    ttnn.eqz,
    ttnn.floor,
    ttnn.ceil,
    ttnn.gez,
    ttnn.gtz,
    ttnn.i0,
    ttnn.identity,
    ttnn.isfinite,
    ttnn.isinf,
    ttnn.isnan,
    ttnn.isneginf,
    ttnn.isposinf,
    ttnn.lez,
    ttnn.log,
    ttnn.log10,
    ttnn.log2,
    ttnn.logical_not,
    ttnn.ltz,
    ttnn.neg,
    ttnn.nez,
    ttnn.relu,
    ttnn.relu6,
    ttnn.sigmoid,
    ttnn.sign,
    ttnn.signbit,
    ttnn.silu,
    ttnn.sin,
    ttnn.sqrt,
    ttnn.square,
    ttnn.tan,
    ttnn.tanh,
    # Unaries with fast_and_approximate_mode
    ttnn.exp,
    ttnn.erf,
    ttnn.erfc,
    ttnn.gelu,
    ttnn.rsqrt,
    # Unaries with float parameter
    # Unaries using op_chain
    ttnn.log_sigmoid,
    ttnn.softplus,
    ttnn.sigmoid_accurate,
    # Other unaries (composite operations - tt_eager dependency)
    ttnn.asinh,
    ttnn.cbrt,
    ttnn.cosh,
    ttnn.deg2rad,
    ttnn.digamma,
    ttnn.hardswish,
    ttnn.hardsigmoid,
    ttnn.lgamma,
    ttnn.log1p,
    ttnn.mish,
    ttnn.hardmish,
    ttnn.multigammaln,
    ttnn.rad2deg,
    ttnn.sinh,
    ttnn.softsign,
    ttnn.swish,
    ttnn.tril,
    ttnn.triu,
]
for unary_function in TTNN_ELTWISE_UNARY_CPP_FUNCTIONS:
    register_ttnn_cpp_unary_function(unary_function)


def _golden_function_tanh(input_tensor, *args, **kwargs):
    import torch

    if input_tensor.dtype == torch.bfloat16:
        # Evaluate BF16 tanh with FP32 intermediates, then apply the hardware DAZ/FTZ boundary.
        # Singleton regressions use the documented two-ULP contract where PCC is undefined.
        input_float = input_tensor.to(torch.float32)
        input_float = torch.where(
            torch.abs(input_float) < torch.finfo(torch.bfloat16).tiny,
            torch.zeros_like(input_float),
            input_float,
        )
        result = torch.tanh(input_float).to(torch.bfloat16)
        result = torch.where(
            torch.abs(result.to(torch.float32)) < torch.finfo(torch.bfloat16).tiny,
            torch.zeros_like(result),
            result,
        )
        ttnn.decorators.set_golden_comparison_config(result, method="ulp", scope="degenerate", ulp_threshold=2)
        return result
    return torch.tanh(input_tensor)


def _preprocess_hyperbolic_golden_inputs(function_args, function_kwargs):
    """Preserve block-float input identity for hyperbolic goldens.
    Adds a BF8 flag so output comparison selects the intended policy.
    """

    input_tensor = function_args[0] if function_args else function_kwargs["input_tensor"]
    golden_args, golden_kwargs = ttnn.decorators.default_preprocess_golden_function_inputs(
        function_args, function_kwargs
    )
    # Default preprocessing exposes BFLOAT8_B values as float32, which otherwise selects the FP32 ULP contract.
    # Retain the source dtype so block-float outputs continue to use their established PCC comparison.
    golden_kwargs["_ttnn_input_is_bfloat8_b"] = input_tensor.dtype == ttnn.bfloat8_b
    return golden_args, golden_kwargs


def _golden_function_sinh(input_tensor, *args, _ttnn_input_is_bfloat8_b=False, **kwargs):
    import torch

    if input_tensor.dtype != torch.float32:
        return torch.sinh(input_tensor)
    # Float32 sinh can overflow during evaluation even when the correctly rounded result is finite.
    # Evaluate in float64, round once to float32, and compare against the full-tensor three-ULP contract.
    # Flush float32 subnormal outputs to the zero produced by the SFPU FTZ path.
    result = torch.sinh(input_tensor.to(torch.float64)).to(torch.float32)
    result = torch.where(torch.abs(result) < torch.finfo(torch.float32).tiny, torch.zeros_like(result), result)
    if not _ttnn_input_is_bfloat8_b:
        ttnn.decorators.set_golden_comparison_config(result, method="ulp", scope="all", ulp_threshold=3)
    return result


def _golden_function_cosh(input_tensor, *args, _ttnn_input_is_bfloat8_b=False, **kwargs):
    import torch

    if input_tensor.dtype != torch.float32:
        return torch.cosh(input_tensor)
    # Match the overflow-safe reference by evaluating cosh in float64 before the output cast.
    # Its kernel contract is one ULP across finite values with exact nonfinite placement.
    # Flush any subnormal result before comparison to mirror the SFPU FTZ path.
    result = torch.cosh(input_tensor.to(torch.float64)).to(torch.float32)
    result = torch.where(torch.abs(result) < torch.finfo(torch.float32).tiny, torch.zeros_like(result), result)
    if not _ttnn_input_is_bfloat8_b:
        ttnn.decorators.set_golden_comparison_config(result, method="ulp", scope="all", ulp_threshold=1)
    return result


ttnn.attach_golden_function(ttnn.tanh, golden_function=_golden_function_tanh)
ttnn.attach_golden_function(
    ttnn.sinh,
    golden_function=_golden_function_sinh,
    preprocess_golden_function_inputs=_preprocess_hyperbolic_golden_inputs,
)
ttnn.attach_golden_function(
    ttnn.cosh,
    golden_function=_golden_function_cosh,
    preprocess_golden_function_inputs=_preprocess_hyperbolic_golden_inputs,
)


def _preprocess_gelu_golden_inputs(function_args, function_kwargs):
    """Prepare GELU golden inputs and identify device-specific variants.
    Marks FastLut and approximate modes to skip unsupported generic comparison.
    """

    golden_args, golden_kwargs = ttnn.decorators.default_preprocess_golden_function_inputs(
        function_args, function_kwargs
    )
    variant = function_kwargs.get("variant")
    fast_and_approximate_mode = function_kwargs.get("fast_and_approximate_mode", False)
    if variant == ttnn.GeluVariant.FastLut or fast_and_approximate_mode:
        golden_kwargs["_ttnn_skip_comparison"] = True
    return golden_args, golden_kwargs


def _golden_function_gelu(
    input_tensor, *args, variant=None, fast_and_approximate_mode=False, _ttnn_skip_comparison=False, **kwargs
):
    import torch

    # GELU variants have different approximation and non-finite contracts, including a
    # device-specific FastLut. Skip generic PCC while preserving this golden for direct callers.
    if _ttnn_skip_comparison:
        return None

    # Tanh changes the function; Accurate and default use Torch's exact reference.
    approximate = "tanh" if variant == ttnn.GeluVariant.Tanh else "none"
    input_dtype = input_tensor.dtype
    if input_dtype == torch.bfloat16:
        # Evaluate BF16 in float32 and round once to match the hardware accurate path.
        input_tensor = input_tensor.to(torch.float32)
    result = torch.nn.functional.gelu(input_tensor, approximate=approximate)
    return result.to(input_dtype)


ttnn.attach_golden_function(
    ttnn.gelu,
    golden_function=_golden_function_gelu,
    preprocess_golden_function_inputs=_preprocess_gelu_golden_inputs,
)


def _golden_function_softplus(input_tensor, *args, beta=1.0, threshold=20.0, **kwargs):
    import torch

    # The generic unary wrapper drops parameters, so forward both operation controls.
    return torch.nn.functional.softplus(input_tensor, beta=beta, threshold=threshold)


ttnn.attach_golden_function(ttnn.softplus, golden_function=_golden_function_softplus)


def _preprocess_inverse_trig_golden_inputs(function_args, function_kwargs):
    """Preserve block-float input identity for inverse-trigonometric goldens.
    Adds a BF8 flag used to select out-of-domain comparison behavior.
    """

    input_tensor = function_args[0] if function_args else function_kwargs["input_tensor"]
    golden_args, golden_kwargs = ttnn.decorators.default_preprocess_golden_function_inputs(
        function_args, function_kwargs
    )
    # Default preprocessing converts BFLOAT8_B to a Torch tensor and loses its block-float identity.
    # Preserve that metadata so out-of-domain calls can use their mask-only comparison policy.
    golden_kwargs["_ttnn_input_is_bfloat8_b"] = input_tensor.dtype == ttnn.bfloat8_b
    return golden_args, golden_kwargs


def _golden_function_asin(input_tensor, *args, _ttnn_input_is_bfloat8_b=False, **kwargs):
    import torch

    # Wide out-of-domain BFLOAT8_B blocks are validated by their non-finite mask, not value PCC.
    # Returning no local golden lets that operation-specific check observe the device output.
    if _ttnn_input_is_bfloat8_b and bool(torch.any(torch.abs(input_tensor) > 1)):
        return None
    result = torch.asin(input_tensor)
    if input_tensor.dtype == torch.bfloat16 and bool(torch.any(torch.abs(input_tensor) > 1)):
        # BF16 packing represents the SFPU's out-of-domain NaN as positive infinity.
        # Mirror that representation for direct ULP callers and compare finite lanes to two ULP.
        result = result.masked_fill(torch.abs(input_tensor) > 1, float("inf"))
        ttnn.decorators.set_golden_comparison_config(
            result, method="ulp", scope="all", ulp_threshold=2, nonfinite="mask"
        )
    return result


ttnn.attach_golden_function(
    ttnn.asin,
    golden_function=_golden_function_asin,
    preprocess_golden_function_inputs=_preprocess_inverse_trig_golden_inputs,
)


def _golden_function_acos(input_tensor, *args, _ttnn_input_is_bfloat8_b=False, **kwargs):
    import torch

    # Wide out-of-domain BFLOAT8_B blocks are validated by their non-finite mask, not value PCC.
    # Returning no local golden lets that operation-specific check observe the device output.
    if _ttnn_input_is_bfloat8_b and bool(torch.any(torch.abs(input_tensor) > 1)):
        return None
    result = torch.acos(input_tensor)
    if input_tensor.dtype == torch.bfloat16 and bool(torch.any(torch.abs(input_tensor) > 1)):
        # BF16 packing represents the SFPU's out-of-domain NaN as positive infinity.
        # Mirror that representation for direct ULP callers and compare finite lanes to two ULP.
        result = result.masked_fill(torch.abs(input_tensor) > 1, float("inf"))
        ttnn.decorators.set_golden_comparison_config(
            result, method="ulp", scope="all", ulp_threshold=2, nonfinite="mask"
        )
    return result


ttnn.attach_golden_function(
    ttnn.acos,
    golden_function=_golden_function_acos,
    preprocess_golden_function_inputs=_preprocess_inverse_trig_golden_inputs,
)


def _golden_function_acosh(input_tensor_a, *args, **kwargs):
    import torch

    result = torch.acosh(input_tensor_a)
    return result.masked_fill_(input_tensor_a < 1, float("inf")) if input_tensor_a.dtype == torch.bfloat16 else result


ttnn.attach_golden_function(ttnn.acosh, golden_function=_golden_function_acosh)


def _golden_function_atanh(input_tensor_a, *args, **kwargs):
    import torch

    result = torch.atanh(input_tensor_a)
    return (
        result.masked_fill_((input_tensor_a <= -1) | (input_tensor_a >= 1), float("inf"))
        if input_tensor_a.dtype == torch.bfloat16
        else result
    )


ttnn.attach_golden_function(ttnn.atanh, golden_function=_golden_function_atanh)


def _golden_function_reciprocal(input_tensor, *args, device=None, **kwargs):
    import torch

    # Reciprocal zero-input behavior is signed infinity, not the finite maximum from nan_to_num.
    # Preserve Torch's NaN and infinity values in both direct and comparison-mode golden calls.
    return torch.reciprocal(input_tensor)


ttnn.attach_golden_function(ttnn.reciprocal, golden_function=_golden_function_reciprocal)


def _golden_function_i1(input_tensor, *args, **kwargs):
    import torch

    # The SFPU implementation saturates its input at +/-88.5 before evaluating the approximation.
    # Clamp the host reference at the same boundary instead of allowing unbounded i1 growth.
    return torch.special.i1(torch.clamp(input_tensor, min=-88.5, max=88.5))


ttnn.attach_golden_function(ttnn.i1, golden_function=_golden_function_i1)


def _golden_function_pow(input_tensor, exponent, *args, **kwargs):
    import torch

    if torch.is_tensor(input_tensor) and integer_golden.is_unsigned_dtype(input_tensor.dtype):
        # Evaluate unsupported unsigned power in int64 and restore TT wraparound.
        return integer_golden.power(input_tensor, exponent)
    return torch.pow(input_tensor, exponent)


ttnn.attach_golden_function(ttnn.pow, golden_function=_golden_function_pow)


def _golden_function_xielu(x, *args, alpha_p=0.8, alpha_n=0.8, **kwargs):
    # xIELU (Expanded Integral of the Exponential Linear Unit)
    # A Custom piecewise trainable activation function from "Deriving Activation Functions Using Integration" paper (https://arxiv.org/abs/2411.13010)

    # With beta = 0.5 and eps = -1e-6:
    #      x > 0 :  alpha_p * x^2 + beta * x
    #      x <= 0:  alpha_n * (expm1(minimum(x, eps))) - (alpha_n * x) + 0.5 * x
    import torch

    dtype = x.dtype
    if dtype == torch.bfloat16:
        # Compute the golden reference in float32 and convert to bf16 only once after evaluation for a more reliable comparison.
        x = x.to(torch.float32)
    beta = 0.5
    eps = -1e-6
    pos_part = alpha_p * x * x + beta * x
    x_clipped = torch.minimum(x, torch.full_like(x, eps))
    neg_part = alpha_n * torch.expm1(x_clipped) - alpha_n * x + beta * x
    out = torch.where(x > 0, pos_part, neg_part)
    return out.to(dtype)


ttnn.attach_golden_function(ttnn.xielu, golden_function=_golden_function_xielu)


def _golden_function_elu(input_tensor_a, *args, alpha=1.0, **kwargs):
    import torch

    return torch.nn.functional.elu(input_tensor_a, alpha=alpha)


ttnn.attach_golden_function(ttnn.elu, golden_function=_golden_function_elu)


def _golden_function_hardtanh(input_tensor_a, *args, min_val=-1.0, max_val=1.0, **kwargs):
    import torch

    return torch.nn.functional.hardtanh(input_tensor_a, min_val, max_val)


ttnn.attach_golden_function(ttnn.hardtanh, golden_function=_golden_function_hardtanh)


def _golden_function_leaky_relu(input_tensor, *args, negative_slope=0.01, **kwargs):
    import torch

    if input_tensor.dtype == torch.uint8 or integer_golden.is_unsigned_dtype(input_tensor.dtype):
        # Torch has no unsigned leaky ReLU kernel, while unsigned inputs are always non-negative.
        # Return the identity for UInt8/UInt16/UInt32 to match the device implementation.
        return input_tensor
    return torch.nn.functional.leaky_relu(input_tensor, negative_slope=negative_slope)


ttnn.attach_golden_function(ttnn.leaky_relu, golden_function=_golden_function_leaky_relu)


def _golden_function_relu_min(input_tensor, lower_limit, *args, **kwargs):
    import torch

    # Torch does not provide maximum kernels for UInt16/UInt32 tensors.
    # Widening for the host reference preserves unsigned ordering and restores the input dtype.
    if integer_golden.is_unsigned_dtype(input_tensor.dtype):
        return integer_golden.binary(input_tensor, lower_limit, torch.maximum)
    return torch.max(input_tensor, torch.tensor(lower_limit))


ttnn.attach_golden_function(ttnn.relu_min, golden_function=_golden_function_relu_min)


def _golden_function_relu_max(input_tensor, upper_limit, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor.dtype):
        # Unsigned ReLU max needs the widened clamp path because Torch lacks its kernel.
        return integer_golden.clamp(input_tensor, 0, upper_limit)
    upper_limit = torch.tensor(upper_limit, dtype=input_tensor.dtype, device=input_tensor.device)
    return torch.relu(torch.minimum(input_tensor, upper_limit))


ttnn.attach_golden_function(ttnn.relu_max, golden_function=_golden_function_relu_max)


def _golden_function_heaviside(input_tensor_a, value, *args, **kwargs):
    import torch

    return torch.heaviside(input_tensor_a, torch.tensor(value, dtype=input_tensor_a.dtype))


ttnn.attach_golden_function(ttnn.heaviside, golden_function=_golden_function_heaviside)


def _golden_function_fill(input_tensor_a, fill_value, *args, **kwargs):
    import torch

    return torch.full_like(input_tensor_a, fill_value)


ttnn.attach_golden_function(ttnn.fill, golden_function=_golden_function_fill)


def _golden_function_polygamma(input_tensor_a, k, *args, **kwargs):
    import torch

    return torch.special.polygamma(n=k, input=input_tensor_a)


ttnn.attach_golden_function(ttnn.polygamma, golden_function=_golden_function_polygamma)


def _golden_function_clamp(input_tensor, min=None, max=None, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor.dtype):
        # PyTorch lacks unsigned clamp kernels; clamp widened values and restore dtype.
        return integer_golden.clamp(input_tensor, min, max)
    return torch.clamp(input_tensor, min, max)


ttnn.attach_golden_function(ttnn.clamp, golden_function=_golden_function_clamp)


def _golden_function_clip(input_tensor, min=None, max=None, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor.dtype):
        # PyTorch lacks unsigned clamp kernels; clamp widened values and restore dtype.
        return integer_golden.clamp(input_tensor, min, max)
    return torch.clip(input_tensor, min, max)


ttnn.attach_golden_function(ttnn.clip, golden_function=_golden_function_clip)


def _golden_function_round(input_tensor_a, decimals=None, *args, **kwargs):
    import torch

    if decimals is None:
        return torch.round(input=input_tensor_a)
    else:
        return torch.round(input=input_tensor_a, decimals=decimals)


ttnn.attach_golden_function(ttnn.round, golden_function=_golden_function_round)


def _golden_function_selu(input_tensor_a, *args, **kwargs):
    import torch

    return torch.nn.functional.selu(input_tensor_a)


ttnn.attach_golden_function(ttnn.selu, golden_function=_golden_function_selu)


def _golden_function_tanhshrink(input_tensor_a, *args, **kwargs):
    import torch

    return torch.nn.functional.tanhshrink(input=input_tensor_a)


ttnn.attach_golden_function(ttnn.tanhshrink, golden_function=_golden_function_tanhshrink)


def _golden_function_threshold(input_tensor, threshold, value, *args, **kwargs):
    import torch

    # Device scalar arguments are represented in the input dtype before the piecewise replacement.
    # Quantize the replacement value likewise and allow the documented one-ULP BF16 result.
    value = torch.tensor(value, dtype=input_tensor.dtype, device=input_tensor.device).item()
    result = torch.threshold(input_tensor, threshold, value)
    if input_tensor.dtype == torch.bfloat16:
        ttnn.decorators.set_golden_comparison_config(result, method="ulp", scope="degenerate", ulp_threshold=1)
    return result


ttnn.attach_golden_function(ttnn.threshold, golden_function=_golden_function_threshold)


def _golden_function_trunc(input_tensor_a, *args, **kwargs):
    import torch

    return torch.trunc(input=input_tensor_a)


ttnn.attach_golden_function(ttnn.trunc, golden_function=_golden_function_trunc)


def _golden_function_rsub(input_tensor_a, value, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor_a.dtype):
        # Evaluate unsupported unsigned reverse subtraction with TT wraparound.
        return integer_golden.binary(input_tensor_a, value, lambda a, b: b - a)
    return torch.sub(value, input_tensor_a)


# Binary registration owns rsub's scalar/BF8 comparison contract and activation semantics.
# Do not replace it here with the legacy unary helper after operation modules are loaded.


def _golden_function_rdiv(input_tensor_a, value, *args, **kwargs):
    import torch

    return torch.div(torch.tensor(value, dtype=input_tensor_a.dtype), input_tensor_a)


ttnn.attach_golden_function(ttnn.rdiv, golden_function=_golden_function_rdiv)


def _golden_function_bitwise_left_shift(input_tensor, shift_amt, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor.dtype):
        # Evaluate shifts in int64, zero invalid counts, and mask to the dtype width.
        return integer_golden.shift(input_tensor, shift_amt, torch.bitwise_left_shift)
    return torch.bitwise_left_shift(input_tensor, shift_amt)


ttnn.attach_golden_function(ttnn.bitwise_left_shift, golden_function=_golden_function_bitwise_left_shift)

ttnn.attach_golden_function(ttnn.logical_left_shift, golden_function=_golden_function_bitwise_left_shift)


def _golden_function_bitwise_right_shift(input_tensor, shift_amt, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor.dtype):
        # Unary right shift has SFPU-specific count and sign behavior, unlike the
        # shared zero-on-invalid helper used by left and binary shift operations.
        return integer_golden.right_shift(input_tensor, shift_amt)
    return torch.bitwise_right_shift(input_tensor, shift_amt)


ttnn.attach_golden_function(ttnn.bitwise_right_shift, golden_function=_golden_function_bitwise_right_shift)


def _golden_function_bitwise_and(input_tensor_a, value, *args, **kwargs):
    import torch

    return torch.bitwise_and(input_tensor_a, value)


ttnn.attach_golden_function(ttnn.bitwise_and, golden_function=_golden_function_bitwise_and)


def _golden_function_bitwise_or(input_tensor_a, value, *args, **kwargs):
    import torch

    return torch.bitwise_or(input_tensor_a, value)


ttnn.attach_golden_function(ttnn.bitwise_or, golden_function=_golden_function_bitwise_or)


def _golden_function_bitwise_xor(input_tensor_a, value, *args, **kwargs):
    import torch

    return torch.bitwise_xor(input_tensor_a, value)


ttnn.attach_golden_function(ttnn.bitwise_xor, golden_function=_golden_function_bitwise_xor)


def _golden_function_bitwise_not(input_tensor_a, *args, **kwargs):
    import torch

    return torch.bitwise_not(input_tensor_a)


ttnn.attach_golden_function(ttnn.bitwise_not, golden_function=_golden_function_bitwise_not)


def _golden_function_glu(input_tensor, dim=-1, *args, **kwargs):
    import torch

    # Comparison-mode calls may omit optional dimensions just like the public operation.
    # Keep the host reference default aligned with the device implementation.
    return torch.nn.functional.glu(input_tensor, dim)


ttnn.attach_golden_function(ttnn.glu, golden_function=_golden_function_glu)


def _golden_function_reglu(input_tensor, dim=-1, *args, **kwargs):
    import torch

    assert isinstance(dim, int), "dim must be an integer"
    assert dim in [-1, 3], "dim must be -1 or 3"
    # Torch does not implement ReLU for uint16/uint32, so widen only the host reference.
    # The default dimension also matches the public fused activation operation.
    golden_input = input_tensor.to(torch.int64) if input_tensor.dtype in (torch.uint16, torch.uint32) else input_tensor
    split_size = golden_input.size(-1) // 2
    split_tensors = torch.split(golden_input, split_size_or_sections=[split_size, split_size], dim=dim)
    tensA, tensB = split_tensors[0], split_tensors[1]
    result = tensA * torch.nn.functional.relu(tensB)
    if integer_golden.is_unsigned_dtype(input_tensor.dtype):
        return integer_golden.restore_unsigned(result, input_tensor.dtype)
    return result


ttnn.attach_golden_function(ttnn.reglu, golden_function=_golden_function_reglu)


def _golden_function_geglu(input_tensor, dim=-1, *args, **kwargs):
    import torch

    # Match the C++ default when comparison mode receives no dim argument.
    assert isinstance(dim, int), "dim must be an integer"
    assert dim in [-1, 3], "dim must be -1 or 3"
    split_size = input_tensor.size(-1) // 2
    split_tensors = torch.split(input_tensor, split_size_or_sections=[split_size, split_size], dim=dim)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * torch.nn.functional.gelu(tensB)


ttnn.attach_golden_function(ttnn.geglu, golden_function=_golden_function_geglu)


def _golden_function_swiglu(input_tensor, dim=-1, *args, **kwargs):
    import torch

    assert isinstance(dim, int), "dim must be an integer"
    assert dim in [-1, 3], "dim must be -1 or 3"
    split_size = input_tensor.size(-1) // 2
    split_tensors = torch.split(input_tensor, split_size_or_sections=[split_size, split_size], dim=dim)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * torch.nn.functional.silu(tensB)


ttnn.attach_golden_function(ttnn.swiglu, golden_function=_golden_function_swiglu)


def _golden_function_logical_not_(input_tensor, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor.dtype):
        # Preserve in-place golden state while evaluating unsigned logical-not.
        input_tensor.copy_(integer_golden.logical_not(input_tensor))
        return input_tensor
    return input_tensor.logical_not_()


ttnn.attach_golden_function(ttnn.logical_not_, golden_function=_golden_function_logical_not_)


def _golden_function_hardshrink(input_tensor_a, *args, lambd=0.5, **kwargs):
    import torch

    return torch.nn.functional.hardshrink(input_tensor_a, lambd=lambd)


ttnn.attach_golden_function(ttnn.hardshrink, golden_function=_golden_function_hardshrink)


def _golden_function_softshrink(input_tensor_a, *args, lambd=0.5, **kwargs):
    import torch

    return torch.nn.functional.softshrink(input_tensor_a, lambd=lambd)


ttnn.attach_golden_function(ttnn.softshrink, golden_function=_golden_function_softshrink)


def _golden_function_logit(input_tensor_a, *args, eps=None, **kwargs):
    import torch

    if eps is not None and eps > 0.5:
        # Manual implementation to avoid platform-dependent UB in torch.special.logit
        # when eps > 0.5 (std::clamp with lo > hi is undefined behavior).
        lo = 1.0 - eps
        hi = eps
        x = torch.clamp(input_tensor_a, lo, hi)
        return torch.log(x / (1.0 - x))
    return torch.special.logit(input_tensor_a, eps=eps)


ttnn.attach_golden_function(ttnn.logit, golden_function=_golden_function_logit)


def _golden_function_celu(input_tensor_a, *args, alpha=1.0, **kwargs):
    import torch

    return torch.celu(input_tensor_a, alpha=alpha)


ttnn.attach_golden_function(ttnn.celu, golden_function=_golden_function_celu)


def _golden_function_softcap(input_tensor_a, beta, *args, **kwargs):
    import torch

    return beta * torch.tanh(input_tensor_a.to(torch.float32) / beta)


ttnn.attach_golden_function(ttnn.softcap, golden_function=_golden_function_softcap)


def torch_reglu(input_tensor, *args, **kwargs):
    # Keep the legacy registration helper aligned with the active golden, including
    # UInt16/UInt32 widening and device-width wraparound.
    return _golden_function_reglu(input_tensor, *args, **kwargs)


def torch_swiglu(input_tensor, *args, **kwargs):
    import torch

    split_size = input_tensor.size(-1) // 2
    split_tensors = torch.split(input_tensor, split_size_or_sections=[split_size, split_size], dim=-1)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * torch.nn.functional.silu(tensB)


def torch_geglu(input_tensor, *args, **kwargs):
    import torch

    split_size = input_tensor.size(-1) // 2
    split_tensors = torch.split(input_tensor, split_size_or_sections=[split_size, split_size], dim=-1)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * torch.nn.functional.gelu(tensB)


def register_ttl_activation_function_glu(name, ttl_activation_function, param):
    def _golden_function(input_tensor: ttnn.Tensor, dim: int = -1, **_):
        import torch

        name_to_torch_function = {
            "glu": torch.nn.functional.glu,
            "reglu": torch_reglu,
            "swiglu": torch_swiglu,
            "geglu": torch_geglu,
        }
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)

        return torch_function(input_tensor, dim=dim)

    doc = f"""{(name)}(input_tensor: ttnn.Tensor, dim: int = -1, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

            Applies the {name} function to the elements of the input tensor :attr:`input_tensor` split along :attr:`{param}`.

            .. math::
                {(name)}(\\mathrm{{input\\_tensor}}_i  \\; , \\; {param})

            Args:
                * :attr:`input_tensor`
                * :attr:`{param}`

            Example::

                >>> tensor = ttnn.from_torch(torch.tensor((32, 64), dtype=torch.bfloat16), device=device)
                >>> output = ttnn.{(name)}(tensor, {param})

            """

    @ttnn.register_python_operation(name=f"ttnn.{name}", golden_function=_golden_function, doc=doc)
    def activation_function(
        input_tensor: ttnn.Tensor, dim: int = -1, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        input_shape = tuple(input_tensor.shape)
        last_dim = input_shape[-1]
        glu_shape = input_shape[:-1] + (int(last_dim / 2),)

        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not _is_scalar(dim):
            raise TypeError("Expected second argument to be a float")

        if not ttnn.is_tensor_storage_on_device(input_tensor):
            raise RuntimeError("input_tensor must be on device!")

        output_tensor = ttl_activation_function(input_tensor, dim, output_mem_config=memory_config)

        output_tensor = ttnn.reshape(output_tensor, ttnn.Shape(glu_shape))
        return output_tensor


def _golden_function_normalize_global(input_tensor_a, *args, **kwargs):
    import torch

    mx = torch.mean(input_tensor_a, [0, 1, 2, 3], keepdim=True)
    sx = torch.std(input_tensor_a, [0, 1, 2, 3], keepdim=True)
    input_tensor_a = (input_tensor_a - mx) / sx

    return input_tensor_a


ttnn.attach_golden_function(ttnn.normalize_global, golden_function=_golden_function_normalize_global)


def _golden_function_normalize_hw(input_tensor_a, *args, **kwargs):
    import torch

    mean_hw = torch.mean(input_tensor_a, [-2, -1], keepdim=True)
    std_hw = torch.std(input_tensor_a, [-2, -1], keepdim=True)

    for i in range(input_tensor_a.shape[0]):
        for j in range(input_tensor_a.shape[1]):
            input_tensor_a[i, j, :, :] = (input_tensor_a[i, j, :, :] - mean_hw[i, j, :, :]) / std_hw[i, j, :, :]

    return input_tensor_a


ttnn.attach_golden_function(ttnn.normalize_hw, golden_function=_golden_function_normalize_hw)


def _golden_function_rpow(input_tensor, dim, *args, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor.dtype):
        # Evaluate unsupported unsigned reverse power in int64 and restore its width.
        return integer_golden.power(input_tensor, dim, reverse=True)
    return torch.pow(dim, input_tensor)


ttnn.attach_golden_function(ttnn.rpow, golden_function=_golden_function_rpow)


def _golden_function_frac(input_tensor_a, *args, **kwargs):
    import torch

    return torch.frac(input_tensor_a)


ttnn.attach_golden_function(ttnn.frac, golden_function=_golden_function_frac)


def _golden_function_rdiv(input_tensor_a, value, *args, rounding_mode=None, **kwargs):
    import torch

    return torch.div(torch.full_like(input_tensor_a, value), input_tensor_a, rounding_mode=rounding_mode)


ttnn.attach_golden_function(ttnn.rdiv, golden_function=_golden_function_rdiv)


def _golden_function_alt_complex_rotate90(input_tensor_a, *args, **kwargs):
    import torch

    x = input_tensor_a.reshape(*input_tensor_a.shape[:-1], -1, 2)
    x_real, x_imag = x.chunk(2, dim=-1)
    return torch.cat([-x_imag, x_real], dim=-1).flatten(-2)


ttnn.attach_golden_function(ttnn.alt_complex_rotate90, golden_function=_golden_function_alt_complex_rotate90)

SigmoidMode = ttnn._ttnn.operations.unary.SigmoidMode
GeluVariant = ttnn._ttnn.operations.unary.GeluVariant

__all__ = []
