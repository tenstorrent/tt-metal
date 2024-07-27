// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/composite/composite_ops.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/auto_format.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/concat/concat_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/copy/copy_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/optimizer/optimizer_ops.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/reduce/reduce_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/reshape/reshape_op.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "tt_numpy/functions.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/ternary/where_op.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/copy.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"

namespace tt {

namespace tt_metal {

Tensor mk_zero_tensor_like(
    uint8_t queue_id,
    const Tensor& reference_tensor,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor = std::nullopt) {
    const DataType& dtype =
        output_tensor.has_value() ? output_tensor.value().get_dtype() : reference_tensor.get_dtype();
    Tensor zero = ttnn::operations::creation::create_scalar(0.0f, dtype, Layout::TILE, reference_tensor.device());
    return ttnn::multiply(queue_id, reference_tensor, zero, std::nullopt, output_mem_config, output_tensor);
}

Tensor mk_zero_tensor_like(
    const Tensor& reference_tensor,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor = std::nullopt) {
    uint8_t default_queue_id = 0;
    return mk_zero_tensor_like(default_queue_id, reference_tensor, output_mem_config, output_tensor);
}

// TODO: enable zeroes(), ones() and eye() type functions on-device using this type of logic
template <typename T>
Tensor mk_filled_tensor_like(
    const Tensor& reference_tensor,
    T val,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor = std::nullopt,
    uint8_t queue_id = 0) {
    const DataType& dtype =
        output_tensor.has_value() ? output_tensor.value().get_dtype() : reference_tensor.get_dtype();
    Tensor k = ttnn::operations::creation::create_scalar(val, dtype, Layout::TILE, reference_tensor.device());
    Tensor zero_like = mk_zero_tensor_like(reference_tensor, output_mem_config);
    if (output_tensor.has_value()) {
        return ttnn::add(queue_id, zero_like, k, std::nullopt, output_mem_config, output_tensor);
    } else {
        return ttnn::add(queue_id, zero_like, k, std::nullopt, output_mem_config);
    }
}

// Function: softshrink
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html
Tensor _softshrink(const Tensor& a, float param, const MemoryConfig& output_mem_config) {
    TT_ASSERT(param >= 0);
    Tensor t_a_plus_param = ttnn::add(a, param, std::nullopt, output_mem_config);
    Tensor t1 = ttnn::multiply(ttnn::ltz(t_a_plus_param, output_mem_config), t_a_plus_param, std::nullopt, output_mem_config);
    t_a_plus_param.deallocate();
    Tensor t_a_minus_param = ttnn::subtract(a, param, std::nullopt, output_mem_config);
    Tensor t2 =
        ttnn::multiply(ttnn::gtz(t_a_minus_param, output_mem_config), t_a_minus_param, std::nullopt, output_mem_config);
    t_a_minus_param.deallocate();
    return ttnn::add(t1, t2, std::nullopt, output_mem_config);
}
Tensor softshrink(const Tensor& a, float param, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _softshrink)(a, param, output_mem_config);
}

// Function: hardshrink
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html
Tensor _hardshrink(const Tensor& a, float param, const MemoryConfig& output_mem_config) {
    TT_ASSERT(param >= 0);
    Tensor t1 = ttnn::multiply(ttnn::ltz(ttnn::add(a, param)), a, std::nullopt, output_mem_config);
    Tensor t2 = ttnn::multiply(ttnn::gtz(ttnn::subtract(a, param)), a, std::nullopt, output_mem_config);
    return ttnn::add(t1, t2, std::nullopt, output_mem_config);
}
Tensor hardshrink(const Tensor& a, float param, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _hardshrink)(a, param, output_mem_config);
}


// ELU :
//  Theano defines it as,
//  return tensor.switch(x > 0, x, alpha * tensor.expm1(x))

// rpow: y = k**(a) = exp( a**log(k) )
Tensor rpow(const Tensor& a, float k, const MemoryConfig& output_mem_config) {
    TT_ASSERT(k > 0.0, "rpow cannot be calcualted for non-positive numbers");
    float log_k = logf(k);

    Tensor scalar = ttnn::operations::creation::create_scalar(log_k, a.get_dtype(), Layout::TILE, a.device());
    Tensor result = ttnn::multiply(a, scalar, std::nullopt, output_mem_config);
    scalar.deallocate();
    return ttnn::exp(result, false, output_mem_config);
}

// compute polyval by Horner's rule
Tensor _polyval(const Tensor& input_tensor, std::vector<float> coeffs, const MemoryConfig& output_mem_config) {
    TT_ASSERT(coeffs.size() != 0 && "coeffs should be 1 or more coefficients");
    if (coeffs.size() == 1) {
        return mk_filled_tensor_like(input_tensor, coeffs[0], output_mem_config);
    }

    Tensor scalar = ttnn::operations::creation::create_scalar(
        coeffs[0], input_tensor.get_dtype(), Layout::TILE, input_tensor.device());
    Tensor result = ttnn::multiply(input_tensor, scalar, std::nullopt, output_mem_config);
    scalar.deallocate();
    for (int idx = 1; idx < coeffs.size() - 1; idx++) {
        Tensor scalar = ttnn::operations::creation::create_scalar(
            coeffs[idx], input_tensor.get_dtype(), Layout::TILE, input_tensor.device());
        result = ttnn::add(result, scalar, std::nullopt, output_mem_config);
        scalar.deallocate();
        result = ttnn::multiply(input_tensor, result, std::nullopt, output_mem_config);
    }
    Tensor last_coeffs = ttnn::operations::creation::create_scalar(
        coeffs.back(), input_tensor.get_dtype(), Layout::TILE, input_tensor.device());
    Tensor final_tensor = ttnn::add(result, last_coeffs, std::nullopt, output_mem_config);
    last_coeffs.deallocate();
    return final_tensor;
}
Tensor polyval(const Tensor& input_tensor, std::vector<float> coeffs, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _polyval)(input_tensor, coeffs, output_mem_config);
}

// Function: MAC
// compute multiply-accumulate: y = a * b + c,  over various 8 combinations of a, b, c
// being a scalar or tensor
Tensor _mac(const Tensor& a, const Tensor& b, const Tensor& c, const MemoryConfig& output_mem_config) {
    bool a_is_scalar = a.intended_volume() == 1;
    bool b_is_scalar = b.intended_volume() == 1;
    bool c_is_scalar = c.intended_volume() == 1;

    if (!a_is_scalar && !b_is_scalar && !c_is_scalar) {
        // all tensors
        return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (!a_is_scalar && !b_is_scalar && c_is_scalar) {
        // a - tensor, b - tensor, c - is scalar
        return ttnn::add(
            ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (!a_is_scalar && b_is_scalar && !c_is_scalar) {
        // a - tensor, b - scalar, c - is tensor
        return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (!a_is_scalar && b_is_scalar && c_is_scalar) {
        // a - tensor, b - scalar, c - is scalar
        return ttnn::add(
            ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (a_is_scalar && !b_is_scalar && !c_is_scalar) {
        // a - scalar, b - tensor, c - tensor
        return ttnn::add(ttnn::multiply(b, a, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (a_is_scalar && !b_is_scalar && c_is_scalar) {
        // a - scalar, b - tensor, c - is scalar
        return ttnn::add(
            ttnn::multiply(b, a, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (a_is_scalar && b_is_scalar && !c_is_scalar) {
        // a - scalar, b - scalar, c - is tensor
        return ttnn::add(
            c, ttnn::multiply(a, b, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    }

    // all scalars
    // a - scalar, b - scalar, c - is scalar
    TT_ASSERT(a_is_scalar && b_is_scalar && c_is_scalar);
    return ttnn::add(ttnn::multiply(a, b), c);
}
Tensor mac(const Tensor& a, const Tensor& b, const Tensor& c, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _mac)(a, b, c, output_mem_config);
}

Tensor _mac_overload(const Tensor& a, float b, float c, const MemoryConfig& output_mem_config) {
    Tensor t_b = ttnn::operations::creation::create_scalar(b, a.get_dtype(), Layout::TILE, a.device());
    Tensor t_c = ttnn::operations::creation::create_scalar(c, a.get_dtype(), Layout::TILE, a.device());
    Tensor return_tensor = mac(a, t_b, t_c, output_mem_config);
    t_b.deallocate();
    t_c.deallocate();
    return return_tensor;
}
Tensor mac(const Tensor& input_a, float b, float c, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _mac_overload)(input_a, b, c, output_mem_config);
}

Tensor _logical_andi(const Tensor& input_a, float immediate, const MemoryConfig& output_mem_config) {
    if (std::fpclassify(immediate) == FP_ZERO) {
        return full_like(input_a, immediate, output_mem_config);
    } else {
        return ttnn::nez(input_a);
    }
}
Tensor logical_andi(const Tensor& input_a, float immediate, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _logical_andi)(input_a, immediate, output_mem_config);
}


// lerp(input, end, weight) = start + weight * (end - start)
Tensor _lerp(const Tensor& input_a, const Tensor& input_b, float value, const MemoryConfig& output_mem_config) {
    Tensor t_value =
        ttnn::operations::creation::create_scalar(value, input_a.get_dtype(), Layout::TILE, input_a.device());
    Tensor t_diff = ttnn::subtract(input_b, input_a, std::nullopt, output_mem_config);
    Tensor t_mul = ttnn::multiply(t_diff, t_value, std::nullopt, output_mem_config);
    Tensor result = ttnn::add(input_a, t_mul, std::nullopt, output_mem_config);
    return result;
}
Tensor lerp(const Tensor& input_a, const Tensor& input_b, float value, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _lerp)(input_a, input_b, value, output_mem_config);
}

// lerp(input, end, weight) = start + weight * (end - start)
Tensor _lerp_overload(
    const Tensor& input_a, const Tensor& input_b, const Tensor& input_c, const MemoryConfig& output_mem_config) {
    Tensor t_diff = ttnn::multiply(
        ttnn::subtract(input_b, input_a, std::nullopt, output_mem_config), input_c, std::nullopt, output_mem_config);
    Tensor result = ttnn::add(input_a, t_diff, std::nullopt, output_mem_config);
    return result;
}
Tensor lerp(
    const Tensor& input_a, const Tensor& input_b, const Tensor& input_c, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _lerp_overload)(input_a, input_b, input_c, output_mem_config);
}

// ldexp(input,other)=input * (2^other)
Tensor _ldexp(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    Tensor result = ttnn::multiply(input_a, ttnn::exp2(input_b, output_mem_config), std::nullopt, output_mem_config);
    return result;
}

Tensor _logical_ori(const Tensor& input_a, float immediate, const MemoryConfig& output_mem_config) {
    if (std::fpclassify(immediate) == FP_ZERO) {
        return ttnn::nez(input_a, output_mem_config);
    } else {
        return full_like(input_a, 1, output_mem_config);
    }
}
Tensor logical_ori(const Tensor& input_a, float immediate, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _logical_ori)(input_a, immediate, output_mem_config);
}

Tensor _logical_noti(const Tensor& input_a, float immediate, const MemoryConfig& output_mem_config) {
    Tensor t_imm = full_like(input_a, immediate, output_mem_config);
    Tensor result = ttnn::logical_not(t_imm, output_mem_config);
    return result;
}
Tensor logical_noti(const Tensor& input_a, float immediate, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _logical_noti)(input_a, immediate, output_mem_config);
}

Tensor _div(const Tensor& input_a, const Tensor& input_b, bool accurate_mode, string round_mode,  const MemoryConfig& output_mem_config) {
    TT_FATAL((round_mode == "None" || round_mode == "trunc" || round_mode == "floor") && "Incorrect rounding mode (expected 'None', 'trunc', or 'floor')");
    Tensor result = ttnn::divide(input_a, input_b);
    if(round_mode == "trunc"){
        result = ttnn::trunc(result);
    }
    else if(round_mode == "floor"){
        result = ttnn::floor(result);
    }

    if (accurate_mode == false) {  // If input_b is non-zero tensor
        return result;
    }

    Tensor t_inf = full_like(input_a, std::numeric_limits<float>::infinity(), output_mem_config);
    Tensor t_nan = full_like(input_a, std::nanf(""), output_mem_config);
    return ttnn::where(
        ttnn::eqz(input_b, output_mem_config),
        ttnn::where(
            ttnn::eqz(input_a, output_mem_config),
            t_nan,
            ttnn::multiply(t_inf, ttnn::sign(input_a, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config),
        result,
        output_mem_config);
}
Tensor div(const Tensor& input_a, const Tensor& input_b, bool accurate_mode, string round_mode, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _div)(input_a, input_b, accurate_mode, round_mode, output_mem_config);
}

Tensor _div_overload(const Tensor& input_a, float scalar, bool accurate_mode, string round_mode,  const MemoryConfig& output_mem_config) {
    TT_FATAL((round_mode == "None" || round_mode == "trunc" || round_mode == "floor") && "Incorrect rounding mode (expected 'None', 'trunc', or 'floor')");
    Tensor result = ttnn::multiply(input_a, (1.0f/scalar));

    if(round_mode == "trunc"){
        result = ttnn::trunc(result);
    }
    else if(round_mode == "floor"){
        result = ttnn::floor(result);
    }

    return result;
}
Tensor div(const Tensor& input_a, float scalar, bool accurate_mode, string round_mode, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _div_overload)(input_a, scalar, accurate_mode, round_mode, output_mem_config);
}


Tensor _frac(const Tensor& input, const MemoryConfig& output_mem_config) {
    auto arch = input.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor trunc_res = ttnn::trunc(input, output_mem_config);
    Tensor result = ttnn::subtract(input, trunc_res, std::nullopt, output_mem_config);
    return result;
}
Tensor frac(const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _frac)(input, output_mem_config);
}

Tensor _div_trunc(
    const Tensor& input_a,
    const Tensor& input_b,
    const MemoryConfig& output_mem_config) {
    auto arch = input_a.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor result = div(input_a, input_b, true);
    return ttnn::trunc(result);
}
Tensor div_trunc(
    const Tensor& input_a,
    const Tensor& input_b,
    const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _div_trunc)(input_a, input_b, output_mem_config);
}

Tensor _div_trunc_overload(
    const Tensor& input,
    float value,
    const MemoryConfig& output_mem_config) {
    auto arch = input.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor result = ttnn::multiply(input, (1 / value));
    return ttnn::trunc(result);
}
Tensor div_trunc(
    const Tensor& input,
    float value,
    const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _div_trunc_overload)(input, value, output_mem_config);
}

Tensor _unary_rdiv_trunc(
    float value,
    const Tensor& input,
    const MemoryConfig& output_mem_config) {
    auto arch = input.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor result = ttnn::multiply(ttnn::full_like(input, value), ttnn::reciprocal(input));
    return ttnn::trunc(result);
}
Tensor unary_rdiv_trunc(
    float value,
    const Tensor& input,
    const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _unary_rdiv_trunc)(value, input, output_mem_config);
}

Tensor is_odd(const Tensor& input, const MemoryConfig& output_mem_config) {
    Tensor result = ttnn::multiply(input, (1.0f/2.0f));
    Tensor floor_res = ttnn::floor(result);
    return ttnn::ne(result, floor_res);
}

Tensor _floor_div(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    auto arch = input_a.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor temp = div(input_a, input_b, true);
    // floor(nan, inf, -inf) = nan, inf, -inf
    return ttnn::where(
        ttnn::logical_or(
            ttnn::eq(temp, std::nanf("")),
            ttnn::logical_or(
                ttnn::eq(temp, std::numeric_limits<float>::infinity()),
                ttnn::eq(temp, -std::numeric_limits<float>::infinity()))),
        temp,
        ttnn::floor(temp, output_mem_config));
}
Tensor floor_div(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _floor_div)(input_a, input_b, output_mem_config);
}

Tensor _floor_div_overload(const Tensor& input, float value, const MemoryConfig& output_mem_config) {
    if (value == 0) {
        Tensor t_inf = full_like(input, std::numeric_limits<float>::infinity(), output_mem_config);
        Tensor t_nan = full_like(input, std::nanf(""), output_mem_config);
        return ttnn::where(
            ttnn::eqz(input, output_mem_config),
            t_nan,
            ttnn::multiply(t_inf, ttnn::sign(input, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config);

    }
    Tensor temp = ttnn::multiply(input, (1.0f/value));
    return ttnn::floor(temp);
}
Tensor floor_div(const Tensor& input_a, float value, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _floor_div_overload)(input_a, value, output_mem_config);
}

Tensor _rfloor_div(float value, const Tensor& input, const MemoryConfig& output_mem_config) {
    Tensor result = ttnn::multiply(ttnn::full_like(input, value), ttnn::reciprocal(input));
    return ttnn::floor(result, output_mem_config);
}
Tensor rfloor_div(float value, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _rfloor_div)(value, input, output_mem_config);
}

Tensor _div_no_nan(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    Tensor div_result = div(input_a, input_b);
    return ttnn::where(ttnn::eqz(input_b, output_mem_config), 0, div_result);
}
Tensor div_no_nan(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _div_no_nan)(input_a, input_b, output_mem_config);
}

Tensor _div_no_nan_overload(const Tensor& input_a, float value, const MemoryConfig& output_mem_config) {
    if (value == 0)
        return full_like(input_a, 0.0f, output_mem_config);
    else
        return ttnn::multiply(input_a, (1.0f/value));
}
Tensor div_no_nan(const Tensor& input_a, float value, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _div_no_nan_overload)(input_a, value, output_mem_config);
}

Tensor _remainder(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    DataType input_dtype = input_a.get_dtype();
    Tensor a = ttnn::typecast(input_a, DataType::FLOAT32);
    Tensor b = ttnn::typecast(input_b, DataType::FLOAT32);
    Tensor result = ttnn::subtract(a, ttnn::multiply(b, floor_div(input_a, input_b, output_mem_config), std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    result = ttnn::where(ttnn::ge(result, b), ttnn::subtract(result, b), result);
    result = ttnn::where(ttnn::ltz(b), ttnn::add(result, b), result);
    result = ttnn::where(ttnn::eq(a, b, std::nullopt, output_mem_config), full_like(input_a, 0.0f, output_mem_config), result, output_mem_config);
    return ttnn::typecast(result, input_dtype);
}
Tensor remainder(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _remainder)(input_a, input_b, output_mem_config);
}

Tensor _fmod(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    DataType input_dtype = input_a.get_dtype();
    Tensor a = ttnn::typecast(input_a, DataType::FLOAT32);
    Tensor b = ttnn::typecast(input_b, DataType::FLOAT32);
    Tensor result = ttnn::subtract(a, ttnn::multiply(div(input_a, input_b, true, "trunc", output_mem_config), b, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    result = ttnn::where(ttnn::eq(a, b, std::nullopt, output_mem_config), full_like(input_a, 0.0f, output_mem_config), result, output_mem_config);
    return ttnn::typecast(result, input_dtype);
}
Tensor fmod(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _fmod)(input_a, input_b, output_mem_config);
}

// logit(input, eps)=log(input / 1 - input)
Tensor _logit(const Tensor& input_a, float eps, const MemoryConfig& output_mem_config) {
    Tensor t_eps = full_like(input_a, eps, output_mem_config);
    Tensor t1m_eps = full_like(input_a, (1 - eps), output_mem_config);
    Tensor logit_input = ttnn::where(
        ttnn::ltz(t_eps, output_mem_config),
        input_a,
        ttnn::where(
            ttnn::lt(input_a, t_eps, std::nullopt, output_mem_config),
            t_eps,
            ttnn::where(ttnn::gt(input_a, t1m_eps, std::nullopt, output_mem_config), t1m_eps, input_a, output_mem_config),
            output_mem_config),
        output_mem_config);
    t_eps.deallocate();
    t1m_eps.deallocate();
    Tensor linput_m1 = ttnn::rsub(logit_input, 1.0, output_mem_config);
    Tensor log_input =
        ttnn::multiply(logit_input, ttnn::reciprocal(linput_m1, output_mem_config), std::nullopt, output_mem_config);
    linput_m1.deallocate();
    Tensor t_inf =
        ttnn::multiply(ttnn::sign(input_a, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor logit_result = ttnn::where(
        ttnn::eq(logit_input, 1.0, std::nullopt, output_mem_config),
        t_inf,
        ttnn::where(ttnn::ltz(log_input, output_mem_config), std::nanf(" "), ttnn::log(log_input, output_mem_config), output_mem_config),
        output_mem_config);
    return logit_result;
}
Tensor logit(const Tensor& input_a, float eps, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _logit)(input_a, eps, output_mem_config);
}

// logical_xori
Tensor _logical_xori(const Tensor& input_a, float value, const MemoryConfig& output_mem_config) {
    if (std::fpclassify(value) == FP_ZERO) {
        return ttnn::nez(input_a);
    } else {
        return ttnn::eqz(input_a);  // eqz( input_a ) = not( nez( input_a ) )
    }
}
Tensor logical_xori(const Tensor& input_a, float value, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _logical_xori)(input_a, value, output_mem_config);
}

// Celu
// torch.where(x > 0, x, alpha * (torch.exp(x / alpha) - 1))
Tensor _celu(const Tensor& input_a, float alpha, const MemoryConfig& output_mem_config) {
    float recip_val = 1.0f / alpha;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam{UnaryOpType::MUL_UNARY_SFPU, recip_val},
    UnaryWithParam{UnaryOpType::EXP, 1.0f},
    UnaryWithParam{UnaryOpType::SUB_UNARY_SFPU, 1.0f}, UnaryWithParam{UnaryOpType::MUL_UNARY_SFPU, alpha} };

    Tensor result = ttnn::unary_chain(input_a, ops_chain, output_mem_config);
    result = ttnn::where(ttnn::gtz(input_a, output_mem_config), input_a, result, output_mem_config);
    return result;
}
Tensor celu(const Tensor& input_a, float alpha, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _celu)(input_a, alpha, output_mem_config);
}


using HWFunctionT = std::function<Tensor(const Tensor& y, const MemoryConfig&)>;
Tensor _make_global_from_hw_impl(
    HWFunctionT fn, const Tensor& y, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    const Shape s_orig = y.get_legacy_shape();
    TT_FATAL(s_orig.rank() == 4, "Cannot support non-rank 4 Tensor");

    // format to HW
    Tensor y_hw = reshape(y, 1, 1, s_orig[2], s_orig[3] * s_orig[1] * s_orig[0], output_mem_config);

    // compute @fn
    Tensor z_0 = fn(y_hw, output_mem_config);
    TT_FATAL(y_hw.get_legacy_shape() == z_0.get_legacy_shape(), "shape match");
    y_hw.deallocate();

    // reformat
    Tensor z_1 = reshape(z_0, s_orig[0], s_orig[1], s_orig[2], s_orig[3], output_mem_config);
    z_0.deallocate();

    return z_1;
}

// Global Norm
Tensor _normalize_global(const Tensor& y, const MemoryConfig& output_mem_config) {
    return _make_global_from_hw_impl(ttnn::normalize_hw, y, output_mem_config);
}
Tensor normalize_global(const Tensor& y, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _normalize_global)(y, output_mem_config);
}

Tensor _scatter(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    tt::tt_metal::Array4D start_index = {0, 0, 0, 0};
    ttnn::Tensor input_tensor_4D = ttnn::unsqueeze_to_4D(input_a);

    Tensor index = ttnn::pad(0, ones_like(input_tensor_4D, output_mem_config), input_b.get_legacy_shape().to_array_4D(), start_index, 0, false, std::nullopt);
    Tensor temp_a = ttnn::pad(0, input_tensor_4D,input_b.get_legacy_shape().to_array_4D(), start_index, 0, false, std::nullopt);
    return ttnn::where(index, temp_a, input_b, output_mem_config);
}
Tensor scatter(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _scatter)(input_a, input_b, output_mem_config);
}

// on-device tensor creation 0s like @reference_tensor
Tensor zeros_like(
    uint8_t queue_id,
    const Tensor& reference_tensor,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return mk_zero_tensor_like(reference_tensor, output_mem_config, output_tensor);
}
Tensor zeros_like(
    const Tensor& reference_tensor, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return mk_zero_tensor_like(default_queue_id, reference_tensor, output_mem_config, output_tensor);
}

// on-device tensor creation 1s like @reference_tensor
Tensor ones_like(const Tensor& reference_tensor, const MemoryConfig& output_mem_config) {
    return mk_filled_tensor_like(reference_tensor, 1.0f, output_mem_config);
}

// on-device tensor creation with value like @reference_tensor
Tensor full_like(
    const Tensor& reference_tensor,
    float value,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return mk_filled_tensor_like(reference_tensor, value, output_mem_config, output_tensor, default_queue_id);
}
Tensor full_like(
    uint8_t queue_id,
    const Tensor& reference_tensor,
    float value,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return mk_filled_tensor_like(reference_tensor, value, output_mem_config, output_tensor, queue_id);
}

// hardtanh
Tensor _hardtanh(
    const Tensor& a, float low /* = -1.0f */, float high /* = +1.0f */, const MemoryConfig& output_mem_config) {
    return ttnn::clip(a, low, high, output_mem_config);
}
Tensor hardtanh(
    const Tensor& a, float low /* = -1.0f */, float high /* = +1.0f */, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _hardtanh)(a, low, high, output_mem_config);
}

// on-device tensor creation 0s with shape
Tensor zeros(
    const Shape shape, DataType data_type, Layout layout, Device* device, const MemoryConfig& output_mem_config) {
    return tt::numpy::zeros(shape, data_type, layout, device, output_mem_config);
}

Tensor empty(
    const Shape shape, DataType data_type, Layout layout, Device* device, const MemoryConfig& output_mem_config) {
    return create_device_tensor(shape, data_type, layout, device, output_mem_config);
}

// on-device tensor creation 1s with shape
Tensor ones(
    const Shape shape, DataType data_type, Layout layout, Device* device, const MemoryConfig& output_mem_config) {
    return tt::numpy::ones(shape, data_type, layout, device, output_mem_config);
}

// on-device tensor creation with shape and filled with value
Tensor full(
    const Shape shape,
    float value,
    DataType data_type,
    Layout layout,
    Device* device,
    const MemoryConfig& output_mem_config) {
    return tt::numpy::full(shape, value, data_type, layout, device, output_mem_config);
}

// on-device with increment
Tensor arange(int32_t start, int32_t end, int32_t step, Device* device, const MemoryConfig& output_mem_config) {
    return tt::numpy::arange<bfloat16>(start, end, step, Layout::ROW_MAJOR, device, output_mem_config);
}

/**
 * outer product = matrix multiply when a = [1,1,N,1] and b = [1,1,1,M]
 * and result is of size [1,1,N,M].
 * - implementation supports any 1D "squeezable tensor" at input operands
 *   by running reshape.
 */
Tensor _outer(Tensor& a, Tensor& b, const MemoryConfig& output_mem_config) {
    const Shape s_a = a.get_legacy_shape();
    const Shape s_b = b.get_legacy_shape();

    auto num_ones = [](const Shape& s) -> uint32_t {
        uint32_t num1s = 0;
        for (uint32_t idx = 0; idx < 4; idx++) num1s += (uint32_t)(s[idx] == 1);
        return num1s;
    };

    // check if 3 dimensions are 1
    TT_ASSERT(!(num_ones(s_a) < 3), "3 dimensions are required to be 1 for use with outer product");
    TT_ASSERT(!(num_ones(s_b) < 3), "3 dimensions are required to be 1 for use with outer product");

    const bool skip_reshape_a = (s_a[0] == 1 && s_a[1] == 1 && s_a[2] >= 1 && s_a[3] == 1);
    const bool skip_reshape_b = (s_b[0] == 1 && s_b[1] == 1 && s_b[2] == 1 && s_b[3] >= 1);

    Tensor a_slim = a;
    Tensor b_slim = b;

    if (!skip_reshape_a) {
        a_slim = reshape(a, 1, 1, a.volume(), 1, output_mem_config);
    }
    if (!skip_reshape_b) {
        b_slim = reshape(b, 1, 1, 1, b.volume(), output_mem_config);
    }
    a_slim = ttnn::to_layout(a_slim, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
    b_slim = ttnn::to_layout(b_slim, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
    Device* device = AutoFormat::GetDefaultDevice();
    if (device != nullptr) {
        if (a_slim.storage_type() != tt::tt_metal::StorageType::DEVICE) {
            a_slim = AutoFormat::move_tensor_to_device(a_slim, device);
        }
        if (b_slim.storage_type() != tt::tt_metal::StorageType::DEVICE) {
            b_slim = AutoFormat::move_tensor_to_device(b_slim, device);
        }
    }

    return ttnn::operations::matmul::matmul(
            a_slim,
            b_slim,
            /*bias=*/std::nullopt,
            tt::operations::primary::Matmul{
            /*program_config=*/std::nullopt,
            /*bcast_batch=*/std::nullopt,
            output_mem_config}
            );
}
Tensor outer(Tensor& a, Tensor& b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _outer)(a, b, output_mem_config);
}

std::vector<Tensor> split_tensor_for_glu(const Tensor& input_a, int32_t dim, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> t_split;
    Shape inshape = input_a.get_legacy_shape();
    TT_FATAL(((inshape[dim] / 2) % TILE_WIDTH == 0), "Split tensor dimension should be in full tile");
    std::vector<uint32_t> s_a = {0, 0, 0, 0};
    std::vector<uint32_t> e_a = {inshape[0] - 1, inshape[1] - 1, inshape[2] - 1, inshape[3] / 2 - 1};

    std::vector<uint32_t> s_b = {0, 0, 0, inshape[3] / 2};
    std::vector<uint32_t> e_b = {inshape[0] - 1, inshape[1] - 1, inshape[2] - 1, inshape[3] - 1};

    Tensor t_a = ttnn::slice(0, input_a, s_a, e_a, output_mem_config);
    Tensor t_b = ttnn::slice(0, input_a, s_b, e_b, output_mem_config);

    t_split.emplace_back(t_a);
    t_split.emplace_back(t_b);

    return t_split;
}


// on-device tensor creation with shape and filled with value
Tensor _sfpu_eps(const Shape shape, Layout layout, Device* device, const MemoryConfig& output_mem_config) {
    float value = device->sfpu_eps();
    return tt::numpy::full(shape, value, DataType::BFLOAT16, layout, device, output_mem_config);
}
Tensor sfpu_eps(const Shape shape, Layout layout, Device* device, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _sfpu_eps)(shape, layout, device, output_mem_config);
}

Tensor create_mask(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    auto& padded_shape = input_a.get_legacy_shape();
    auto& unpadded_shape = padded_shape.without_padding();
    if (padded_shape == unpadded_shape)
        return input_a;
    float t_inf = -std::numeric_limits<float>::infinity();
    Tensor masked_input = tt::numpy::mask_padded_input<bfloat16>(padded_shape, unpadded_shape, DataType::BFLOAT16);
    masked_input = ttnn::where(masked_input, input_a, t_inf, output_mem_config);
    return masked_input;
}

}  // namespace tt_metal

}  // namespace tt
