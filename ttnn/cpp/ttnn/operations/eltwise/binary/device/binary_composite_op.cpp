// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_composite_op.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/types.hpp"
#include "tt_metal/common/bfloat16.hpp"

namespace ttnn::operations::binary{


Tensor _hypot(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor a_sq = ttnn::square(input_a, output_mem_config);
    Tensor b_sq = ttnn::square(input_b, output_mem_config);
    Tensor c_sq = ttnn::add(a_sq, b_sq, std::nullopt, output_mem_config);
    a_sq.deallocate();
    b_sq.deallocate();
    return ttnn::sqrt(c_sq, output_mem_config);
}

// xlogy(x,y)=x*log(y)
Tensor _xlogy(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor t_nan = ttnn::full_like(input_b, std::nanf(" "));
    Tensor result = ttnn::multiply(input_a, ttnn::log(input_b, output_mem_config), std::nullopt, output_mem_config);
    result = ttnn::where(
        ttnn::logical_or(
            ttnn::ltz(input_b, output_mem_config),
            ttnn::eq(input_b, t_nan, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        t_nan,
        result);
    return result;
}

// subalpha(input,other,alpha)=input-alpha*other
Tensor _subalpha(const Tensor& input_a, const Tensor& input_b, float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor result = ttnn::add(
        ttnn::neg(ttnn::multiply(input_b, alpha, std::nullopt, output_mem_config), output_mem_config), input_a, std::nullopt, output_mem_config);
    return result;
}

// addalpha(input, other, alpha) = input + (alpha * other)
Tensor _addalpha(
    const Tensor& input_a,
    const Tensor& input_b,
    float alpha,
    const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::add(ttnn::multiply(input_b, alpha, std::nullopt, output_mem_config), input_a, std::nullopt, output_mem_config);
}


// nextafter
Tensor _nextafter(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    const float eps = input_a.device()->sfpu_eps();
    Tensor result(input_a);
    {
        Tensor eps_gt(input_a);
        {
            eps_gt = ttnn::where(
                ttnn::gt(input_a, input_b, std::nullopt, output_mem_config),
                ttnn::add(input_a, eps, std::nullopt, output_mem_config),
                input_a);
        }
        result = ttnn::where(
            ttnn::lt(input_a, input_b, std::nullopt, output_mem_config),
            ttnn::subtract(input_a, eps, std::nullopt, output_mem_config),
            eps_gt);
    }
    return result;
}

// ∣input−other∣≤ atol+rtol×∣other∣
Tensor _isclose(
    const Tensor& input_a,
    const Tensor& input_b,
    float rtol,
    float atol,
    bool equal_nan,
    const std::optional<MemoryConfig>& output_mem_config) {
    Tensor value1 = input_a;
    Tensor value2 = input_b;
    if (!equal_nan) {
        value1 = ttnn::where(ttnn::isnan(value1, output_mem_config), 1.0f, value1);
        value2 = ttnn::where(ttnn::isnan(value2, output_mem_config), 0.0f, value2);
    }
    Tensor is_close_lhs = ttnn::abs(ttnn::subtract(value1, value2, std::nullopt, output_mem_config), output_mem_config);
    Tensor is_close_rhs = input_b;
    Tensor mul_result = ttnn::multiply(ttnn::abs(value2, output_mem_config), rtol, std::nullopt, output_mem_config);
    is_close_rhs = ttnn::add(mul_result, atol, std::nullopt, output_mem_config);
    mul_result.deallocate();
    Tensor result = ttnn::where(ttnn::le(is_close_lhs, is_close_rhs, std::nullopt, output_mem_config), 1.0, 0.0);
    return result;
}

// minimum(a,b) = a - (a - b > 0 )*(a-b)
Tensor _minimum(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor t_diff = ttnn::subtract(input_a, input_b, std::nullopt, output_mem_config);
    Tensor result = ttnn::where(t_diff, input_b, input_a);
    return result;
}

// maximum(a,b) = a + (b - a > 0 )*(b-a)
Tensor _maximum(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor t_diff = ttnn::subtract(input_b, input_a, std::nullopt, output_mem_config);
    Tensor result = ttnn::where(t_diff, input_b, input_a);
    return result;
}

Tensor _atan2(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor result(input_a);
    {
        Tensor atan_input = ttnn::multiply(
            ttnn::abs(input_b, output_mem_config),
            ttnn::reciprocal(ttnn::abs(input_a, output_mem_config), output_mem_config),
            std::nullopt,
            output_mem_config);
        result = ttnn::atan(atan_input, output_mem_config);
    }
    Tensor res(result);
    {
        Tensor ib_gtz = ttnn::gtz(input_b, output_mem_config);
        Tensor ib_gt = ttnn::gtz(input_b, output_mem_config);
        Tensor ib_lt = ttnn::ltz(input_b, output_mem_config);
        float pi_2 = M_PI_2;
        Tensor neg_result = ttnn::neg(result, output_mem_config);

        res = ttnn::where(
            ttnn::gtz(input_a, output_mem_config),
            ttnn::where(ib_gtz, result, neg_result),
            ttnn::where(
                ttnn::ltz(input_a, output_mem_config),
                ttnn::where(
                    ib_gt,
                    ttnn::add(neg_result, M_PI, std::nullopt, output_mem_config),
                    ttnn::where(ib_lt, ttnn::subtract(result, M_PI, std::nullopt, output_mem_config), M_PI)),
                ttnn::where(ib_gt, pi_2, ttnn::where(ib_lt, -pi_2, 0.0f))));
    }
    return res;
}

Tensor _logical_xor(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor in_a_eq_zero = ttnn::eqz(input_a, output_mem_config);
    Tensor in_b_eq_zero = ttnn::eqz(input_b, output_mem_config);
    Tensor in_b_neq_zero = ttnn::nez(input_b, output_mem_config);
    Tensor result = ttnn::where(in_a_eq_zero, in_b_neq_zero, in_b_eq_zero);
    return result;
}

Tensor _div_overload(const Tensor& input_a, float value, bool accurate_mode, std::string round_mode, const std::optional<MemoryConfig>& output_mem_config) {
    TT_FATAL((round_mode == "None" || round_mode == "trunc" || round_mode == "floor") && "Incorrect rounding mode (expected 'None', 'trunc', or 'floor')");
    Tensor result = ttnn::multiply(input_a, (1.0f/value), std::nullopt, output_mem_config);
    if(round_mode == "trunc"){
        result = trunc(result);
    }
    else if(round_mode == "floor"){
        result = ttnn::floor(result);
    }
    return result;
}

Tensor _div(const Tensor& input_a, const Tensor& input_b, bool accurate_mode, std::string round_mode, const std::optional<MemoryConfig>& output_mem_config) {
    TT_FATAL((round_mode == "None" || round_mode == "trunc" || round_mode == "floor") && "Incorrect rounding mode (expected 'None', 'trunc', or 'floor')");
    auto arch = input_a.device()->arch();
    if (arch == tt::ARCH::WORMHOLE_B0) {
        DataType input_dtype = input_a.get_dtype();
        Tensor a = typecast(input_a, DataType::FLOAT32);
        Tensor b = typecast(input_b, DataType::FLOAT32);
        Tensor result = ttnn::divide(a, b);

        if(round_mode == "trunc"){
            result = trunc(result);
        }
        else if(round_mode == "floor"){
            result = ttnn::floor(result);
        }

        if (accurate_mode == false) {  // If input_b is non-zero tensor
            return typecast(result, input_dtype);
        }

        Tensor t_inf = ttnn::full_like(input_a, std::numeric_limits<float>::infinity());
        Tensor t_nan = ttnn::full_like(input_a, std::nanf(""));
        return typecast(where(
            ttnn::eqz(input_b, output_mem_config),
            ttnn::where(
                ttnn::eqz(input_a, output_mem_config),
                t_nan,
                ttnn::multiply(t_inf, ttnn::sign(input_a, output_mem_config), std::nullopt, output_mem_config)),
            result),
            input_dtype);
    } else {
        Tensor result = ttnn::divide(input_a, input_b);

        if(round_mode == "trunc"){
            result = trunc(result);
        }
        else if(round_mode == "floor"){
            result = ttnn::floor(result);
        }

        if (accurate_mode == false) {  // If input_b is non-zero tensor
            return result;
        }

        Tensor t_inf = ttnn::full_like(input_a, std::numeric_limits<float>::infinity());
        Tensor t_nan = ttnn::full_like(input_a, std::nanf(""));
        return ttnn::where(
            ttnn::eqz(input_b, output_mem_config),
            ttnn::where(
                ttnn::eqz(input_a, output_mem_config),
                t_nan,
                ttnn::multiply(t_inf, ttnn::sign(input_a, output_mem_config), std::nullopt, output_mem_config)),
            result);
    }
}

Tensor _div_no_nan_overload(const Tensor& input_a, float value, const std::optional<MemoryConfig>& output_mem_config) {
    if (value == 0)
        return ttnn::full_like(input_a, 0.0f);
    else
        return ttnn::multiply(input_a, (1.0f/value));
}

Tensor _div_no_nan(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor div_result = _div(input_a, input_b, false, "None", output_mem_config);
    return ttnn::where(ttnn::eqz(input_b, output_mem_config), 0, div_result);
}

// Binary remainder will be overloaded by unary remainder in another PR
Tensor _binary_remainder(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    auto arch = input_a.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    DataType input_dtype = input_a.get_dtype();
    Tensor a = typecast(input_a, DataType::FLOAT32);
    Tensor b = typecast(input_b, DataType::FLOAT32);
    Tensor result = ttnn::subtract(a, ttnn::multiply(b, _div(input_a, input_b, true, "floor", output_mem_config), std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    result = ttnn::where(ttnn::ge(result, b), ttnn::subtract(result, b), result);
    result = ttnn::where(ttnn::ltz(b), ttnn::add(result, b), result);
    result = ttnn::where(ttnn::eq(a, b, std::nullopt, output_mem_config), ttnn::full_like(input_a, 0.0f), result);
    return typecast(result, input_dtype);
}

// Binary FMOD will be overloaded by unary FMOD in another PR
Tensor _binary_fmod(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    auto arch = input_a.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    DataType input_dtype = input_a.get_dtype();
    Tensor a = typecast(input_a, DataType::FLOAT32);
    Tensor b = typecast(input_b, DataType::FLOAT32);
    Tensor result = ttnn::subtract(a, ttnn::multiply(_div(input_a, input_b, true, "trunc", output_mem_config), b, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    result = ttnn::where(ttnn::eq(a, b, std::nullopt, output_mem_config), ttnn::full_like(input_a, 0.0f), result);
    return typecast(result, input_dtype);
}

Tensor _floor_div_overload(const Tensor& input_a, float value, const std::optional<MemoryConfig>& output_mem_config) {
    auto arch = input_a.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    if (value == 0) {
        Tensor t_inf = ttnn::full_like(input_a, std::numeric_limits<float>::infinity());
        Tensor t_nan = ttnn::full_like(input_a, std::nanf(""));
        return ttnn::where(
            ttnn::eqz(input_a, output_mem_config),
            t_nan,
            ttnn::multiply(t_inf, ttnn::sign(input_a, output_mem_config), std::nullopt, output_mem_config));
    }
    Tensor temp = ttnn::multiply(input_a, (1.0f/value), std::nullopt, output_mem_config);
    return ttnn::floor(temp);
}

Tensor _floor_div(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    auto arch = input_a.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor temp = _div(input_a, input_b, true, "None", output_mem_config);
    Tensor result = _div(input_a, input_b, true, "floor", output_mem_config);
    // floor(nan, inf, -inf) = nan, inf, -inf
    return ttnn::where(
        ttnn::logical_or(
            ttnn::eq(temp, std::nanf("")),
            ttnn::logical_or(
                ttnn::eq(temp, std::numeric_limits<float>::infinity()),
                ttnn::eq(temp, -std::numeric_limits<float>::infinity()))),
        temp,
        result);
}

Tensor _logical_and_(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::logical_and(input_a, input_b, std::nullopt, output_mem_config, input_a);
}

Tensor _logical_or_(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::logical_or(input_a, input_b, std::nullopt, output_mem_config, input_a);
}

Tensor _logical_xor_(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor in_a_eq_zero = ttnn::eqz(input_a, output_mem_config, input_a );
    Tensor in_b_eq_zero = ttnn::nez(input_b, output_mem_config, input_b );
    in_b_eq_zero = ttnn::eqz(input_b, output_mem_config);
    Tensor result = ttnn::where(input_a, input_b, in_b_eq_zero, output_mem_config, input_a);
    return result;
}

} // namespace ttnn::operations::binary
