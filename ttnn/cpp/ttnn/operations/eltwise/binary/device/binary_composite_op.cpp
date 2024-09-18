// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_composite_op.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/types.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/ternary/where.hpp"
#include "ttnn/cpp/ttnn/operations/copy.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"

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

Tensor ExecuteDiv::invoke(uint8_t queue_id, const Tensor& input, float value, bool accurate_mode, const std::string& round_mode, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> output_tensor) {
    TT_FATAL((round_mode == "None" || round_mode == "trunc" || round_mode == "floor"), "Incorrect rounding mode (expected 'None', 'trunc', or 'floor')");
    output_tensor = output_tensor.value_or(ttnn::zeros_like(input));
    ttnn::multiply(queue_id, input, (1.0f/value), std::nullopt, output_mem_config, output_tensor);
    if(round_mode == "trunc"){
        ttnn::trunc(queue_id, output_tensor.value(), output_mem_config, output_tensor);
    }
    else if(round_mode == "floor"){
        ttnn::floor(queue_id, output_tensor.value(), output_mem_config, output_tensor);
    }
    return output_tensor.value();
}

Tensor ExecuteDiv::invoke(const Tensor& input, float value, bool accurate_mode, const std::string& round_mode, const std::optional<MemoryConfig>& output_mem_config) {
   return ExecuteDiv::invoke(DefaultQueueId, input, value, accurate_mode, round_mode, output_mem_config);
}

Tensor ExecuteDiv::invoke(uint8_t queue_id, const Tensor& input_a, const Tensor& input_b, bool accurate_mode, const std::string& round_mode, const std::optional<MemoryConfig>& output_mem_config, std::optional<Tensor> output_tensor) {
    TT_FATAL((round_mode == "None" || round_mode == "trunc" || round_mode == "floor"), "Incorrect rounding mode (expected 'None', 'trunc', or 'floor')");
    output_tensor = output_tensor.value_or(ttnn::zeros_like(input_a));
    auto arch = input_a.device()->arch();
    if (arch == tt::ARCH::WORMHOLE_B0) {
        DataType input_dtype = input_a.get_dtype();
        Tensor a = typecast(input_a, DataType::FLOAT32);
        Tensor b = typecast(input_b, DataType::FLOAT32);
        ttnn::divide(queue_id, a, b, std::nullopt, std::nullopt, output_tensor);

        if(round_mode == "trunc"){
            ttnn::trunc(queue_id, output_tensor.value(), output_mem_config, output_tensor);
        }
        else if(round_mode == "floor"){
            ttnn::floor(queue_id, output_tensor.value(), output_mem_config, output_tensor);
        }

        if (accurate_mode == false) {  // If input_b is non-zero tensor
            return typecast(queue_id, output_tensor.value(), input_dtype, output_mem_config, output_tensor);
        }

        Tensor t_inf = ttnn::full_like(input_a, std::numeric_limits<float>::infinity());
        Tensor t_nan = ttnn::full_like(input_a, std::nanf(""));
        return typecast(queue_id, where(
            queue_id,
            ttnn::eqz(queue_id, input_b, output_mem_config),
            ttnn::where(
                queue_id,
                ttnn::eqz(queue_id, input_a, output_mem_config),
                t_nan,
                ttnn::multiply(queue_id, t_inf, ttnn::sign(input_a, output_mem_config), std::nullopt, output_mem_config)),
            output_tensor.value()),
            input_dtype,
            output_mem_config,
            output_tensor);
    } else {
        ttnn::divide(queue_id, input_a, input_b, std::nullopt, std::nullopt, output_tensor);

        if(round_mode == "trunc"){
            ttnn::trunc(queue_id, output_tensor.value(), output_mem_config, output_tensor);
        }
        else if(round_mode == "floor"){
            ttnn::floor(queue_id, output_tensor.value(), output_mem_config, output_tensor);
        }

        if (accurate_mode == false) {  // If input_b is non-zero tensor
            return output_tensor.value();
        }

        Tensor t_inf = ttnn::full_like(queue_id, input_a, std::numeric_limits<float>::infinity());
        Tensor t_nan = ttnn::full_like(queue_id, input_a, std::nanf(""));
        return ttnn::where(
            queue_id,
            ttnn::eqz(queue_id, input_b, output_mem_config),
            ttnn::where(
                queue_id,
                ttnn::eqz(queue_id, input_a, output_mem_config),
                t_nan,
                ttnn::multiply(queue_id, t_inf, ttnn::sign(input_a, output_mem_config), std::nullopt, output_mem_config)),
            output_tensor.value(),
            output_mem_config,
            output_tensor);
    }
}

Tensor ExecuteDiv::invoke(const Tensor& input_a, const Tensor& input_b, bool accurate_mode, const std::string& round_mode, const std::optional<MemoryConfig>& output_mem_config) {
   return ExecuteDiv::invoke(DefaultQueueId, input_a, input_b, accurate_mode, round_mode, output_mem_config);
}

Tensor _div_no_nan_overload(const Tensor& input_a, float value, const std::optional<MemoryConfig>& output_mem_config) {
    if (value == 0)
        return ttnn::full_like(input_a, 0.0f);
    else
        return ttnn::multiply(input_a, (1.0f/value));
}

Tensor _div_no_nan(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor div_result = ttnn::div(input_a, input_b, false, "None", output_mem_config);
    return ttnn::where(ttnn::eqz(input_b, output_mem_config), 0, div_result);
}

// Binary remainder will be overloaded by unary remainder in another PR
Tensor ExecuteBinaryRemainder::invoke(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    auto arch = input_a.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    DataType input_dtype = input_a.get_dtype();
    Tensor a = typecast(input_a, DataType::FLOAT32);
    Tensor b = typecast(input_b, DataType::FLOAT32);
    Tensor result = ttnn::subtract(a, ttnn::multiply(b, ttnn::div(input_a, input_b, true, "floor", output_mem_config), std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    result = ttnn::where(ttnn::ge(result, b), ttnn::subtract(result, b), result);
    result = ttnn::where(ttnn::ltz(b), ttnn::add(result, b), result);
    result = ttnn::where(ttnn::eq(a, b, std::nullopt, output_mem_config), ttnn::full_like(input_a, 0.0f), result);
    return typecast(result, input_dtype);
}


Tensor ExecuteBinaryRemainder::invoke(const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::unary_remainder(input, scalar);
}

// Binary FMOD will be overloaded by unary FMOD in another PR
Tensor ExecuteBinaryFmod::invoke(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    auto arch = input_a.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    DataType input_dtype = input_a.get_dtype();
    Tensor a = typecast(input_a, DataType::FLOAT32);
    Tensor b = typecast(input_b, DataType::FLOAT32);
    Tensor result = ttnn::subtract(a, ttnn::multiply(ttnn::div(input_a, input_b, true, "trunc", output_mem_config), b, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    result = ttnn::where(ttnn::eq(a, b, std::nullopt, output_mem_config), ttnn::full_like(input_a, 0.0f), result);
    return typecast(result, input_dtype);
}

Tensor ExecuteBinaryFmod::invoke(const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::unary_fmod(input, scalar);
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
    Tensor temp = ttnn::div(input_a, input_b, true, "None", output_mem_config);
    Tensor result = ttnn::div(input_a, input_b, true, "floor", output_mem_config);
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

Tensor _logical_xor_(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor in_a_eq_zero = ttnn::eqz(input_a, output_mem_config, input_a );
    Tensor in_b_eq_zero = ttnn::nez(input_b, output_mem_config, input_b );
    in_b_eq_zero = ttnn::eqz(input_b, output_mem_config);
    Tensor result = ttnn::where(input_a, input_b, in_b_eq_zero, output_mem_config, input_a);
    return result;
}

Tensor _scatter(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    tt::tt_metal::Array4D start_index = {0, 0, 0, 0};
    Tensor index_pad = ttnn::pad(0, ttnn::ones_like(input_a), input_b.get_legacy_shape().to_array_4D(), start_index, 0, false, std::nullopt);
    Tensor temp_a = ttnn::pad(0, input_a, input_b.get_legacy_shape().to_array_4D(), start_index, 0, false, std::nullopt);
    return ttnn::where(index_pad, temp_a, input_b);
}

/**
 * outer product = matrix multiply when a = [1,1,N,1] and b = [1,1,1,M]
 * and result is of size [1,1,N,M].
 * - implementation supports any 1D "squeezable tensor" at input operands
 *   by running reshape.
 */
Tensor _outer(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    const tt::tt_metal::LegacyShape s_a = input_a.get_legacy_shape();
    const tt::tt_metal::LegacyShape s_b = input_b.get_legacy_shape();
    auto num_ones = [](const tt::tt_metal::LegacyShape& s) -> uint32_t {
        uint32_t num1s = 0;
        for (uint32_t idx = 0; idx < 4; idx++) num1s += (uint32_t)(s[idx] == 1);
        return num1s;
    };

    // check if 3 dimensions are 1
    TT_FATAL((num_ones(s_a) >= 3), "3 dimensions are required to be 1 for use with outer product");
    TT_FATAL((num_ones(s_b) >= 3), "3 dimensions are required to be 1 for use with outer product");

    const bool skip_reshape_a = (s_a[0] == 1 && s_a[1] == 1 && s_a[2] >= 1 && s_a[3] == 1);
    const bool skip_reshape_b = (s_b[0] == 1 && s_b[1] == 1 && s_b[2] == 1 && s_b[3] >= 1);

    Tensor a_slim = input_a;
    Tensor b_slim = input_b;

    if(!skip_reshape_a) {
        a_slim = ttnn::reshape(input_a, ttnn::Shape{std::array<uint32_t, 4>{1, 1, input_a.volume(), 1}});
    }
    if(!skip_reshape_b) {
        b_slim = ttnn::reshape(input_b, ttnn::Shape{std::array<uint32_t, 4>{1, 1, 1, input_b.volume()}});
    }
    a_slim = ttnn::to_layout(a_slim, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
    b_slim = ttnn::to_layout(b_slim, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);

    auto device = ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice();
    if(device != nullptr) {
        if (a_slim.storage_type() != tt::tt_metal::StorageType::DEVICE) {
            a_slim = ttnn::operations::experimental::auto_format::AutoFormat::move_tensor_to_device(a_slim, device);
        }
        if (b_slim.storage_type() != tt::tt_metal::StorageType::DEVICE) {
            b_slim = ttnn::operations::experimental::auto_format::AutoFormat::move_tensor_to_device(b_slim, device);
        }
    }

    return ttnn::matmul(a_slim, b_slim);
}

Tensor _polyval(const Tensor& input_a, const std::vector<float>& coeffs, const std::optional<MemoryConfig>& output_mem_config) {
    TT_ASSERT(coeffs.size() != 0 && "coeffs should be 1 or more coefficients");
    if (coeffs.size() == 1) {
        return ttnn::full_like(input_a, coeffs[0]);
    }
    Tensor result = ttnn::multiply(input_a, coeffs[0], std::nullopt, output_mem_config);
    for (int idx = 1; idx < coeffs.size() - 1; idx++) {
        result = ttnn::add(result, coeffs[idx], std::nullopt, output_mem_config);
        result = ttnn::multiply(input_a, result, std::nullopt, output_mem_config);
    }
    Tensor final_tensor = ttnn::add(result, coeffs.back(), std::nullopt, output_mem_config);
    return final_tensor;
}

} // namespace ttnn::operations::binary
