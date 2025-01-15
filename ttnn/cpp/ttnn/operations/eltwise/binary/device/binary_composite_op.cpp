// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_composite_op.hpp"
#include <magic_enum/magic_enum.hpp>
#include <utility>
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/types.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/experimental/hal.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/ternary/where.hpp"
#include "ttnn/cpp/ttnn/operations/copy.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"

namespace ttnn::operations::binary {

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
    float t_nan = std::nanf(" ");
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
Tensor _subalpha(
    const Tensor& input_a, const Tensor& input_b, float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor result = ttnn::add(
        ttnn::neg(ttnn::multiply(input_b, alpha, std::nullopt, output_mem_config), output_mem_config),
        input_a,
        std::nullopt,
        output_mem_config);
    return result;
}

// addalpha(input, other, alpha) = input + (alpha * other)
Tensor _addalpha(
    const Tensor& input_a, const Tensor& input_b, float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::add(
        ttnn::multiply(input_b, alpha, std::nullopt, output_mem_config), input_a, std::nullopt, output_mem_config);
}

// nextafter
Tensor _nextafter(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    const float eps = tt::tt_metal::experimental::hal::get_eps();
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
Tensor ExecuteMinimum::invoke(
    const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor t_diff = ttnn::subtract(input_a, input_b, std::nullopt, output_mem_config);
    Tensor result = ttnn::where(t_diff, input_b, input_a);
    return result;
}

Tensor ExecuteMinimum::invoke(
    const Tensor& input_a, float value, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor t_diff = ttnn::subtract(input_a, value, std::nullopt, output_mem_config);
    Tensor result = ttnn::where(t_diff, value, input_a);
    return result;
}

// maximum(a,b) = a + (b - a > 0 )*(b-a)
Tensor ExecuteMaximum::invoke(
    const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor t_diff = ttnn::subtract(input_b, input_a, std::nullopt, output_mem_config);
    Tensor result = ttnn::where(t_diff, input_b, input_a);
    return result;
}

Tensor ExecuteMaximum::invoke(
    const Tensor& input_a, float value, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor t_diff = ttnn::rsub(input_a, value, output_mem_config);
    Tensor result = ttnn::where(t_diff, value, input_a);
    return result;
}

Tensor _atan2(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor result(input_a);
    {
        Tensor atan_input =
            ttnn::multiply(input_b, ttnn::reciprocal(input_a, output_mem_config), std::nullopt, output_mem_config);
        result = ttnn::atan(atan_input, output_mem_config);
    }
    Tensor res(result);
    {
        Tensor ia_gtz = ttnn::gtz(input_a, output_mem_config);
        Tensor ia_ltz = ttnn::ltz(input_a, output_mem_config);
        Tensor ib_ltz = ttnn::ltz(input_b, output_mem_config);

        Tensor altz_bgte = ttnn::logical_and(ia_ltz, ttnn::ge(input_b, 0.0), std::nullopt, output_mem_config);
        Tensor altz_bltz = ttnn::logical_and(ia_ltz, ib_ltz, std::nullopt, output_mem_config);

        Tensor a_eqz = ttnn::eqz(input_a, output_mem_config);
        Tensor b_gtz = ttnn::gtz(input_b, output_mem_config);
        Tensor b_eqz = ttnn::eqz(input_b, output_mem_config);

        Tensor az_bltz = ttnn::logical_and(a_eqz, ib_ltz, std::nullopt, output_mem_config);
        Tensor az_bgtz = ttnn::logical_and(a_eqz, b_gtz, std::nullopt, output_mem_config);
        Tensor az_bz = ttnn::logical_and(a_eqz, b_eqz, std::nullopt, output_mem_config);
        float pi_2 = M_PI_2;
        res = ttnn::where(
            ia_gtz,
            result,
            ttnn::where(
                altz_bgte,
                ttnn::add(result, M_PI, std::nullopt, output_mem_config),
                ttnn::where(
                    altz_bltz,
                    ttnn::subtract(result, M_PI, std::nullopt, output_mem_config),
                    ttnn::where(
                        az_bltz, M_PI_2, ttnn::where(az_bgtz, -M_PI_2, 0.0, output_mem_config), output_mem_config),
                    output_mem_config),
                output_mem_config),
            output_mem_config);
    }
    return res;
}

Tensor ExecuteDiv::invoke(
    uint8_t queue_id,
    const Tensor& input,
    float value,
    bool accurate_mode,
    const std::optional<std::string>& round_mode,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    TT_FATAL(
        (round_mode == std::nullopt || round_mode == "trunc" || round_mode == "floor"),
        "Incorrect rounding mode (expected None, 'trunc', or 'floor')");
    output_tensor = output_tensor.value_or(ttnn::zeros_like(input));
    ttnn::multiply(queue_id, input, (1.0f / value), std::nullopt, output_mem_config, output_tensor);
    if (round_mode == "trunc") {
        ttnn::trunc(queue_id, output_tensor.value(), output_mem_config, output_tensor);
    } else if (round_mode == "floor") {
        ttnn::floor(queue_id, output_tensor.value(), output_mem_config, output_tensor);
    }
    return output_tensor.value();
}

Tensor ExecuteDiv::invoke(
    const Tensor& input,
    float value,
    bool accurate_mode,
    const std::optional<std::string>& round_mode,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return ExecuteDiv::invoke(
        DefaultQueueId, input, value, accurate_mode, round_mode, output_mem_config, std::move(output_tensor));
}

Tensor ExecuteDiv::invoke(
    uint8_t queue_id,
    const Tensor& input_a,
    const Tensor& input_b,
    bool accurate_mode,
    const std::optional<std::string>& round_mode,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    TT_FATAL(
        (round_mode == std::nullopt || round_mode == "trunc" || round_mode == "floor"),
        "Incorrect rounding mode (expected None, 'trunc', or 'floor')");
    output_tensor = output_tensor.value_or(ttnn::empty_like(input_a));
    auto arch = input_a.device()->arch();
    if (arch != tt::ARCH::GRAYSKULL) {
        DataType input_dtype = input_a.get_dtype();

        // No accurate_mode for FP32 div as inf/nan are handled at kernel level
        if (input_dtype == DataType::FLOAT32 && input_b.get_dtype() == DataType::FLOAT32) {
            Tensor result = ttnn::divide(queue_id, input_a, input_b, std::nullopt, output_mem_config, output_tensor);
            if (round_mode == "trunc") {
                result = ttnn::trunc(queue_id, result, output_mem_config, output_tensor);
            } else if (round_mode == "floor") {
                result = ttnn::floor(queue_id, result, output_mem_config, output_tensor);
            }
            return result;
        }

        Tensor a = typecast(queue_id, input_a, DataType::FLOAT32);
        Tensor b = typecast(queue_id, input_b, DataType::FLOAT32);

        // Div operation without inf/nan handling as reciprocal(0) = 1.7014118346046923e+38 not inf/nan
        Tensor result =
            ttnn::multiply(queue_id, a, ttnn::reciprocal(b), std::nullopt, output_mem_config, output_tensor);

        if (round_mode == "trunc") {
            result = ttnn::trunc(queue_id, result, output_mem_config, output_tensor);
        } else if (round_mode == "floor") {
            result = ttnn::floor(queue_id, result, output_mem_config, output_tensor);
        }

        if (accurate_mode == false) {  // If input_b is non-zero tensor
            return typecast(queue_id, result, input_dtype, std::nullopt, output_tensor);
        }

        float t_nan = std::nanf("");
        float t_inf = std::numeric_limits<float>::infinity();
        typecast(
            queue_id,
            where(
                queue_id,
                ttnn::eqz(queue_id, input_b, output_mem_config),
                ttnn::where(
                    queue_id,
                    ttnn::eqz(queue_id, input_a, output_mem_config),
                    t_nan,
                    ttnn::multiply(
                        queue_id,
                        ttnn::sign(queue_id, input_a, output_mem_config),
                        t_inf,
                        std::nullopt,
                        output_mem_config)),
                result),
            input_dtype,
            std::nullopt,
            output_tensor);
        return output_tensor.value();
    } else {
        ttnn::divide(queue_id, input_a, input_b, std::nullopt, std::nullopt, output_tensor);

        if (round_mode == "trunc") {
            ttnn::trunc(queue_id, output_tensor.value(), output_mem_config, output_tensor);
        } else if (round_mode == "floor") {
            ttnn::floor(queue_id, output_tensor.value(), output_mem_config, output_tensor);
        }

        if (accurate_mode == false) {  // If input_b is non-zero tensor
            return output_tensor.value();
        }

        float t_nan = std::nanf("");
        float t_inf = std::numeric_limits<float>::infinity();
        return ttnn::where(
            queue_id,
            ttnn::eqz(queue_id, input_b, output_mem_config),
            ttnn::where(
                queue_id,
                ttnn::eqz(queue_id, input_a, output_mem_config),
                t_nan,
                ttnn::multiply(
                    queue_id, ttnn::sign(input_a, output_mem_config), t_inf, std::nullopt, output_mem_config)),
            output_tensor.value(),
            output_mem_config,
            output_tensor);
    }
}

Tensor ExecuteDiv::invoke(
    const Tensor& input_a,
    const Tensor& input_b,
    bool accurate_mode,
    const std::optional<std::string>& round_mode,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return ExecuteDiv::invoke(
        DefaultQueueId, input_a, input_b, accurate_mode, round_mode, output_mem_config, std::move(output_tensor));
}

Tensor _div_no_nan_overload(const Tensor& input_a, float value, const std::optional<MemoryConfig>& output_mem_config) {
    if (value == 0) {
        return ttnn::zeros_like(input_a);
    } else {
        return ttnn::multiply(input_a, (1.0f / value));
    }
}

Tensor _div_no_nan(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    if (input_a.get_dtype() == DataType::FLOAT32 && input_b.get_dtype() == DataType::FLOAT32) {
        // Not using SFPU div op here since inf/nan handling is not required
        Tensor div_result = ttnn::multiply(input_a, ttnn::reciprocal(input_b), std::nullopt, output_mem_config);
        return ttnn::where(ttnn::eqz(input_b, output_mem_config), 0.0f, div_result);
    } else {
        Tensor div_result = ttnn::divide(input_a, input_b, std::nullopt, output_mem_config);
        return ttnn::where(ttnn::eqz(input_b, output_mem_config), 0.0f, div_result);
    }
}

Tensor ExecutePrelu::invoke(const Tensor& input, float weight, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::prelu_sfpu(input, weight);
}

Tensor ExecutePrelu::invoke(
    const Tensor& input, const std::array<float, 1>& weight, const std::optional<MemoryConfig>& output_mem_config) {
    float scalar_weight = weight[0];
    return ttnn::prelu_sfpu(input, scalar_weight);
}

Tensor ExecutePrelu::invoke(
    const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    const auto s_a = input_a.get_logical_shape();
    const auto volume = input_b.get_logical_volume();
    TT_FATAL(
        s_a[1] == volume,
        "Mismatch of parameter numbers and input channel size. Found parameter numbers = {} and channel size = {}.",
        volume,
        s_a[1]);
    Tensor b = input_b;
    if (s_a.rank() > 2) {
        SmallVector<uint32_t> reshape(s_a.rank(), 1);
        reshape[1] = s_a[1];
        b = ttnn::reshape(input_b, ttnn::SimpleShape(reshape));
    }

    Tensor result = ttnn::where(ttnn::ltz(input_a, output_mem_config), ttnn::multiply(input_a, b), input_a);
    return result;
}
// Binary remainder will be overloaded by unary remainder in another PR
Tensor ExecuteBinaryRemainder::invoke(
    const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    auto arch = input_a.device()->arch();
    TT_FATAL(arch != tt::ARCH::GRAYSKULL, "Op is supported on Wormhole or Blackhole");
    DataType input_dtype = input_a.get_dtype();
    // No typecast for FP32 input
    if (input_dtype == DataType::FLOAT32 && input_b.get_dtype() == DataType::FLOAT32) {
        Tensor result = ttnn::subtract(
            input_a,
            ttnn::multiply(
                input_b,
                ttnn::div(input_a, input_b, true, "floor", output_mem_config),
                std::nullopt,
                output_mem_config),
            std::nullopt,
            output_mem_config);
        result = ttnn::where(ttnn::ge(result, input_b), ttnn::subtract(result, input_b), result);
        result = ttnn::where(ttnn::ltz(input_b), ttnn::add(result, input_b), result);
        result = ttnn::where(ttnn::eq(input_a, input_b, std::nullopt, output_mem_config), 0.0f, result);
        return result;
    }
    Tensor a = typecast(input_a, DataType::FLOAT32);
    Tensor b = typecast(input_b, DataType::FLOAT32);
    Tensor result = ttnn::subtract(
        a,
        ttnn::multiply(
            b, ttnn::div(input_a, input_b, false, "floor", output_mem_config), std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    result = ttnn::where(ttnn::ge(result, b), ttnn::subtract(result, b), result);
    result = ttnn::where(ttnn::ltz(b), ttnn::add(result, b), result);
    result = ttnn::where(ttnn::eq(input_a, input_b, std::nullopt, output_mem_config), 0.0f, result);
    return typecast(result, input_dtype);
}

Tensor ExecuteBinaryRemainder::invoke(
    const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::unary_remainder(input, scalar);
}

// FMOD result = input − (other * trunc(input/other))
// Binary FMOD will be overloaded by unary FMOD in another PR
Tensor ExecuteBinaryFmod::invoke(
    const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    auto arch = input_a.device()->arch();
    TT_FATAL(
        arch == tt::ARCH::WORMHOLE_B0 or arch == tt::ARCH::BLACKHOLE, "Op is only supported on Wormhole or Blackhole");

    DataType input_dtype = input_a.get_dtype();
    // No typecast for FP32 input
    if (input_dtype == DataType::FLOAT32 && input_b.get_dtype() == DataType::FLOAT32) {
        Tensor div_res = ttnn::divide(input_a, input_b, std::nullopt, output_mem_config);
        div_res = ttnn::trunc(div_res, output_mem_config);
        Tensor result = ttnn::subtract(
            input_a,
            ttnn::multiply(div_res, input_b, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config);
        result = ttnn::where(ttnn::eq(input_a, input_b, std::nullopt, output_mem_config), 0.0f, result);
        return result;
    }
    // For bfloat16 with decimal values, need to typecast to FP32 to improve precision
    Tensor a = typecast(input_a, DataType::FLOAT32);
    Tensor b = typecast(input_b, DataType::FLOAT32);

    Tensor div_res = typecast(ttnn::div(input_a, input_b, true, "trunc", output_mem_config), DataType::FLOAT32);
    Tensor result =
        ttnn::subtract(a, ttnn::multiply(div_res, b, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    result = ttnn::where(ttnn::eq(input_a, input_b, std::nullopt, output_mem_config), 0.0f, result);
    return typecast(result, input_dtype);
}

Tensor ExecuteBinaryFmod::invoke(
    const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::unary_fmod(input, scalar);
}

Tensor _floor_div_overload(const Tensor& input_a, float value, const std::optional<MemoryConfig>& output_mem_config) {
    auto arch = input_a.device()->arch();
    TT_FATAL(arch != tt::ARCH::GRAYSKULL, "Op is supported on Wormhole");
    if (value == 0) {
        float t_inf = std::numeric_limits<float>::infinity();
        float t_nan = std::nanf("");
        return ttnn::where(
            ttnn::eqz(input_a, output_mem_config),
            t_nan,
            ttnn::multiply(ttnn::sign(input_a, output_mem_config), t_inf, std::nullopt, output_mem_config));
    }
    Tensor temp = ttnn::multiply(input_a, (1.0f / value), std::nullopt, output_mem_config);
    return ttnn::floor(temp);
}

Tensor _floor_div(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    auto arch = input_a.device()->arch();
    TT_FATAL(arch != tt::ARCH::GRAYSKULL, "Op is supported on Wormhole");
    Tensor temp = ttnn::div(input_a, input_b, true, std::nullopt, output_mem_config);
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

Tensor _scatter(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    tt::tt_metal::Array4D start_index = {0, 0, 0, 0};
    Tensor index_pad = ttnn::pad(
        0, ttnn::ones_like(input_a), input_b.get_legacy_shape().to_array_4D(), start_index, 0, false, std::nullopt);
    Tensor temp_a =
        ttnn::pad(0, input_a, input_b.get_legacy_shape().to_array_4D(), start_index, 0, false, std::nullopt);
    return ttnn::where(index_pad, temp_a, input_b);
}

/**
 * outer product = matrix multiply when a = [1,1,N,1] and b = [1,1,1,M]
 * and result is of size [1,1,N,M].
 * - implementation supports any 1D "squeezable tensor" at input operands
 *   by running reshape.
 */
Tensor _outer(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    const ttnn::SimpleShape s_a = input_a.padded_shape();
    const ttnn::SimpleShape s_b = input_b.padded_shape();
    auto num_ones = [](const ttnn::SimpleShape& s) -> uint32_t {
        uint32_t num1s = 0;
        for (uint32_t idx = 0; idx < 4; idx++) {
            num1s += (uint32_t)(s[idx] == 1);
        }
        return num1s;
    };

    // check if 3 dimensions are 1
    TT_FATAL((num_ones(s_a) >= 3), "3 dimensions are required to be 1 for use with outer product");
    TT_FATAL((num_ones(s_b) >= 3), "3 dimensions are required to be 1 for use with outer product");

    const bool skip_reshape_a = (s_a[0] == 1 && s_a[1] == 1 && s_a[2] >= 1 && s_a[3] == 1);
    const bool skip_reshape_b = (s_b[0] == 1 && s_b[1] == 1 && s_b[2] == 1 && s_b[3] >= 1);

    Tensor a_slim = input_a;
    Tensor b_slim = input_b;

    if (!skip_reshape_a) {
        a_slim = ttnn::reshape(input_a, ttnn::SimpleShape{std::array<uint32_t, 4>{1, 1, input_a.volume(), 1}});
    }
    if (!skip_reshape_b) {
        b_slim = ttnn::reshape(input_b, ttnn::SimpleShape{std::array<uint32_t, 4>{1, 1, 1, input_b.volume()}});
    }
    a_slim = ttnn::to_layout(a_slim, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, (IDevice*)nullptr);
    b_slim = ttnn::to_layout(b_slim, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, (IDevice*)nullptr);

    auto device = ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice();
    if (device != nullptr) {
        if (a_slim.storage_type() != tt::tt_metal::StorageType::DEVICE) {
            a_slim = ttnn::operations::experimental::auto_format::AutoFormat::move_tensor_to_device(a_slim, device);
        }
        if (b_slim.storage_type() != tt::tt_metal::StorageType::DEVICE) {
            b_slim = ttnn::operations::experimental::auto_format::AutoFormat::move_tensor_to_device(b_slim, device);
        }
    }

    return ttnn::matmul(a_slim, b_slim);
}

Tensor _polyval(
    const Tensor& input_a, const std::vector<float>& coeffs, const std::optional<MemoryConfig>& output_mem_config) {
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

Tensor ExecuteGCD::invoke(
    const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor input_a_abs = ttnn::abs(input_a);
    Tensor input_b_abs = ttnn::abs(input_b);
    Tensor a_gt_b = ttnn::gt(input_a_abs, input_b_abs);
    Tensor min = ttnn::where(a_gt_b, input_b_abs, input_a_abs);
    Tensor max = ttnn::where(a_gt_b, input_a_abs, input_b_abs);
    a_gt_b.deallocate();
    // https://en.wikipedia.org/wiki/Lam%C3%A9%27s_theorem
    // While 186 is the theoretical maximum iterations for numbers within the floating point range according to Lame's
    // theorem, in practice when evaluating gcd of consecutive Fibonacci numbers coerced to floating point, the
    // maximum number of iterations reached is only 14 because the remainder converges to 0 much more quickly. In
    // addition, limited precision in bfloat16 format decreases support for input to the range [-1024, 1024]
    constexpr std::size_t max_iterations = 14;
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
        Tensor isz = ttnn::eqz(min);
        //  0's in min are replaced with 1
        Tensor rem = ttnn::remainder(max, ttnn::where(isz, isz, min));
        max = ttnn::where(isz, max, min);
        min = rem;
    }
    return max;
}

Tensor ExecuteLCM::invoke(
    const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor val = ttnn::multiply(input_a, input_b, std::nullopt, output_mem_config);
    Tensor tmp_result = ttnn::gcd(input_a, input_b);
    Tensor result = ttnn::divide(val, tmp_result, std::nullopt, output_mem_config);
    result = ttnn::abs(result);
    return result;
}

// power - floating point exponent
Tensor ExecutePower::invoke(
    uint8_t queue_id,
    const Tensor& input_a,
    float exponent,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    TT_FATAL(exponent >= 0.0f, "works for positive exponents only");
    const uint32_t exponent_floor = static_cast<uint32_t>(std::floor(exponent));
    if (static_cast<float>(exponent_floor) == exponent) {
        if (output_tensor.has_value()) {
            ttnn::power(queue_id, input_a, exponent_floor, output_mem_config, output_tensor);
            return output_tensor.value();
        }
        return ttnn::power(queue_id, input_a, exponent_floor, output_mem_config);
    }
    const float exponent_trunc = exponent - static_cast<float>(exponent_floor);
    Tensor pow_trunc_log = ttnn::multiply(
        queue_id, ttnn::log(queue_id, input_a, output_mem_config), exponent_trunc, std::nullopt, output_mem_config);
    Tensor pow_frac = ttnn::exp(queue_id, pow_trunc_log, false, output_mem_config);
    pow_trunc_log.deallocate();
    float t_nan = std::nanf("");
    Tensor result = ttnn::multiply(
        queue_id,
        ttnn::power(queue_id, input_a, exponent_floor, output_mem_config),
        pow_frac,
        std::nullopt,
        output_mem_config);
    // To handle negative inputs:
    // in torch For -ve inputs with float exponent power returns nan
    auto output_memory_config = output_tensor.has_value() ? output_tensor.value().memory_config()
                                                          : output_mem_config.value_or(input_a.memory_config());
    result = ttnn::where(
        ttnn::ltz(queue_id, input_a, output_mem_config), t_nan, result, output_memory_config, output_tensor);
    return result;
}

// power - floating point exponent
Tensor ExecutePower::invoke(
    const Tensor& input_a,
    float exponent,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    return ExecutePower::invoke(DefaultQueueId, input_a, exponent, output_mem_config, std::move(output_tensor));
}

// power - integer exponent
Tensor ExecutePower::invoke(
    uint8_t queue_id,
    const Tensor& input,
    uint32_t exponent,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    return ttnn::power(queue_id, input, exponent, output_mem_config, output_tensor);
}

// power - integer exponent
Tensor ExecutePower::invoke(
    const Tensor& input,
    uint32_t exponent,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    return ExecutePower::invoke(DefaultQueueId, input, exponent, output_mem_config, std::move(output_tensor));
}

// power - tensor exponent
Tensor ExecutePower::invoke(
    uint8_t queue_id,
    const Tensor& input,
    const Tensor& exponent,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    return BinaryOperationSfpu<operations::binary::BinaryOpType::POWER>::invoke(
        queue_id, input, exponent, std::nullopt, output_mem_config, output_tensor);
}

// power - tensor exponent
Tensor ExecutePower::invoke(
    const Tensor& input,
    const Tensor& exponent,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    return ExecutePower::invoke(DefaultQueueId, input, exponent, output_mem_config, std::move(output_tensor));
}

// power - scalar input
Tensor ExecutePower::invoke(
    uint8_t queue_id,
    float input_a,
    const Tensor& exponent,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    Tensor input = ttnn::full_like(exponent, input_a);
    return ExecutePower::invoke(queue_id, input, exponent, output_mem_config, std::move(output_tensor));
}

// power - scalar input
Tensor ExecutePower::invoke(
    float input_a,
    const Tensor& exponent,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    return ExecutePower::invoke(DefaultQueueId, input_a, exponent, output_mem_config, std::move(output_tensor));
}

Tensor ExecuteRsub::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<unary::FusedActivations>& activations,
    const std::optional<unary::UnaryWithParam>& input_tensor_a_activation) {
    return BinaryOperation<operations::binary::BinaryOpType::RSUB>::invoke(
        queue_id,
        input_tensor_a,
        input_tensor_b,
        output_dtype,
        memory_config,
        optional_output_tensor,
        activations,
        input_tensor_a_activation);
}

Tensor ExecuteRsub::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<unary::FusedActivations>& activations,
    const std::optional<unary::UnaryWithParam>& input_tensor_a_activation) {

    return ExecuteRsub::invoke(
        ttnn::DefaultQueueId,
        input_tensor_a,
        input_tensor_b,
        output_dtype,
        memory_config,
        optional_output_tensor,
        activations,
        input_tensor_a_activation);
}

Tensor ExecuteRsub::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const float input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::operations::unary::ExecuteUnaryWithFloatParameter<ttnn::operations::unary::UnaryOpType::RSUB>::invoke(
        queue_id, input_tensor_a, input_b, memory_config, optional_output_tensor);
}

Tensor ExecuteRsub::invoke(
    const Tensor& input_tensor_a,
    const float input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {

    return ExecuteRsub::invoke(
        ttnn::DefaultQueueId,
        input_tensor_a,
        input_b,
        memory_config,
        std::move(optional_output_tensor));
}

// Bitwise AND
Tensor ExecuteBitwiseAnd::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return BinaryOperationSfpu<operations::binary::BinaryOpType::BITWISE_AND>::invoke(
        queue_id, input_tensor_a, input_tensor_b, std::nullopt, memory_config, optional_output_tensor);
}

Tensor ExecuteBitwiseAnd::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {

    return ExecuteBitwiseAnd::invoke(
        ttnn::DefaultQueueId,
        input_tensor_a,
        input_tensor_b,
        memory_config,
        optional_output_tensor);
}

Tensor ExecuteBitwiseAnd::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const int32_t input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::operations::unary::
        ExecuteUnaryWithIntegerParameter<ttnn::operations::unary::UnaryOpType::BITWISE_AND, int32_t>::invoke(
            queue_id, input_tensor_a, input_b, memory_config, optional_output_tensor);
}

Tensor ExecuteBitwiseAnd::invoke(
    const Tensor& input_tensor_a,
    const int32_t input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {

    return ExecuteBitwiseAnd::invoke(
        ttnn::DefaultQueueId,
        input_tensor_a,
        input_b,
        memory_config,
        std::move(optional_output_tensor));
}

// Bitwise OR
Tensor ExecuteBitwiseOr::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return BinaryOperationSfpu<operations::binary::BinaryOpType::BITWISE_OR>::invoke(
        queue_id, input_tensor_a, input_tensor_b, std::nullopt, memory_config, optional_output_tensor);
}

Tensor ExecuteBitwiseOr::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {

    return ExecuteBitwiseOr::invoke(
        ttnn::DefaultQueueId,
        input_tensor_a,
        input_tensor_b,
        memory_config,
        optional_output_tensor);
}

Tensor ExecuteBitwiseOr::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const int32_t input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::operations::unary::
        ExecuteUnaryWithIntegerParameter<ttnn::operations::unary::UnaryOpType::BITWISE_OR, int32_t>::invoke(
            queue_id, input_tensor_a, input_b, memory_config, optional_output_tensor);
}

Tensor ExecuteBitwiseOr::invoke(
    const Tensor& input_tensor_a,
    const int32_t input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {

    return ExecuteBitwiseOr::invoke(
        ttnn::DefaultQueueId,
        input_tensor_a,
        input_b,
        memory_config,
        std::move(optional_output_tensor));
}

// Bitwise XOR
Tensor ExecuteBitwiseXor::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return BinaryOperationSfpu<operations::binary::BinaryOpType::BITWISE_XOR>::invoke(
        queue_id, input_tensor_a, input_tensor_b, std::nullopt, memory_config, optional_output_tensor);
}

Tensor ExecuteBitwiseXor::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {

    return ExecuteBitwiseXor::invoke(
        ttnn::DefaultQueueId,
        input_tensor_a,
        input_tensor_b,
        memory_config,
        optional_output_tensor);
}

Tensor ExecuteBitwiseXor::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const int32_t input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::operations::unary::
        ExecuteUnaryWithIntegerParameter<ttnn::operations::unary::UnaryOpType::BITWISE_XOR, int32_t>::invoke(
            queue_id, input_tensor_a, input_b, memory_config, optional_output_tensor);
}

Tensor ExecuteBitwiseXor::invoke(
    const Tensor& input_tensor_a,
    const int32_t input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {

    return ExecuteBitwiseXor::invoke(
        ttnn::DefaultQueueId,
        input_tensor_a,
        input_b,
        memory_config,
        std::move(optional_output_tensor));
}

// Bitwise Left Shift
Tensor ExecuteBitwiseLeftShift::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return BinaryOperationSfpu<operations::binary::BinaryOpType::LEFT_SHIFT>::invoke(
        queue_id, input_tensor_a, input_tensor_b, std::nullopt, memory_config, optional_output_tensor);
}

Tensor ExecuteBitwiseLeftShift::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ExecuteBitwiseLeftShift::invoke(
        ttnn::DefaultQueueId, input_tensor_a, input_tensor_b, memory_config, optional_output_tensor);
}

Tensor ExecuteBitwiseLeftShift::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const int32_t input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::operations::unary::
        ExecuteUnaryWithIntegerParameter<ttnn::operations::unary::UnaryOpType::LEFT_SHIFT, int32_t>::invoke(
            queue_id, input_tensor_a, input_b, memory_config, optional_output_tensor);
}

Tensor ExecuteBitwiseLeftShift::invoke(
    const Tensor& input_tensor_a,
    const int32_t input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ExecuteBitwiseLeftShift::invoke(
        ttnn::DefaultQueueId, input_tensor_a, input_b, memory_config, std::move(optional_output_tensor));
}

// Bitwise Right Shift
Tensor ExecuteBitwiseRightShift::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return BinaryOperationSfpu<operations::binary::BinaryOpType::RIGHT_SHIFT>::invoke(
        queue_id, input_tensor_a, input_tensor_b, std::nullopt, memory_config, optional_output_tensor);
}

Tensor ExecuteBitwiseRightShift::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ExecuteBitwiseRightShift::invoke(
        ttnn::DefaultQueueId, input_tensor_a, input_tensor_b, memory_config, optional_output_tensor);
}

Tensor ExecuteBitwiseRightShift::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const int32_t input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::operations::unary::
        ExecuteUnaryWithIntegerParameter<ttnn::operations::unary::UnaryOpType::RIGHT_SHIFT, int32_t>::invoke(
            queue_id, input_tensor_a, input_b, memory_config, optional_output_tensor);
}

Tensor ExecuteBitwiseRightShift::invoke(
    const Tensor& input_tensor_a,
    const int32_t input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ExecuteBitwiseRightShift::invoke(
        ttnn::DefaultQueueId, input_tensor_a, input_b, memory_config, std::move(optional_output_tensor));
}

}  // namespace ttnn::operations::binary
