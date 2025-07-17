// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_composite_op.hpp"
#include <enchantum/enchantum.hpp>
#include <utility>
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/where.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include <variant>

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

// nextafter
Tensor _nextafter(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    const float eps = tt::tt_metal::hal::get_eps();
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
    Tensor result = ttnn::where(ttnn::le(is_close_lhs, is_close_rhs, std::nullopt, output_mem_config), 1.f, 0.f);
    return result;
}

Tensor ExecuteMinimum::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    return BinaryOperationSfpu<operations::binary::BinaryOpType::MINIMUM>::invoke(
        queue_id,
        input_tensor_a,
        input_tensor_b,
        std::nullopt,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

Tensor ExecuteMinimum::invoke(
    QueueId queue_id,
    const Tensor& input_a,
    const std::variant<int32_t, float> value,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    return std::visit(
        [&](auto input_b) {
            return ttnn::operations::unary::
                ExecuteUnaryWithVariantFloatIntParameter<ttnn::operations::unary::UnaryOpType::MINIMUM>::invoke(
                    queue_id, input_a, input_b, memory_config, optional_output_tensor);
        },
        value);
}

Tensor ExecuteMaximum::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    return BinaryOperationSfpu<operations::binary::BinaryOpType::MAXIMUM>::invoke(
        queue_id,
        input_tensor_a,
        input_tensor_b,
        std::nullopt,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

Tensor ExecuteMaximum::invoke(
    QueueId queue_id,
    const Tensor& input_a,
    const std::variant<int32_t, float> value,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    return std::visit(
        [&](auto input_b) {
            return ttnn::operations::unary::
                ExecuteUnaryWithVariantFloatIntParameter<ttnn::operations::unary::UnaryOpType::MAXIMUM>::invoke(
                    queue_id, input_a, input_b, memory_config, optional_output_tensor);
        },
        value);
}

Tensor _atan2(const Tensor& input_b, const Tensor& input_a, const std::optional<MemoryConfig>& output_mem_config) {
    log_info(tt::LogOp, "Input arguments for the atan2 function are in the format (y, x)");
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
                    ttnn::where(az_bltz, -pi_2, ttnn::where(az_bgtz, pi_2, 0.f, output_mem_config), output_mem_config),
                    output_mem_config),
                output_mem_config),
            output_mem_config);
    }
    return res;
}

Tensor ExecuteDiv::invoke(
    QueueId queue_id,
    const Tensor& input,
    float value,
    bool accurate_mode,
    const std::optional<std::string>& round_mode,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy) {
    const auto has_legacy_only_args = round_mode.has_value() or accurate_mode;
    if (not(use_legacy ? *use_legacy
                       : has_legacy_only_args or
                             binary::is_legacy_only(
                                 input, value, output_mem_config, output_tensor, lhs_activations, rhs_activations))) {
        TT_FATAL(
            not has_legacy_only_args,
            "round_mode, accurate_mode are not valid when passing use_legacy parameter in div");
        return BinaryOperation<BinaryOpType::DIV>::invoke(
            queue_id,
            input,
            value,
            output_dtype,
            output_mem_config,
            output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            use_legacy);
    }

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
    QueueId queue_id,
    const Tensor& input_a,
    const Tensor& input_b,
    bool accurate_mode,
    const std::optional<std::string>& round_mode,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy) {
    const auto has_legacy_only_args = round_mode.has_value() or accurate_mode;
    if (not(use_legacy
                ? *use_legacy
                : has_legacy_only_args or
                      binary::is_legacy_only(
                          input_a, input_b, output_mem_config, output_tensor, lhs_activations, rhs_activations))) {
        TT_FATAL(
            not has_legacy_only_args,
            "round_mode, accurate_mode are not valid when passing use_legacy parameter in div");
        return BinaryOperation<BinaryOpType::DIV>::invoke(
            queue_id,
            input_a,
            input_b,
            std::nullopt,
            output_mem_config,
            output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            use_legacy);
    }

    TT_FATAL(
        (round_mode == std::nullopt || round_mode == "trunc" || round_mode == "floor"),
        "Incorrect rounding mode (expected None, 'trunc', or 'floor')");

    DataType input_dtype = input_a.dtype();
    const bool is_fp32 = input_dtype == DataType::FLOAT32 && input_b.dtype() == DataType::FLOAT32;
    Tensor result;

    // No accurate_mode for FP32 div as inf/nan are handled at kernel level
    if (is_fp32) {
        result = ttnn::divide(queue_id, input_a, input_b, std::nullopt, output_mem_config, output_tensor);
    } else {
        Tensor a = typecast(queue_id, input_a, DataType::FLOAT32);
        Tensor b = typecast(queue_id, input_b, DataType::FLOAT32);

        // Div operation without inf/nan handling as reciprocal(0) = 1.7014118346046923e+38 not inf/nan
        result = ttnn::multiply(
            queue_id,
            a,
            ttnn::reciprocal(queue_id, b, output_mem_config),
            std::nullopt,
            output_mem_config,
            output_tensor);
    }

    if (round_mode == "trunc") {
        result = ttnn::trunc(queue_id, result, output_mem_config, output_tensor);
    } else if (round_mode == "floor") {
        result = ttnn::floor(queue_id, result, output_mem_config, output_tensor);
    }

    if (is_fp32) {
        return result;
    }

    // Accurate mode: handle division by zero (inf/nan cases)
    if (accurate_mode) {
        float t_nan = std::nanf("");
        float t_inf = std::numeric_limits<float>::infinity();
        result = where(
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
            result);
    }

    return typecast(queue_id, result, input_dtype, std::nullopt, output_tensor);
}

Tensor _div_no_nan_overload(const Tensor& input_a, float value, const std::optional<MemoryConfig>& output_mem_config) {
    if (value == 0) {
        return ttnn::zeros_like(input_a);
    } else {
        return ttnn::multiply(input_a, (1.0f / value));
    }
}

Tensor _div_no_nan(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    if (input_a.dtype() == DataType::FLOAT32 && input_b.dtype() == DataType::FLOAT32) {
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
    const auto& s_a = input_a.logical_shape();
    const auto volume = input_b.logical_volume();
    TT_FATAL(
        s_a[1] == volume,
        "Mismatch of parameter numbers and input channel size. Found parameter numbers = {} and channel size = {}.",
        volume,
        s_a[1]);
    Tensor b = input_b;
    if (s_a.rank() > 2) {
        SmallVector<uint32_t> reshape(s_a.rank(), 1);
        reshape[1] = s_a[1];
        b = ttnn::reshape(input_b, ttnn::Shape(reshape));
    }

    Tensor result = ttnn::where(ttnn::ltz(input_a, output_mem_config), ttnn::multiply(input_a, b), input_a);
    return result;
}

Tensor run_remainder(
    const Tensor& input_a, const Tensor& input_b, float t_nan, const std::optional<MemoryConfig>& output_mem_config) {
    using FusedActivations = tt::stl::Span<const unary::UnaryWithParam>;
    // explicitly using binary_ng to avoid fallback to legacy because of row boradcast
    Tensor result = ttnn::subtract(
        input_a,
        ttnn::multiply(
            input_b,
            ttnn::div(input_a, input_b, true, "floor", std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        std::nullopt,
        output_mem_config,
        std::nullopt,
        FusedActivations{},
        FusedActivations{},
        FusedActivations{},
        false);
    result = ttnn::where(ttnn::ge(result, input_b), ttnn::subtract(result, input_b), result);
    result = ttnn::where(ttnn::ltz(input_b), ttnn::add(result, input_b), result);
    result = ttnn::where(ttnn::eq(input_a, input_b, std::nullopt, output_mem_config), 0.0f, result);
    result = ttnn::where(ttnn::eqz(input_a), 0.0f, ttnn::where(ttnn::eqz(input_b), t_nan, result));
    result = ttnn::where(ttnn::logical_and(ttnn::eqz(input_a), ttnn::eqz(input_b)), t_nan, result);
    return result;
}
// Binary remainder will be overloaded by unary remainder in another PR
Tensor ExecuteBinaryRemainder::invoke(
    const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    DataType input_dtype = input_a.dtype();

    float t_nan = tt::tt_metal::hal::get_nan();

    // No typecast for FP32 input
    const auto do_typecast = input_dtype != DataType::FLOAT32 or input_b.dtype() != DataType::FLOAT32;
    const auto& a = do_typecast ? typecast(input_a, DataType::FLOAT32) : input_a;
    const auto& b = do_typecast ? typecast(input_b, DataType::FLOAT32) : input_b;

    // Perform the remainder operation
    Tensor result = run_remainder(a, b, t_nan, output_mem_config);

    // Return the result, typecasted if necessary
    return do_typecast ? typecast(result, input_dtype) : result;
}

Tensor ExecuteBinaryRemainder::invoke(
    const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::unary_remainder(input, scalar);
}

Tensor run_fmod(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& division_result,
    const std::optional<MemoryConfig>& output_mem_config) {
    Tensor result = ttnn::subtract(
        input_a,
        ttnn::multiply(division_result, input_b, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    return ttnn::where(ttnn::eq(input_a, input_b, std::nullopt, output_mem_config), 0.0f, result);
}

// FMOD result = input − (other * trunc(input/other))
// When inputs are of data type BF16 and when input_b==0, expected is nan, but FMOD gives inf
Tensor ExecuteBinaryFmod::invoke(
    const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    DataType input_dtype = input_a.dtype();
    Tensor div_res = ttnn::divide(input_a, input_b, std::nullopt, output_mem_config);
    div_res = ttnn::trunc(div_res, output_mem_config);
    // No typecast for FP32 input
    if (input_dtype == DataType::FLOAT32 && input_b.dtype() == DataType::FLOAT32) {
        return run_fmod(input_a, input_b, div_res, output_mem_config);
    }
    // For bfloat16 with decimal values, need to typecast to FP32 to improve precision
    Tensor a = typecast(input_a, DataType::FLOAT32);
    Tensor b = typecast(input_b, DataType::FLOAT32);
    div_res = typecast(div_res, DataType::FLOAT32);
    return typecast(run_fmod(a, b, div_res, output_mem_config), input_dtype);
}

Tensor ExecuteBinaryFmod::invoke(
    const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::operations::unary::ExecuteUnaryWithFloatParameter<ttnn::operations::unary::UnaryOpType::FMOD>::invoke(
        ttnn::DefaultQueueId, input, scalar, output_mem_config);
}

Tensor _floor_div_overload(const Tensor& input_a, float value, const std::optional<MemoryConfig>& output_mem_config) {
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
    Tensor temp = ttnn::div(input_a, input_b, true, std::nullopt, std::nullopt, output_mem_config);
    Tensor result = ttnn::div(input_a, input_b, true, "floor", std::nullopt, output_mem_config);
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
        ttnn::DefaultQueueId,
        ttnn::ones_like(input_a),
        input_b.padded_shape().to_array_4D(),
        start_index,
        0,
        false,
        std::nullopt);
    Tensor temp_a = ttnn::pad(
        ttnn::DefaultQueueId, input_a, input_b.padded_shape().to_array_4D(), start_index, 0, false, std::nullopt);
    return ttnn::where(index_pad, temp_a, input_b);
}

/**
 * outer product = matrix multiply when a = [1,1,N,1] and b = [1,1,1,M]
 * and result is of size [1,1,N,M].
 * - implementation supports any 1D "squeezable tensor" at input operands
 *   by running reshape.
 */
Tensor _outer(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config) {
    const ttnn::Shape& s_a = input_a.logical_shape();
    const ttnn::Shape& s_b = input_b.logical_shape();
    auto num_ones = [](const ttnn::Shape& s) -> uint32_t {
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
        uint32_t a_volume = s_a[0] * s_a[1] * s_a[2] * s_a[3];
        a_slim = ttnn::reshape(input_a, ttnn::Shape{std::array<uint32_t, 4>{1, 1, a_volume, 1}});
    }
    if (!skip_reshape_b) {
        uint32_t b_volume = s_b[0] * s_b[1] * s_b[2] * s_b[3];
        b_slim = ttnn::reshape(input_b, ttnn::Shape{std::array<uint32_t, 4>{1, 1, 1, b_volume}});
    }
    a_slim = ttnn::to_layout(a_slim, ttnn::TILE_LAYOUT);
    b_slim = ttnn::to_layout(b_slim, ttnn::TILE_LAYOUT);

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
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    return BinaryOperationSfpu<operations::binary::BinaryOpType::GCD>::invoke(
        queue_id,
        input_tensor_a,
        input_tensor_b,
        std::nullopt,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

Tensor ExecuteLCM::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    return BinaryOperationSfpu<operations::binary::BinaryOpType::LCM>::invoke(
        queue_id,
        input_tensor_a,
        input_tensor_b,
        std::nullopt,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

// power - floating point exponent
Tensor ExecutePower::invoke(
    QueueId queue_id,
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

// power - integer exponent
Tensor ExecutePower::invoke(
    QueueId queue_id,
    const Tensor& input,
    uint32_t exponent,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    return ttnn::power(queue_id, input, exponent, output_mem_config, output_tensor);
}

// power - tensor exponent
Tensor ExecutePower::invoke(
    QueueId queue_id,
    const Tensor& input,
    const Tensor& exponent,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    return BinaryOperationSfpu<operations::binary::BinaryOpType::POWER>::invoke(
        queue_id,
        input,
        exponent,
        std::nullopt,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

// power - scalar input, tensor exponent
Tensor ExecutePower::invoke(
    QueueId queue_id,
    float input_a,
    const Tensor& exponent,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    // As per binary infra, first input is always a tensor but this support needed for pytorch2 tracing
    // https://github.com/tenstorrent/pytorch2.0_ttnn/blob/main/docs/operations/aten.pow.Scalar.md

    Tensor input = ttnn::full_like(exponent, input_a);
    return ExecutePower::invoke(
        queue_id,
        input,
        exponent,
        std::nullopt,
        memory_config,
        std::move(optional_output_tensor),
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

Tensor ExecuteRsub::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    return BinaryOperation<operations::binary::BinaryOpType::RSUB>::invoke(
        queue_id,
        input_tensor_a,
        input_tensor_b,
        output_dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

Tensor ExecuteRsub::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const float input_b,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    if (not(use_legacy ? *use_legacy
                       : binary::is_legacy_only(
                             input_tensor_a,
                             input_b,
                             memory_config,
                             optional_output_tensor,
                             lhs_activations,
                             rhs_activations))) {
        return BinaryOperation<operations::binary::BinaryOpType::RSUB>::invoke(
            queue_id,
            input_tensor_a,
            input_b,
            output_dtype,
            memory_config,
            optional_output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            use_legacy);
    }

    return ttnn::operations::unary::ExecuteUnaryWithFloatParameter<ttnn::operations::unary::UnaryOpType::RSUB>::invoke(
        queue_id, input_tensor_a, input_b, memory_config, optional_output_tensor);
}

// Bitwise AND
Tensor ExecuteBitwiseAnd::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    if (not(use_legacy ? *use_legacy
                       : binary::is_legacy_only(
                             input_tensor_a,
                             input_tensor_b,
                             memory_config,
                             optional_output_tensor,
                             lhs_activations,
                             rhs_activations))) {
        return BinaryOperation<operations::binary::BinaryOpType::BITWISE_AND>::invoke(
            queue_id,
            input_tensor_a,
            input_tensor_b,
            std::nullopt,
            memory_config,
            optional_output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            use_legacy);
    }

    return BinaryOperationSfpu<operations::binary::BinaryOpType::BITWISE_AND>::invoke(
        queue_id,
        input_tensor_a,
        input_tensor_b,
        std::nullopt,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

Tensor ExecuteBitwiseAnd::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const int32_t input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    if (not(use_legacy ? *use_legacy
                       : binary::is_legacy_only(
                             input_tensor_a,
                             input_b,
                             memory_config,
                             optional_output_tensor,
                             lhs_activations,
                             rhs_activations))) {
        return BinaryOperation<operations::binary::BinaryOpType::BITWISE_AND>::invoke(
            queue_id,
            input_tensor_a,
            input_b,
            std::nullopt,
            memory_config,
            optional_output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            use_legacy);
    }

    return ttnn::operations::unary::
        ExecuteUnaryWithIntegerParameter<ttnn::operations::unary::UnaryOpType::BITWISE_AND, int32_t>::invoke(
            queue_id, input_tensor_a, input_b, memory_config, optional_output_tensor);
}

// Bitwise OR
Tensor ExecuteBitwiseOr::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    if (not(use_legacy ? *use_legacy
                       : binary::is_legacy_only(
                             input_tensor_a,
                             input_tensor_b,
                             memory_config,
                             optional_output_tensor,
                             lhs_activations,
                             rhs_activations))) {
        return BinaryOperation<operations::binary::BinaryOpType::BITWISE_OR>::invoke(
            queue_id,
            input_tensor_a,
            input_tensor_b,
            std::nullopt,
            memory_config,
            optional_output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            use_legacy);
    }

    return BinaryOperationSfpu<operations::binary::BinaryOpType::BITWISE_OR>::invoke(
        queue_id, input_tensor_a, input_tensor_b, std::nullopt, memory_config, optional_output_tensor);
}

Tensor ExecuteBitwiseOr::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const int32_t input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    if (not(use_legacy ? *use_legacy
                       : binary::is_legacy_only(
                             input_tensor_a,
                             input_b,
                             memory_config,
                             optional_output_tensor,
                             lhs_activations,
                             rhs_activations))) {
        return BinaryOperation<operations::binary::BinaryOpType::BITWISE_OR>::invoke(
            queue_id,
            input_tensor_a,
            input_b,
            std::nullopt,
            memory_config,
            optional_output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            use_legacy);
    }

    return ttnn::operations::unary::
        ExecuteUnaryWithIntegerParameter<ttnn::operations::unary::UnaryOpType::BITWISE_OR, int32_t>::invoke(
            queue_id, input_tensor_a, input_b, memory_config, optional_output_tensor);
}

// Bitwise XOR
Tensor ExecuteBitwiseXor::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    if (not(use_legacy ? *use_legacy
                       : binary::is_legacy_only(
                             input_tensor_a,
                             input_tensor_b,
                             memory_config,
                             optional_output_tensor,
                             lhs_activations,
                             rhs_activations))) {
        return BinaryOperation<operations::binary::BinaryOpType::BITWISE_XOR>::invoke(
            queue_id,
            input_tensor_a,
            input_tensor_b,
            std::nullopt,
            memory_config,
            optional_output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            use_legacy);
    }

    return BinaryOperationSfpu<operations::binary::BinaryOpType::BITWISE_XOR>::invoke(
        queue_id, input_tensor_a, input_tensor_b, std::nullopt, memory_config, optional_output_tensor);
}

Tensor ExecuteBitwiseXor::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const int32_t input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    if (not(use_legacy ? *use_legacy
                       : binary::is_legacy_only(
                             input_tensor_a,
                             input_b,
                             memory_config,
                             optional_output_tensor,
                             lhs_activations,
                             rhs_activations))) {
        return BinaryOperation<operations::binary::BinaryOpType::BITWISE_XOR>::invoke(
            queue_id,
            input_tensor_a,
            input_b,
            std::nullopt,
            memory_config,
            optional_output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            use_legacy);
    }

    return ttnn::operations::unary::
        ExecuteUnaryWithIntegerParameter<ttnn::operations::unary::UnaryOpType::BITWISE_XOR, int32_t>::invoke(
            queue_id, input_tensor_a, input_b, memory_config, optional_output_tensor);
}

// Bitwise Left Shift
Tensor ExecuteBitwiseLeftShift::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    if (not(use_legacy ? *use_legacy
                       : binary::is_legacy_only(
                             input_tensor_a,
                             input_tensor_b,
                             memory_config,
                             optional_output_tensor,
                             lhs_activations,
                             rhs_activations))) {
        return BinaryOperation<operations::binary::BinaryOpType::LEFT_SHIFT>::invoke(
            queue_id,
            input_tensor_a,
            input_tensor_b,
            std::nullopt,
            memory_config,
            optional_output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            use_legacy);
    }

    return BinaryOperationSfpu<operations::binary::BinaryOpType::LEFT_SHIFT>::invoke(
        queue_id, input_tensor_a, input_tensor_b, std::nullopt, memory_config, optional_output_tensor);
}

Tensor ExecuteBitwiseLeftShift::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const int32_t input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    if (not(use_legacy ? *use_legacy
                       : binary::is_legacy_only(
                             input_tensor_a,
                             input_b,
                             memory_config,
                             optional_output_tensor,
                             lhs_activations,
                             rhs_activations))) {
        return BinaryOperation<operations::binary::BinaryOpType::LEFT_SHIFT>::invoke(
            queue_id,
            input_tensor_a,
            input_b,
            std::nullopt,
            memory_config,
            optional_output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            use_legacy);
    }

    return ttnn::operations::unary::
        ExecuteUnaryWithIntegerParameter<ttnn::operations::unary::UnaryOpType::LEFT_SHIFT, int32_t>::invoke(
            queue_id, input_tensor_a, input_b, memory_config, optional_output_tensor);
}

// Bitwise Right Shift
Tensor ExecuteBitwiseRightShift::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    if (not(use_legacy ? *use_legacy
                       : binary::is_legacy_only(
                             input_tensor_a,
                             input_tensor_b,
                             memory_config,
                             optional_output_tensor,
                             lhs_activations,
                             rhs_activations))) {
        return BinaryOperation<operations::binary::BinaryOpType::RIGHT_SHIFT>::invoke(
            queue_id,
            input_tensor_a,
            input_tensor_b,
            std::nullopt,
            memory_config,
            optional_output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            use_legacy);
    }

    return BinaryOperationSfpu<operations::binary::BinaryOpType::RIGHT_SHIFT>::invoke(
        queue_id, input_tensor_a, input_tensor_b, std::nullopt, memory_config, optional_output_tensor);
}

Tensor ExecuteBitwiseRightShift::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const int32_t input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    tt::stl::Span<const unary::UnaryWithParam> post_activations,
    tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    if (not(use_legacy ? *use_legacy
                       : binary::is_legacy_only(
                             input_tensor_a,
                             input_b,
                             memory_config,
                             optional_output_tensor,
                             lhs_activations,
                             rhs_activations))) {
        return BinaryOperation<operations::binary::BinaryOpType::RIGHT_SHIFT>::invoke(
            queue_id,
            input_tensor_a,
            input_b,
            std::nullopt,
            memory_config,
            optional_output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            use_legacy);
    }

    return ttnn::operations::unary::
        ExecuteUnaryWithIntegerParameter<ttnn::operations::unary::UnaryOpType::RIGHT_SHIFT, int32_t>::invoke(
            queue_id, input_tensor_a, input_b, memory_config, optional_output_tensor);
}

}  // namespace ttnn::operations::binary
