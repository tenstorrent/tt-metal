// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_composite_op.hpp"
namespace ttnn {
namespace operations {
namespace unary {

namespace detail {

// TODO: decide on a structure for composite ops
Tensor _power_fp(uint8_t queue_id, const Tensor& input_a, float exponent, const std::optional<MemoryConfig>& output_mem_config, const std::optional<Tensor>& output_tensor) {
    TT_FATAL(exponent >= 0.0f, "works for positive exponents only");
    const uint32_t exponent_floor = static_cast<uint32_t>(std::floor(exponent));
    if (static_cast<float>(exponent_floor) == exponent) {
        if(output_tensor.has_value()){
            ttnn::power(queue_id,input_a, exponent_floor, output_mem_config, output_tensor);
            return output_tensor.value();
        }
        return ttnn::power(queue_id, input_a, exponent_floor, output_mem_config);
    }
    const float exponent_trunc = exponent - static_cast<float>(exponent_floor);
    Tensor pow_trunc_log = ttnn::multiply(queue_id, ttnn::log(queue_id, input_a, output_mem_config), exponent_trunc, std::nullopt, output_mem_config);
    Tensor pow_frac = ttnn::exp(queue_id, pow_trunc_log, false, output_mem_config);
    pow_trunc_log.deallocate();
    float t_nan = std::nanf("");
    Tensor result = ttnn::multiply(queue_id, ttnn::power(queue_id, input_a, exponent_floor, output_mem_config), pow_frac, std::nullopt, output_mem_config);
    // To handle negative inputs:
    // in torch For -ve inputs with float exponent power returns nan
    auto output_memory_config = output_tensor.has_value() ? output_tensor.value().memory_config() : output_mem_config.value_or(input_a.memory_config());
    result = tt::tt_metal::where(ttnn::ltz(queue_id, input_a, output_mem_config), t_nan, result, output_memory_config, output_tensor);
    return result;
}

Tensor power_fp(
    uint8_t queue_id,
    const Tensor& input_a,
    float exponent,
    const std::optional<MemoryConfig>& output_mem_config  = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::operation::decorate_as_composite(__func__, _power_fp)(queue_id, input_a, exponent, output_mem_config, optional_output_tensor);
}

}

struct Power{

    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& input_tensor,
        uint32_t exponent,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return ttnn::power(queue_id, input_tensor, exponent, memory_config, optional_output_tensor);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        uint32_t exponent,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return ttnn::power(DefaultQueueId, input_tensor, exponent, memory_config, optional_output_tensor);
    }

    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& input_tensor,
        float exponent,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::power_fp(queue_id, input_tensor, exponent, memory_config, optional_output_tensor);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        float exponent,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::power_fp(DefaultQueueId, input_tensor, exponent, memory_config, optional_output_tensor);
    }
};
template <UnaryCompositeOpType unary_comp_op_type>
struct ExecuteUnaryCompositeOp
{
    static ttnn::Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt)
        {
            auto op_type = get_function_type1<unary_comp_op_type>();
            return op_type(input_tensor, memory_config);
        }

};

template <UnaryCompositeOpType unary_comp_op_type>
struct ExecuteUnaryCompositeOpWithScaleShift
{
    static ttnn::Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        float scale,
        float shift,
        const std::optional<MemoryConfig>& memory_config = std::nullopt)
        {
            auto op_type = get_function_type2<unary_comp_op_type>();
            return op_type(input_tensor, scale, shift, memory_config);
        }

};

template <UnaryCompositeOpType unary_comp_op_type>
struct ExecuteUnaryCompositeOpWithLowHigh
{
    static ttnn::Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        float low,
        float high,
        const std::optional<MemoryConfig>& memory_config = std::nullopt)
        {
            auto op_type = get_function_type3<unary_comp_op_type>();
            return op_type(input_tensor, low, high, memory_config);
        }

};

template <UnaryCompositeOpType unary_comp_op_type>
struct ExecuteUnaryCompositeOpWithScaleAlpha
{
    static ttnn::Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        float scale,
        float alpha,
        const std::optional<MemoryConfig>& memory_config = std::nullopt)
        {
            auto op_type = get_function_type4<unary_comp_op_type>();
            return op_type(input_tensor, scale, alpha, memory_config);
        }

};

template <UnaryCompositeOpType unary_comp_op_type>
struct ExecuteUnaryCompositeOpWithThresholdValue
{
    static ttnn::Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        float threshold,
        float value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt)
        {
            auto op_type = get_function_type5<unary_comp_op_type>();
            return op_type(input_tensor, threshold, value, memory_config);
        }

};


// re-implement tt_eager composite unary op => ttnn composite unary ops.
Tensor rdiv(uint8_t queue_id, const Tensor& input_tensor, float value, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    float t_inf = std::numeric_limits<float>::infinity();
    Tensor recip_result = ttnn::reciprocal(queue_id, input_tensor, memory_config, optional_output_tensor);
    Tensor result = ttnn::multiply(queue_id, recip_result, value, std::nullopt, memory_config, optional_output_tensor);

    auto output_memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config() : memory_config.value_or(input_tensor.memory_config());
    return tt::tt_metal::where(ttnn::eqz(queue_id, input_tensor, output_memory_config), t_inf, result, output_memory_config, optional_output_tensor);
}

// To be used for div op overloading in binary composite
Tensor div_unary(uint8_t queue_id, const Tensor& input_tensor, float value, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return ttnn::multiply(queue_id, input_tensor, (1.0f / value), std::nullopt, memory_config, optional_output_tensor);
}

// TODO: update these composite unary ops pending decision on TensorAsync implementation.

// TODO: implement these composite unary ops with optional output tensor and queue id.

Tensor tril(
    uint8_t queue_id,
    const Tensor& input_tensor,
    int32_t diag=0,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::tril(input_tensor, diag, memory_config.value_or(input_tensor.memory_config()));
}
Tensor triu(
    uint8_t queue_id,
    const Tensor& input_tensor,
    int32_t diag=0,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::triu(input_tensor, diag, memory_config.value_or(input_tensor.memory_config()));
}

}  // namespace unary
}  // namespace operations

// auto prelu = ttnn::leaky_relu;  // Alias for leaky_relu. TODO(#8544): implement PReLU properly

constexpr auto pow = ttnn::register_operation<ttnn::operations::unary::Power>("ttnn::pow");

// Other unaries

// This function is used to transform the arguments of a function before calling it
// where the lambda is applied to the type that matches T.
// Example: https://godbolt.org/z/3P9YedMdj
template <typename T, typename Func, typename Lambda, typename... Args>
constexpr auto transform_args_lambda(Func func, Lambda lambda, Args&&... args) -> decltype(auto) {
    auto transformer = [lambda](auto&& arg) -> decltype(auto) {
        if constexpr (std::is_same_v<T, std::decay_t<decltype(arg)>>) {
            return lambda(std::forward<decltype(arg)>(arg));
        } else {
            return std::forward<decltype(arg)>(arg);
        }
    };

    return func(transformer(std::forward<Args>(args))...);
}

template <typename T, typename Lambda>
auto transform_first_matching_arg(Lambda lambda) {
    static_assert(!std::is_same<T, T>::value, "No matching type found");
}

template <typename T, typename Lambda, typename First, typename... Rest>
auto transform_first_matching_arg(Lambda lambda, First&& first, Rest&&... rest) {
    if constexpr (std::is_same_v<T, std::decay_t<First>>) {
        return lambda(std::forward<First>(first));
    } else {
        return transform_first_matching_arg<T>(lambda, std::forward<Rest>(rest)...);
    }
}
#define WRAP_WITH_RESHAPE(function)                                                                    \
    ([](auto&&... args) {                                                                              \
        const auto original_shape = transform_first_matching_arg<Tensor>(                              \
            [&](auto&& tensor) { return tensor.get_shape(); }, std::forward<decltype(args)>(args)...); \
        return ttnn::reshape(                                                                          \
            transform_args_lambda<Tensor>(                                                             \
                function, [&](auto&& tensor) { return ttnn::unsqueeze_to_4D(tensor); }, args...),      \
            original_shape);                                                                           \
    })

constexpr auto rdiv = REGISTER_OPERATION_FROM_FUNCTION("ttnn::rdiv", WRAP_WITH_RESHAPE(ttnn::operations::unary::rdiv));

constexpr auto tril = REGISTER_OPERATION_FROM_FUNCTION("ttnn::tril", WRAP_WITH_RESHAPE(ttnn::operations::unary::tril));
constexpr auto triu = REGISTER_OPERATION_FROM_FUNCTION("ttnn::triu", WRAP_WITH_RESHAPE(ttnn::operations::unary::triu));

// newly imported
constexpr auto tanhshrink = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::TANHSHRINK>>("ttnn::tanhshrink");
constexpr auto deg2rad= ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::DEG2RAD>>("ttnn::deg2rad");
constexpr auto rad2deg= ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::RAD2DEG>>("ttnn::rad2deg");
constexpr auto acosh = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::ACOSH>>("ttnn::acosh");
constexpr auto asinh = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::ASINH>>("ttnn::asinh");
constexpr auto atanh = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::ATANH>>("ttnn::atanh");
constexpr auto cbrt = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::CBRT>>("ttnn::cbrt");
constexpr auto cosh = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::COSH>>("ttnn::cosh");
constexpr auto digamma = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::DIGAMMA>>("ttnn::digamma");
constexpr auto lgamma = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::LGAMMA>>("ttnn::lgamma");
constexpr auto log1p = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::LOG1P>>("ttnn::log1p");
constexpr auto mish = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::MISH>>("ttnn::mish");
constexpr auto multigammaln = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::MULTIGAMMALN>>("ttnn::multigammaln");
constexpr auto sinh = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::SINH>>("ttnn::sinh");
constexpr auto softsign= ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::SOFTSIGN>>("ttnn::softsign");
constexpr auto swish = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::SWISH>>("ttnn::swish");
constexpr auto trunc = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::TRUNC>>("ttnn::trunc");
constexpr auto var_hw = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::VAR_HW>>("ttnn::var_hw");
constexpr auto std_hw = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::STD_HW>>("ttnn::std_hw");
constexpr auto normalize_hw = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::NORMALIZE_HW>>("ttnn::normalize_hw");

constexpr auto hardswish= ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOpWithScaleShift<operations::unary::UnaryCompositeOpType::HARDSWISH>>("ttnn::hardswish");
constexpr auto hardsigmoid = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOpWithScaleShift<operations::unary::UnaryCompositeOpType::HARDSIGMOID>>("ttnn::hardsigmoid");

constexpr auto hardtanh = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOpWithLowHigh<operations::unary::UnaryCompositeOpType::HARDTANH>>("ttnn::hardtanh");
constexpr auto clip = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOpWithLowHigh<operations::unary::UnaryCompositeOpType::CLIP>>("ttnn::clip");
constexpr auto clamp = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOpWithLowHigh<operations::unary::UnaryCompositeOpType::CLAMP>>("ttnn::clamp");
constexpr auto selu = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOpWithScaleAlpha<operations::unary::UnaryCompositeOpType::SELU>>("ttnn::selu");
constexpr auto threshold = ttnn::register_operation<operations::unary::ExecuteUnaryCompositeOpWithThresholdValue<operations::unary::UnaryCompositeOpType::THRESHOLD>>("ttnn::threshold");

}  // namespace ttnn
