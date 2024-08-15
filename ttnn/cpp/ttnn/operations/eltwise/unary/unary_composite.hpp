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

struct ExecutePower{

     static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        uint32_t exponent,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt)
        {
        return OpHandler<UnaryCompositeOpType::POW>::handle(queue_id, input_tensor, exponent, memory_config.value_or(input_tensor.memory_config()), optional_output_tensor);
        }

    static Tensor invoke(
        const Tensor& input_tensor,
        uint32_t exponent,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt)
        {
        return OpHandler<UnaryCompositeOpType::POW>::handle(DefaultQueueId, input_tensor, exponent, memory_config.value_or(input_tensor.memory_config()), optional_output_tensor);
        }

    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        float exponent,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt)
        {
        return OpHandler<UnaryCompositeOpType::POW>::handle(queue_id, input_tensor, exponent, memory_config.value_or(input_tensor.memory_config()), optional_output_tensor);
        }

    static Tensor invoke(
        const Tensor& input_tensor,
        float exponent,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt)
        {
        return OpHandler<UnaryCompositeOpType::POW>::handle(DefaultQueueId, input_tensor, exponent, memory_config.value_or(input_tensor.memory_config()), optional_output_tensor);
        }
};

template <UnaryCompositeOpType unary_comp_op_type>
struct ExecuteUnaryCompositeOp {
    static Tensor invoke(
        const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
        return OpHandler<unary_comp_op_type>::handle(input_tensor, output_memory_config);
    }
};

//OpHandler_float : get_function_type_float
template <UnaryCompositeOpType unary_comp_op_type>
struct ExecuteUnaryCompositeOpWithFloat {

    //Type : 1 inputs, 1 float
    static ttnn::Tensor invoke(
        const Tensor &input_tensor,
        float param1,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        return OpHandler<unary_comp_op_type>::handle(input_tensor, param1, memory_config);
        }
};

template <UnaryCompositeOpType unary_comp_op_type>
struct ExecuteUnaryCompositeOpWithDim
{
    static Tensor invoke(
        const Tensor& input_tensor,
        int32_t dim,
        const std::optional<MemoryConfig>& memory_config = std::nullopt)
        {
            auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
            return OpHandler<unary_comp_op_type>::handle(input_tensor, dim, output_memory_config);
        }
};

template <UnaryCompositeOpType unary_comp_op_type>
struct ExecuteUnaryCompositeOpWithFloats {
    //Type 1: 1 inputs, 2 float
    static Tensor invoke(
        const Tensor &input_tensor,
        float param1,
        float param2,
        const std::optional<MemoryConfig> &memory_config = std::nullopt)
        {
            auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
            return OpHandler<unary_comp_op_type>::handle(input_tensor, param1, param2, output_memory_config);
        }
};

template <UnaryCompositeOpType unary_comp_op_type>
struct ExecuteUnaryCompositeOpWithInt {

    static Tensor invoke(
        const Tensor &input_tensor,
        int32_t param1,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
        return OpHandler<unary_comp_op_type>::handle(input_tensor, param1, output_memory_config);
        }
};

struct ExecuteRdiv {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        float value,
        const std::string& round_mode = "None",
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};

}  // namespace unary
}  // namespace operations

// auto prelu = ttnn::leaky_relu;  // Alias for leaky_relu. TODO(#8544): implement PReLU properly


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

constexpr auto rdiv = ttnn::register_operation_with_auto_launch_op<"ttnn::rdiv", operations::unary::ExecuteRdiv>();

constexpr auto pow = ttnn::register_operation_with_auto_launch_op<
    "ttnn::pow",
    operations::unary::ExecutePower>();
constexpr auto tanhshrink = ttnn::register_operation_with_auto_launch_op<
    "ttnn::tanhshrink",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::TANHSHRINK>>();
constexpr auto deg2rad = ttnn::register_operation_with_auto_launch_op<
    "ttnn::deg2rad",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::DEG2RAD>>();
constexpr auto rad2deg = ttnn::register_operation_with_auto_launch_op<
    "ttnn::rad2deg",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::RAD2DEG>>();
constexpr auto acosh = ttnn::register_operation_with_auto_launch_op<
    "ttnn::acosh",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::ACOSH>>();
constexpr auto asinh = ttnn::register_operation_with_auto_launch_op<
    "ttnn::asinh",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::ASINH>>();
constexpr auto atanh = ttnn::register_operation_with_auto_launch_op<
    "ttnn::atanh",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::ATANH>>();
constexpr auto cbrt = ttnn::register_operation_with_auto_launch_op<
    "ttnn::cbrt",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::CBRT>>();
constexpr auto cosh = ttnn::register_operation_with_auto_launch_op<
    "ttnn::cosh",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::COSH>>();
constexpr auto digamma = ttnn::register_operation_with_auto_launch_op<
    "ttnn::digamma",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::DIGAMMA>>();
constexpr auto lgamma = ttnn::register_operation_with_auto_launch_op<
    "ttnn::lgamma",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::LGAMMA>>();
constexpr auto log1p = ttnn::register_operation_with_auto_launch_op<
    "ttnn::log1p",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::LOG1P>>();
constexpr auto mish = ttnn::register_operation_with_auto_launch_op<
    "ttnn::mish",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::MISH>>();
constexpr auto multigammaln = ttnn::register_operation_with_auto_launch_op<
    "ttnn::multigammaln",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::MULTIGAMMALN>>();
constexpr auto sinh = ttnn::register_operation_with_auto_launch_op<
    "ttnn::sinh",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::SINH>>();
constexpr auto softsign = ttnn::register_operation_with_auto_launch_op<
    "ttnn::softsign",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::SOFTSIGN>>();
constexpr auto swish = ttnn::register_operation_with_auto_launch_op<
    "ttnn::swish",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::SWISH>>();
constexpr auto trunc = ttnn::register_operation_with_auto_launch_op<
    "ttnn::trunc",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::TRUNC>>();
constexpr auto var_hw = ttnn::register_operation_with_auto_launch_op<
    "ttnn::var_hw",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::VAR_HW>>();
constexpr auto std_hw = ttnn::register_operation_with_auto_launch_op<
    "ttnn::std_hw",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::STD_HW>>();
constexpr auto normalize_hw = ttnn::register_operation_with_auto_launch_op<
    "ttnn::normalize_hw",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::NORMALIZE_HW>>();

constexpr auto hardswish = ttnn::register_operation_with_auto_launch_op<
    "ttnn::hardswish",
    operations::unary::ExecuteUnaryCompositeOpWithFloats<operations::unary::UnaryCompositeOpType::HARDSWISH>>();
constexpr auto hardsigmoid = ttnn::register_operation_with_auto_launch_op<
    "ttnn::hardsigmoid",
    operations::unary::ExecuteUnaryCompositeOpWithFloats<operations::unary::UnaryCompositeOpType::HARDSIGMOID>>();

constexpr auto hardtanh = ttnn::register_operation_with_auto_launch_op<
    "ttnn::hardtanh",
    operations::unary::ExecuteUnaryCompositeOpWithFloats<operations::unary::UnaryCompositeOpType::HARDTANH>>();
constexpr auto clip = ttnn::register_operation_with_auto_launch_op<
    "ttnn::clip",
    operations::unary::ExecuteUnaryCompositeOpWithFloats<operations::unary::UnaryCompositeOpType::CLIP>>();
constexpr auto clamp = ttnn::register_operation_with_auto_launch_op<
    "ttnn::clamp",
    operations::unary::ExecuteUnaryCompositeOpWithFloats<operations::unary::UnaryCompositeOpType::CLAMP>>();
constexpr auto selu = ttnn::register_operation_with_auto_launch_op<
    "ttnn::selu",
    operations::unary::ExecuteUnaryCompositeOpWithFloats<operations::unary::UnaryCompositeOpType::SELU>>();
constexpr auto threshold = ttnn::register_operation_with_auto_launch_op<
    "ttnn::threshold",
    operations::unary::ExecuteUnaryCompositeOpWithFloats<operations::unary::UnaryCompositeOpType::THRESHOLD>>();

constexpr auto glu = ttnn::register_operation_with_auto_launch_op<
    "ttnn::glu",
    operations::unary::ExecuteUnaryCompositeOpWithDim<operations::unary::UnaryCompositeOpType::GLU>>();
constexpr auto reglu = ttnn::register_operation_with_auto_launch_op<
    "ttnn::reglu",
    operations::unary::ExecuteUnaryCompositeOpWithDim<operations::unary::UnaryCompositeOpType::REGLU>>();
constexpr auto geglu = ttnn::register_operation_with_auto_launch_op<
    "ttnn::geglu",
    operations::unary::ExecuteUnaryCompositeOpWithDim<operations::unary::UnaryCompositeOpType::GEGLU>>();
constexpr auto swiglu = ttnn::register_operation_with_auto_launch_op<
    "ttnn::swiglu",
    operations::unary::ExecuteUnaryCompositeOpWithDim<operations::unary::UnaryCompositeOpType::SWIGLU>>();
constexpr auto hardshrink = ttnn::register_operation_with_auto_launch_op<
    "ttnn::hardshrink",
    operations::unary::ExecuteUnaryCompositeOpWithFloat<operations::unary::UnaryCompositeOpType::HARDSHRINK>>();
constexpr auto logical_not_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::logical_not_",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::LOGICAL_NOT_>>();
constexpr auto softshrink = ttnn::register_operation_with_auto_launch_op<
    "ttnn::softshrink",
    operations::unary::ExecuteUnaryCompositeOpWithFloat<operations::unary::UnaryCompositeOpType::SOFTSHRINK>>();
constexpr auto logit = ttnn::register_operation_with_auto_launch_op<
    "ttnn::logit",
    operations::unary::ExecuteUnaryCompositeOpWithFloat<operations::unary::UnaryCompositeOpType::LOGIT>>();
constexpr auto celu = ttnn::register_operation_with_auto_launch_op<
    "ttnn::celu",
    operations::unary::ExecuteUnaryCompositeOpWithFloat<operations::unary::UnaryCompositeOpType::CELU>>();
constexpr auto tril = ttnn::register_operation_with_auto_launch_op<
    "ttnn::tril",
    operations::unary::ExecuteUnaryCompositeOpWithInt<operations::unary::UnaryCompositeOpType::TRIL>>();
constexpr auto triu = ttnn::register_operation_with_auto_launch_op<
    "ttnn::triu",
    operations::unary::ExecuteUnaryCompositeOpWithInt<operations::unary::UnaryCompositeOpType::TRIU>>();
constexpr auto round = ttnn::register_operation_with_auto_launch_op<
    "ttnn::round",
    operations::unary::ExecuteUnaryCompositeOpWithInt<operations::unary::UnaryCompositeOpType::ROUND>>();
constexpr auto polygamma = ttnn::register_operation_with_auto_launch_op<
    "ttnn::polygamma",
    operations::unary::ExecuteUnaryCompositeOpWithInt<operations::unary::UnaryCompositeOpType::POLYGAMMA>>();
constexpr auto rpow = ttnn::register_operation_with_auto_launch_op<
    "ttnn::rpow",
    operations::unary::ExecuteUnaryCompositeOpWithFloat<operations::unary::UnaryCompositeOpType::RPOW>>();
constexpr auto normalize_global = ttnn::register_operation_with_auto_launch_op<
    "ttnn::normalize_global",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::NORMALIZE_GLOBAL>>();
constexpr auto frac = ttnn::register_operation_with_auto_launch_op<
    "ttnn::frac",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::FRAC>>();


}  // namespace ttnn
