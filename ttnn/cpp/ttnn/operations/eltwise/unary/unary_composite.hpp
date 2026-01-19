// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_composite_op.hpp"

namespace ttnn {

namespace operations::unary {

template <UnaryCompositeOpType unary_comp_op_type>
struct ExecuteUnaryCompositeOp {
    static Tensor invoke(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
        return OpHandler<unary_comp_op_type>::handle(input_tensor, output_memory_config);
    }
};

// OpHandler_float : get_function_type_float
template <UnaryCompositeOpType unary_comp_op_type>
struct ExecuteUnaryCompositeOpWithFloat {
    // Type : 1 inputs, 1 float
    static ttnn::Tensor invoke(
        const Tensor& input_tensor, float param1, const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<unary_comp_op_type>::handle(input_tensor, param1, memory_config);
    }
};

template <UnaryCompositeOpType unary_comp_op_type>
struct ExecuteUnaryCompositeOpWithDim {
    static Tensor invoke(
        const Tensor& input_tensor, int32_t dim, const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
        return OpHandler<unary_comp_op_type>::handle(input_tensor, dim, output_memory_config);
    }
};

template <UnaryCompositeOpType unary_comp_op_type>
struct ExecuteUnaryCompositeOpWithFloats {
    // Type 1: 1 inputs, 2 float
    static Tensor invoke(
        const Tensor& input_tensor,
        float param1,
        float param2,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
        return OpHandler<unary_comp_op_type>::handle(input_tensor, param1, param2, output_memory_config);
    }
};

struct ExecuteUnaryCompositeClamp {
    static Tensor invoke(
        const Tensor& input_a,
        std::optional<std::variant<float, int32_t>> min = std::nullopt,
        std::optional<std::variant<float, int32_t>> max = std::nullopt,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<Tensor>& output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& a,
        std::optional<Tensor> min = std::nullopt,
        std::optional<Tensor> max = std::nullopt,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<Tensor>& output_tensor = std::nullopt);
};

struct ExecuteUnaryCompositeClip {
    static Tensor invoke(
        const Tensor& a,
        std::optional<float> min = std::nullopt,
        std::optional<float> max = std::nullopt,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

    static Tensor invoke(
        const Tensor& a,
        std::optional<Tensor> min = std::nullopt,
        std::optional<Tensor> max = std::nullopt,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

template <UnaryCompositeOpType unary_comp_op_type>
struct ExecuteUnaryCompositeOpWithInt {
    static Tensor invoke(
        const Tensor& input_tensor, int32_t param1, const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
        return OpHandler<unary_comp_op_type>::handle(input_tensor, param1, output_memory_config);
    }
};

}  // namespace operations::unary

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
auto transform_first_matching_arg(Lambda /*lambda*/) {
    static_assert(!std::is_same_v<T, T>, "No matching type found");
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

constexpr auto digamma = ttnn::register_operation<
    "ttnn::digamma",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::DIGAMMA>>();
constexpr auto lgamma = ttnn::register_operation<
    "ttnn::lgamma",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::LGAMMA>>();
constexpr auto multigammaln = ttnn::register_operation<
    "ttnn::multigammaln",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::MULTIGAMMALN>>();
constexpr auto var_hw = ttnn::register_operation<
    "ttnn::var_hw",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::VAR_HW>>();
constexpr auto std_hw = ttnn::register_operation<
    "ttnn::std_hw",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::STD_HW>>();
constexpr auto normalize_hw = ttnn::register_operation<
    "ttnn::normalize_hw",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::NORMALIZE_HW>>();
constexpr auto clip = ttnn::register_operation<"ttnn::clip", operations::unary::ExecuteUnaryCompositeClip>();
constexpr auto clamp = ttnn::register_operation<"ttnn::clamp", operations::unary::ExecuteUnaryCompositeClamp>();
constexpr auto glu = ttnn::register_operation<
    "ttnn::glu",
    operations::unary::ExecuteUnaryCompositeOpWithDim<operations::unary::UnaryCompositeOpType::GLU>>();
constexpr auto reglu = ttnn::register_operation<
    "ttnn::reglu",
    operations::unary::ExecuteUnaryCompositeOpWithDim<operations::unary::UnaryCompositeOpType::REGLU>>();
constexpr auto geglu = ttnn::register_operation<
    "ttnn::geglu",
    operations::unary::ExecuteUnaryCompositeOpWithDim<operations::unary::UnaryCompositeOpType::GEGLU>>();
constexpr auto swiglu = ttnn::register_operation<
    "ttnn::swiglu",
    operations::unary::ExecuteUnaryCompositeOpWithDim<operations::unary::UnaryCompositeOpType::SWIGLU>>();
constexpr auto logical_not_ = ttnn::register_operation<
    "ttnn::logical_not_",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::LOGICAL_NOT_>>();
constexpr auto tril = ttnn::register_operation<
    "ttnn::tril",
    operations::unary::ExecuteUnaryCompositeOpWithInt<operations::unary::UnaryCompositeOpType::TRIL>>();
constexpr auto triu = ttnn::register_operation<
    "ttnn::triu",
    operations::unary::ExecuteUnaryCompositeOpWithInt<operations::unary::UnaryCompositeOpType::TRIU>>();
constexpr auto polygamma = ttnn::register_operation<
    "ttnn::polygamma",
    operations::unary::ExecuteUnaryCompositeOpWithInt<operations::unary::UnaryCompositeOpType::POLYGAMMA>>();
constexpr auto normalize_global = ttnn::register_operation<
    "ttnn::normalize_global",
    operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::NORMALIZE_GLOBAL>>();

}  // namespace ttnn
