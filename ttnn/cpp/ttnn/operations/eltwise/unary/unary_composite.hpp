// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

// Free functions for unary composite operations
inline Tensor digamma(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::DIGAMMA>::invoke(t, m);
}
inline Tensor lgamma(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::LGAMMA>::invoke(t, m);
}
inline Tensor multigammaln(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::MULTIGAMMALN>::invoke(
        t, m);
}
inline Tensor var_hw(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::VAR_HW>::invoke(t, m);
}
inline Tensor std_hw(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::STD_HW>::invoke(t, m);
}
inline Tensor normalize_hw(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::NORMALIZE_HW>::invoke(
        t, m);
}
inline Tensor clip(
    const Tensor& a,
    std::optional<float> min = std::nullopt,
    std::optional<float> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeClip::invoke(a, min, max, output_mem_config);
}
inline Tensor clip(
    const Tensor& a,
    std::optional<Tensor> min = std::nullopt,
    std::optional<Tensor> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeClip::invoke(a, min, max, output_mem_config);
}
inline Tensor clamp(
    const Tensor& input_a,
    std::optional<std::variant<float, int32_t>> min = std::nullopt,
    std::optional<std::variant<float, int32_t>> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeClamp::invoke(input_a, min, max, output_mem_config, output_tensor);
}
inline Tensor clamp(
    const Tensor& a,
    std::optional<Tensor> min = std::nullopt,
    std::optional<Tensor> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeClamp::invoke(a, min, max, output_mem_config, output_tensor);
}
inline Tensor glu(const Tensor& t, int32_t dim = -1, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeOpWithDim<operations::unary::UnaryCompositeOpType::GLU>::invoke(
        t, dim, m);
}
inline Tensor reglu(const Tensor& t, int32_t dim = -1, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeOpWithDim<operations::unary::UnaryCompositeOpType::REGLU>::invoke(
        t, dim, m);
}
inline Tensor geglu(const Tensor& t, int32_t dim = -1, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeOpWithDim<operations::unary::UnaryCompositeOpType::GEGLU>::invoke(
        t, dim, m);
}
inline Tensor swiglu(const Tensor& t, int32_t dim = -1, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeOpWithDim<operations::unary::UnaryCompositeOpType::SWIGLU>::invoke(
        t, dim, m);
}
inline Tensor logical_not_(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeOp<operations::unary::UnaryCompositeOpType::LOGICAL_NOT_>::invoke(
        t, m);
}
inline Tensor tril(const Tensor& t, int32_t diagonal = 0, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeOpWithInt<operations::unary::UnaryCompositeOpType::TRIL>::invoke(
        t, diagonal, m);
}
inline Tensor triu(const Tensor& t, int32_t diagonal = 0, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeOpWithInt<operations::unary::UnaryCompositeOpType::TRIU>::invoke(
        t, diagonal, m);
}
inline Tensor polygamma(const Tensor& t, int32_t param, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeOpWithInt<
        operations::unary::UnaryCompositeOpType::POLYGAMMA>::invoke(t, param, m);
}
inline Tensor normalize_global(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::ExecuteUnaryCompositeOp<
        operations::unary::UnaryCompositeOpType::NORMALIZE_GLOBAL>::invoke(t, m);
}

}  // namespace ttnn
