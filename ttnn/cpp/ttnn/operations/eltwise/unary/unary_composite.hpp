// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_composite_op.hpp"

namespace ttnn {

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

// Free functions for unary composite operations (impl is the public API)
inline Tensor digamma(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::digamma(t, m);
}
Tensor lgamma(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt);
inline Tensor multigammaln(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::multigammaln(t, m);
}
inline Tensor var_hw(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::var_hw(t, m);
}
inline Tensor std_hw(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::std_hw(t, m);
}
inline Tensor normalize_hw(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::normalize_hw(t, m);
}
inline Tensor clip(
    const Tensor& a,
    std::optional<float> min = std::nullopt,
    std::optional<float> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt) {
    return operations::unary::clip(a, min, max, output_mem_config);
}
inline Tensor clip(
    const Tensor& a,
    std::optional<Tensor> min = std::nullopt,
    std::optional<Tensor> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt) {
    return operations::unary::clip(a, std::move(min), std::move(max), output_mem_config);
}
inline Tensor clamp(
    const Tensor& input_a,
    std::optional<std::variant<float, int32_t>> min = std::nullopt,
    std::optional<std::variant<float, int32_t>> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt) {
    return operations::unary::clamp(input_a, min, max, output_mem_config, output_tensor);
}
inline Tensor clamp(
    const Tensor& a,
    std::optional<Tensor> min = std::nullopt,
    std::optional<Tensor> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt) {
    return operations::unary::clamp(a, std::move(min), std::move(max), output_mem_config, output_tensor);
}
inline Tensor glu(const Tensor& t, int32_t dim = -1, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::glu(t, dim, m);
}
inline Tensor reglu(const Tensor& t, int32_t dim = -1, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::reglu(t, dim, m);
}
inline Tensor geglu(const Tensor& t, int32_t dim = -1, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::geglu(t, dim, m);
}
inline Tensor swiglu(const Tensor& t, int32_t dim = -1, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::swiglu(t, dim, m);
}
inline Tensor logical_not_(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::logical_not_(t, m);
}
inline Tensor tril(const Tensor& t, int32_t diagonal = 0, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::tril(t, diagonal, m);
}
inline Tensor triu(const Tensor& t, int32_t diagonal = 0, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::triu(t, diagonal, m);
}
inline Tensor polygamma(const Tensor& t, int32_t param, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::polygamma(t, param, m);
}
inline Tensor normalize_global(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt) {
    return operations::unary::normalize_global(t, m);
}

}  // namespace ttnn
