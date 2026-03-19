// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_composite_op.hpp"

namespace ttnn {

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
