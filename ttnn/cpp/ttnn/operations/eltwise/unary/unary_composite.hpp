// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_composite_op.hpp"

namespace ttnn {

Tensor digamma(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt);
Tensor lgamma(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt);
Tensor multigammaln(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt);
Tensor var_hw(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt);
Tensor std_hw(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt);
Tensor normalize_hw(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt);
Tensor clip(
    const Tensor& a,
    std::optional<float> min = std::nullopt,
    std::optional<float> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor clip(
    const Tensor& a,
    std::optional<Tensor> min = std::nullopt,
    std::optional<Tensor> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor clamp(
    const Tensor& input_a,
    std::optional<std::variant<float, int32_t>> min = std::nullopt,
    std::optional<std::variant<float, int32_t>> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);
Tensor clamp(
    const Tensor& a,
    std::optional<Tensor> min = std::nullopt,
    std::optional<Tensor> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);
Tensor glu(const Tensor& t, int32_t dim = -1, const std::optional<MemoryConfig>& m = std::nullopt);
Tensor reglu(const Tensor& t, int32_t dim = -1, const std::optional<MemoryConfig>& m = std::nullopt);
Tensor geglu(const Tensor& t, int32_t dim = -1, const std::optional<MemoryConfig>& m = std::nullopt);
Tensor swiglu(const Tensor& t, int32_t dim = -1, const std::optional<MemoryConfig>& m = std::nullopt);
Tensor logical_not_(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt);
Tensor tril(const Tensor& t, int32_t diagonal = 0, const std::optional<MemoryConfig>& m = std::nullopt);
Tensor triu(const Tensor& t, int32_t diagonal = 0, const std::optional<MemoryConfig>& m = std::nullopt);
Tensor polygamma(const Tensor& t, int32_t param, const std::optional<MemoryConfig>& m = std::nullopt);
Tensor normalize_global(const Tensor& t, const std::optional<MemoryConfig>& m = std::nullopt);

}  // namespace ttnn
