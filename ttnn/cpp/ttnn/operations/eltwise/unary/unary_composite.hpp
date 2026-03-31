// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_composite_op.hpp"

namespace ttnn {

Tensor multigammaln(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor var_hw(const Tensor& y, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor std_hw(const Tensor& y, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor normalize_hw(const Tensor& y, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor clip(
    const Tensor& input_a,
    std::optional<float> min = std::nullopt,
    std::optional<float> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor clip(
    const Tensor& input_a,
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
    const Tensor& input_a,
    std::optional<Tensor> min = std::nullopt,
    std::optional<Tensor> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);
Tensor glu(
    const Tensor& input_a, int32_t dim = -1, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor reglu(
    const Tensor& input_a, int32_t dim = -1, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor geglu(
    const Tensor& input_a, int32_t dim = -1, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor swiglu(
    const Tensor& input_a, int32_t dim = -1, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor logical_not_(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor tril(
    const Tensor& input_a, int32_t diag = 0, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor triu(
    const Tensor& input_a, int32_t diag = 0, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor polygamma(const Tensor& input_a, int32_t k, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor normalize_global(const Tensor& y, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

}  // namespace ttnn
