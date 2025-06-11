// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::reduction {

enum class ReduceType {
    Sum,
    Mean,
    Max,
    Min,
    Std,
    Var,
};

template <ReduceType reduce_type>
struct Reduce {
    static Tensor invoke(
        const Tensor& input_tensor_arg,
        const std::optional<std::variant<int, ttnn::SmallVector<int>>>& dim_arg = std::nullopt,
        const bool keepdim = false,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
        float scalar = 1.0f,
        bool correction = true);
};

// Entry point for pool op, which uses non-standard tensors that cannot be padded.
[[deprecated]]
Tensor pool_sum(
    const Tensor& input_tensor_arg,
    int dim_arg,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar);

}  // namespace operations::reduction

// Generic reductions
constexpr auto sum = ttnn::register_operation<
    "ttnn::sum",
    ttnn::operations::reduction::Reduce<ttnn::operations::reduction::ReduceType::Sum>>();

constexpr auto mean = ttnn::register_operation<
    "ttnn::mean",
    ttnn::operations::reduction::Reduce<ttnn::operations::reduction::ReduceType::Mean>>();

constexpr auto max = ttnn::register_operation<
    "ttnn::max",
    ttnn::operations::reduction::Reduce<ttnn::operations::reduction::ReduceType::Max>>();

constexpr auto min = ttnn::register_operation<
    "ttnn::min",
    ttnn::operations::reduction::Reduce<ttnn::operations::reduction::ReduceType::Min>>();

constexpr auto std = ttnn::register_operation<
    "ttnn::std",
    ttnn::operations::reduction::Reduce<ttnn::operations::reduction::ReduceType::Std>>();

constexpr auto var = ttnn::register_operation<
    "ttnn::var",
    ttnn::operations::reduction::Reduce<ttnn::operations::reduction::ReduceType::Var>>();

}  // namespace ttnn
