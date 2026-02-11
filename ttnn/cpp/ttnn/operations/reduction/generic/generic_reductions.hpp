// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/core_coord.hpp>

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
        bool keepdim = false,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
        float scalar = 1.0f,
        bool correction = true,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
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
Tensor sum(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor mean(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor max(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor min(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor std(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor var(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn
