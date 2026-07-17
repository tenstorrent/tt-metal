// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/core_coord.hpp>
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar {

// Entry point for pool op, which uses non-standard tensors that cannot be padded.
[[deprecated]]
Tensor pool_sum(
    const Tensor& input_tensor_arg,
    int dim_arg,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar);

// Generic reductions
Tensor sum(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor mean(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor max(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor min(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

// use_legacy is deprecated and non-functional: the Welford implementation is always
// used. The parameter is kept only for API compatibility and will be removed.
Tensor std(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    bool use_legacy = false);

// use_legacy is deprecated and non-functional: the Welford implementation is always
// used. The parameter is kept only for API compatibility and will be removed.
Tensor var(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    bool use_legacy = false);

}  // namespace ttnn::operations::experimental::quasar
