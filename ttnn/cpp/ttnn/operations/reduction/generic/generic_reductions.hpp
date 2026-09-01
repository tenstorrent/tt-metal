// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include <tt_stl/small_vector.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/core_coord.hpp>
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations::reduction {

// Entry point for pool op, which uses non-standard tensors that cannot be padded.
[[deprecated]]
Tensor pool_sum(
    const Tensor& input_tensor_arg,
    int dim_arg,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    const std::optional<Layout>& output_layout = std::nullopt);

// Collapse dims 1 and 2 of a 4D tensor to (D0, 1, 1, D3) in a single launch, applying `scalar` once.
// Chooses how based on the buffer: a tensor contiguous across the dim-1/dim-2 boundary is reshaped
// into one long axis, which keeps the H-axis split available; a ROW_MAJOR tensor whose dim 2 is
// padded has pad rows between the dim-1 slices and no single-axis form, so it is reduced with NC
// grouping instead. Split out from the single-dim entry point because pool needs both axes at once.
[[deprecated]]
Tensor pool_sum_dims_1_2(
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    const std::optional<Layout>& output_layout = std::nullopt);

}  // namespace operations::reduction

// Generic reductions
Tensor sum(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    // When false (default), fp32 sum runs on the accurate SFPU path; true selects the faster tf32 FPU path.
    bool fast_and_approximate_mode = false,
    // Layout of the result. std::nullopt (default) is TILE, except a ROW_MAJOR input reduced over
    // -1/-2 on the dense RM path, which stays ROW_MAJOR. An explicit layout is always honored.
    const std::optional<Layout>& output_layout = std::nullopt);

Tensor mean(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    // When false (default), fp32 mean runs on the accurate SFPU path; true selects the faster tf32 FPU path.
    bool fast_and_approximate_mode = false,
    // See ttnn::sum above.
    const std::optional<Layout>& output_layout = std::nullopt);

Tensor max(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    // When false (default), fp32 max runs on the accurate SFPU path; true selects the faster tf32 FPU path.
    bool fast_and_approximate_mode = false);

Tensor min(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    // When false (default), fp32 min runs on the accurate SFPU path; true selects the faster tf32 FPU path.
    bool fast_and_approximate_mode = false);

Tensor std(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor var(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn
