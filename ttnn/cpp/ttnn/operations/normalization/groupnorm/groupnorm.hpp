// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "device/groupnorm_device_operation_types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {

Tensor group_norm(
    const Tensor& input_tensor,
    int num_groups,
    float epsilon,
    const std::optional<Tensor>& input_mask = std::nullopt,
    const std::optional<Tensor>& weight = std::nullopt,
    const std::optional<Tensor>& bias = std::nullopt,
    const std::optional<Tensor>& reciprocals = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<DataType> dtype = std::nullopt,
    std::optional<CoreGrid> core_grid = std::nullopt,
    std::optional<bool> inplace = std::nullopt,
    std::optional<Layout> output_layout = std::nullopt,
    std::optional<int> num_out_blocks = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<Tensor>& negative_mask = std::nullopt,
    bool use_welford = false);

}  // namespace ttnn
