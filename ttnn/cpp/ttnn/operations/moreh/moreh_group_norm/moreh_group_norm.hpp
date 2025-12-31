// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using tt::tt_metal::MemoryConfig;

namespace ttnn {

std::vector<std::optional<Tensor>> moreh_group_norm(
    const Tensor& input,
    uint32_t num_groups,
    float eps,
    const std::optional<const Tensor>& gamma = std::nullopt,
    const std::optional<const Tensor>& beta = std::nullopt,
    const std::vector<bool>& are_required_outputs = {true, true, true},
    const std::optional<const Tensor>& output = std::nullopt,
    const std::optional<const Tensor>& mean = std::nullopt,
    const std::optional<const Tensor>& rstd = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<MemoryConfig>& mean_memory_config = std::nullopt,
    const std::optional<MemoryConfig>& rstd_memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn
