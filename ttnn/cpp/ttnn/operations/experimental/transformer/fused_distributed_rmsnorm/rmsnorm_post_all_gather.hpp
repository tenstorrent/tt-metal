// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using tt::tt_metal::DataType;
using tt::tt_metal::MemoryConfig;

namespace ttnn::experimental {

Tensor wan_fused_rmsnorm_post_allgather(
    const Tensor& input_tensor,
    const Tensor& stats,
    float epsilon = 1e-5,
    uint32_t num_heads = 1,
    const std::optional<const Tensor>& weight = std::nullopt,
    const std::optional<const Tensor>& transformation_mat = std::nullopt,
    const std::optional<const Tensor>& rope_cos = std::nullopt,
    const std::optional<const Tensor>& rope_sin = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<const DataType>& dtype = std::nullopt);

}  // namespace ttnn::experimental
