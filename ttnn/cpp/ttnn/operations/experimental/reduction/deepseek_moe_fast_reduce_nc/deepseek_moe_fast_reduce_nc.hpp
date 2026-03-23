// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::reduction {

std::vector<ttnn::Tensor> deepseek_moe_fast_reduce_nc(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    uint64_t split_size,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental::reduction
