// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {

Tensor randn(
    const ttnn::Shape& size,
    MeshDevice& device,
    DataType dtype = DataType::BFLOAT16,
    Layout layout = Layout::TILE,
    const MemoryConfig& memory_config = types::DRAM_MEMORY_CONFIG,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    std::optional<uint32_t> seed = std::nullopt);

}  // namespace ttnn
