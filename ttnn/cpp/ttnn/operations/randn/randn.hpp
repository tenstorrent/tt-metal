// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::randn {
struct Randn {
    static Tensor invoke(
        const ttnn::Shape& size,
        MeshDevice& device,
        DataType dtype = DataType::BFLOAT16,
        Layout layout = Layout::TILE,
        const MemoryConfig& memory_config = types::DRAM_MEMORY_CONFIG,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
        uint32_t seed = 0);
};
}  // namespace ttnn::operations::randn

namespace ttnn {
constexpr auto randn = ttnn::register_operation<"ttnn::randn", ttnn::operations::randn::Randn>();
}  // namespace ttnn
