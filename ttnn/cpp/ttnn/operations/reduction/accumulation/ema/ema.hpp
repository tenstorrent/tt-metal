// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::reduction::accumulation {

struct EmaOperation {
    static Tensor invoke(
        const Tensor& input_tensor,
        const float& alpha,
        std::optional<Tensor> optional_out,
        std::optional<CoreGrid> core_grid = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace operations::reduction::accumulation

constexpr auto ema = ttnn::register_operation<"ttnn::ema", ttnn::operations::reduction::accumulation::EmaOperation>();

}  // namespace ttnn
