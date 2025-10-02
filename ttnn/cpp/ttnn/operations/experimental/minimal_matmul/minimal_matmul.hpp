// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
// #include <tt-metalium/operation.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "device/minimal_matmul_device_operation.hpp"

namespace ttnn::operations::experimental::minimal_matmul {

struct ExecuteMinimalMatmul {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        const std::optional<ttnn::Tensor>& bias_tensor,
        const MinimalMatmulConfig& config,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace ttnn::operations::experimental::minimal_matmul

namespace ttnn::experimental {
constexpr auto minimal_matmul = ttnn::register_operation<
    "ttnn::experimental::minimal_matmul",
    ttnn::operations::experimental::minimal_matmul::ExecuteMinimalMatmul>();
}
