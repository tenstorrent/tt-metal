// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
#include "device/conv3d_device_operation.hpp"

namespace ttnn::operations::experimental::conv3d {

struct ExecuteConv3d {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        const std::optional<ttnn::Tensor>& bias_tensor,
        const Conv3dConfig& config,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace ttnn::operations::experimental::conv3d

namespace ttnn::experimental {
constexpr auto conv3d =
    ttnn::register_operation<"ttnn::experimental::conv3d", ttnn::operations::experimental::conv3d::ExecuteConv3d>();
}
