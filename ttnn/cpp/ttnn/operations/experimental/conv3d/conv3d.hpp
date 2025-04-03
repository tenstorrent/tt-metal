// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "device/conv3d_device_operation.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
// #include <tt-metalium/operation.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations {
namespace experimental {
namespace conv3d {
struct Conv3dConfig;
}  // namespace conv3d
}  // namespace experimental
}  // namespace operations
}  // namespace ttnn

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
constexpr auto conv3d = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::conv3d",
    ttnn::operations::experimental::conv3d::ExecuteConv3d>();

}
