// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "device/maxpool3d_device_operation.hpp"

namespace ttnn::operations::experimental::maxpool3d {

struct ExecuteMaxPool3d {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const std::array<uint32_t, 3>& kernel_size = {2, 2, 2},
        const std::array<uint32_t, 3>& stride = {2, 2, 2},
        const std::array<uint32_t, 3>& padding = {0, 0, 0},
        const std::string& padding_mode = "zeros",
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace ttnn::operations::experimental::maxpool3d

namespace ttnn::experimental {
constexpr auto maxpool3d = ttnn::
    register_operation<"ttnn::experimental::maxpool3d", ttnn::operations::experimental::maxpool3d::ExecuteMaxPool3d>();
}
