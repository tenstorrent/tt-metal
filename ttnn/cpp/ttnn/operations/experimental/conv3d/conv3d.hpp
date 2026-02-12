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
#include "device/conv3d_device_operation.hpp"

namespace ttnn::operations::experimental::conv3d {

struct ExecuteConv3d {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        const std::optional<ttnn::Tensor>& bias_tensor,
        const ttnn::experimental::prim::Conv3dConfig& config,
        tt::tt_metal::DataType dtype_,
        uint32_t output_channels_,
        const std::array<uint32_t, 3>& kernel_size_,
        const std::array<uint32_t, 3>& stride_ = std::array<uint32_t, 3>{1, 1, 1},
        const std::array<uint32_t, 3>& padding_ = std::array<uint32_t, 3>{0, 0, 0},
        const std::array<uint32_t, 3>& dilation_ = std::array<uint32_t, 3>{1, 1, 1},
        const std::string& padding_mode_ = "zeros",
        uint32_t groups_ = 1,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace ttnn::operations::experimental::conv3d

namespace ttnn::experimental {
constexpr auto conv3d =
    ttnn::register_operation<"ttnn::experimental::conv3d", ttnn::operations::experimental::conv3d::ExecuteConv3d>();
}
