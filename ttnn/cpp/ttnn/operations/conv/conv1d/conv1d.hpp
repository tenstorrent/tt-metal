// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::conv::conv1d {

using OutputLength = uint32_t;
using Result = std::variant<
    ttnn::Tensor,
    std::tuple<ttnn::Tensor, OutputLength>,
    std::tuple<ttnn::Tensor, std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>>>,
    std::tuple<ttnn::Tensor, OutputLength, std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>>>>;

using Conv1dConfig = ttnn::prim::Conv2dConfig;

// Conv1dOperation is a wrapper around Conv2dOperation to handle 1D convolutions.
// It uses the same logic as Conv2dOperation but with 1D-specific parameters.
// The input and weight tensors are reshaped to 4D tensors before invoking the Conv2dOperation.
struct Conv1dOperation {
    static Result invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        MeshDevice* device,
        uint32_t in_channels,
        uint32_t out_channels,
        uint32_t batch_size,
        uint32_t input_length,
        uint32_t kernel_size,
        uint32_t stride = 1,
        std::variant<std::array<uint32_t, 2>, uint32_t> padding = uint32_t{0},
        uint32_t dilation = 1,
        uint32_t groups = 1,
        const std::optional<const DataType>& dtype = std::nullopt,
        const std::optional<const ttnn::Tensor>& bias_tensor = std::nullopt,
        const std::optional<const Conv1dConfig>& conv_config = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig>& compute_config = std::nullopt,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        bool return_output_dim = true,
        bool return_weights_and_bias = true);
};

}  // namespace ttnn::operations::conv::conv1d

namespace ttnn {
constexpr auto conv1d = ttnn::register_operation<"ttnn::conv1d", operations::conv::conv1d::Conv1dOperation>();
}
