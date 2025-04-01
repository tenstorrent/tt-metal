// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"

namespace ttnn {

namespace operations::conv {
namespace conv1d {

using OutputLength = uint32_t;
using Result = std::tuple<ttnn::Tensor, OutputLength, ttnn::Tensor, std::optional<ttnn::Tensor>>;
using Conv1dConfig = ttnn::operations::conv::conv2d::Conv2dConfig;

template <typename T>
Result conv1d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    T* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_length,
    uint32_t kernel_size,
    uint32_t stride,
    std::variant<std::array<uint32_t, 2>, uint32_t> padding,
    uint32_t dilation = 1,
    uint32_t groups = 1,
    std::optional<const ttnn::Tensor> bias_tensor = std::nullopt,
    const std::optional<const Conv1dConfig>& conv_config_ = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt);

struct Conv1dOperation {
    static Result invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        IDevice* device,
        uint32_t in_channels,
        uint32_t out_channels,
        uint32_t batch_size,
        uint32_t input_length,
        uint32_t kernel_size,
        uint32_t stride,
        std::variant<std::array<uint32_t, 2>, uint32_t> padding,
        uint32_t dilation = 1,
        uint32_t groups = 1,
        std::optional<const ttnn::Tensor> bias_tensor = std::nullopt,
        const std::optional<const Conv1dConfig>& conv_config_ = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt);

    static Result invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        MeshDevice* device,
        uint32_t in_channels,
        uint32_t out_channels,
        uint32_t batch_size,
        uint32_t input_length,
        uint32_t kernel_size,
        uint32_t stride,
        std::variant<std::array<uint32_t, 2>, uint32_t> padding,
        uint32_t dilation = 1,
        uint32_t groups = 1,
        std::optional<const ttnn::Tensor> bias_tensor = std::nullopt,
        const std::optional<const Conv1dConfig>& conv_config_ = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt);
};
}  // namespace conv1d
}  // namespace operations::conv
}  // namespace ttnn

namespace ttnn {
constexpr auto conv1d = ttnn::register_operation<"ttnn::conv1d", operations::conv::conv1d::Conv1dOperation>();
}
