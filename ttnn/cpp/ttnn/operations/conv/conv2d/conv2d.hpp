// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>

#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"

namespace ttnn {

namespace operations::conv {
namespace conv2d {

using OutputHeight = uint32_t;
using OutputWidth = uint32_t;
using Result = std::tuple<ttnn::Tensor, OutputHeight, OutputWidth, ttnn::Tensor, std::optional<ttnn::Tensor>>;

template <typename T>
Result conv2d_L1(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    T* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config);

template <typename T>
Result conv2d_DRAM(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    T* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor = std::nullopt,
    const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
    const std::optional<const MemoryConfig>& memory_config_ = std::nullopt,
    const Conv2dSliceConfig& dram_slice_config_ = Conv2dSliceConfig{
        .slice_type = Conv2dSliceConfig::SliceType::WIDTH, .num_slices = 0});

// GCC refuses to create a symbol for this function if it is defined in the source file, leading to linker errors.
template <typename T>
inline Result conv2d(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    T* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor = std::nullopt,
    const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
    const std::optional<const MemoryConfig>& memory_config_ = std::nullopt,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_ = std::nullopt) {
    if (dram_slice_config_.has_value()) {
        return conv2d_DRAM(
            queue_id,
            input_tensor,
            weight_tensor,
            device,
            in_channels,
            out_channels,
            batch_size,
            input_height,
            input_width,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias_tensor,
            conv_config_,
            compute_config_,
            memory_config_,
            dram_slice_config_.value());
    } else {
        return conv2d_L1(
            queue_id,
            input_tensor,
            weight_tensor,
            device,
            in_channels,
            out_channels,
            batch_size,
            input_height,
            input_width,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias_tensor,
            conv_config_,
            compute_config_,
            memory_config_);
    }
}

struct Conv2dOperation {
    static Result invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        IDevice* device,
        uint32_t in_channels,
        uint32_t out_channels,
        uint32_t batch_size,
        uint32_t input_height,
        uint32_t input_width,
        std::array<uint32_t, 2> kernel_size,
        std::array<uint32_t, 2> stride,
        std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
        std::array<uint32_t, 2> dilation,
        uint32_t groups,
        std::optional<const ttnn::Tensor> bias_tensor = std::nullopt,
        const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
        const std::optional<const MemoryConfig>& memory_config_ = std::nullopt,
        const std::optional<const Conv2dSliceConfig>& dram_slice_config_ = std::nullopt);

    static Result invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        MeshDevice* device,
        uint32_t in_channels,
        uint32_t out_channels,
        uint32_t batch_size,
        uint32_t input_height,
        uint32_t input_width,
        std::array<uint32_t, 2> kernel_size,
        std::array<uint32_t, 2> stride,
        std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
        std::array<uint32_t, 2> dilation,
        uint32_t groups,
        std::optional<const ttnn::Tensor> bias_tensor = std::nullopt,
        const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
        const std::optional<const MemoryConfig>& memory_config_ = std::nullopt,
        const std::optional<const Conv2dSliceConfig>& dram_slice_config_ = std::nullopt);
};
}  // namespace conv2d
}  // namespace operations::conv
}  // namespace ttnn

namespace ttnn {
constexpr auto conv2d = ttnn::register_operation<"ttnn::conv2d", operations::conv::conv2d::Conv2dOperation>();
}
