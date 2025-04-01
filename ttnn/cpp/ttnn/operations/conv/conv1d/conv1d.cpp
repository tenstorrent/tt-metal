// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <utility>

#include <tt-metalium/buffer_constants.hpp>

#include "tt-metalium/logger.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/conv/conv1d/conv1d.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"

namespace ttnn {
namespace operations::conv {
using namespace tt;
using sliding_window::ParallelConfig;
using sliding_window::SlidingWindowConfig;

namespace conv1d {

using OutputLength = uint32_t;
using Result = std::tuple<ttnn::Tensor, OutputLength, ttnn::Tensor, std::optional<ttnn::Tensor>>;

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
    uint32_t dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv1dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config) {
    // reshape input tensor to 4D, if it is not already
    const ttnn::Tensor& input_tensor_4d =
        (input_tensor.get_logical_shape().rank() < 4)
            ? ttnn::reshape(input_tensor, Shape({batch_size, input_length, 1, in_channels}))
            : input_tensor;

    // reshape input tensor to 4D, if it is not already
    const ttnn::Tensor& weight_tensor_4d =
        (weight_tensor.get_logical_shape().rank() < 4)
            ? ttnn::reshape(weight_tensor, Shape({out_channels, in_channels / groups, kernel_size, 1}))
            : weight_tensor;
    // padding for conv2d based on conv1d padding
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> conv2d_padding;
    if (std::holds_alternative<uint32_t>(padding)) {
        conv2d_padding = std::array<uint32_t, 2>{std::get<uint32_t>(padding), 0};
    } else {
        std::array<uint32_t, 2> padding_lr = std::get<std::array<uint32_t, 2>>(padding);

        conv2d_padding = std::array<uint32_t, 4>{
            padding_lr[0],  // up
            padding_lr[1],  // down
            0,              // left
            0               // right
        };
    };

    auto [output_tensor, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device] = conv2d::conv2d(
        input_tensor_4d,
        // input_tensor,
        weight_tensor_4d,
        // weight_tensor,
        device,
        in_channels,
        out_channels,
        batch_size,
        input_length,
        1,  // input_width
        std::array<uint32_t, 2>{kernel_size, 1},
        std::array<uint32_t, 2>{stride, 1},
        conv2d_padding,
        std::array<uint32_t, 2>{dilation, 1},
        groups,
        std::move(bias_tensor),
        conv_config_,
        compute_config_,
        memory_config);

    return Result(output_tensor, output_height, weight_tensor_on_device, bias_tensor_on_device);
};

Result Conv1dOperation::invoke(
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
    uint32_t dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv1dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config) {
    return conv1d(
        input_tensor,
        weight_tensor,
        device,
        in_channels,
        out_channels,
        batch_size,
        input_length,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        std::move(bias_tensor),
        std::move(conv_config_),
        std::move(compute_config_),
        memory_config);
}

Result Conv1dOperation::invoke(
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
    uint32_t dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv1dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config) {
    return conv1d(
        input_tensor,
        weight_tensor,
        device,
        in_channels,
        out_channels,
        batch_size,
        input_length,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        std::move(bias_tensor),
        std::move(conv_config_),
        std::move(compute_config_),
        memory_config);
}

}  // namespace conv1d
}  // namespace operations::conv
}  // namespace ttnn
