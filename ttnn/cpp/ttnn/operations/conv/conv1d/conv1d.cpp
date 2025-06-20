// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

#include <tt-metalium/buffer_types.hpp>

#include <tt-logger/tt-logger.hpp>
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

template <typename T>
Result conv1d(
    QueueId queue_id,
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
    const std::optional<const DataType>& dtype,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv1dConfig>& conv_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_config,
    const std::optional<const MemoryConfig>& memory_config,
    bool return_output_dim,
    bool return_weights_and_bias) {
    // reshape input tensor to 4D, if it is not already
    const ttnn::Tensor& input_tensor_4d =
        (input_tensor.logical_shape().rank() < 4)
            ? ttnn::reshape(input_tensor, Shape({batch_size, input_length, 1, in_channels}))
            : input_tensor;

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

    auto [output_tensor, output_dimensions, weights_and_bias] =
        std::get<static_cast<int>(ResultType::OUTPUT_DIM_WEIGHTS_AND_BIAS)>(ttnn::conv2d(
            queue_id,
            input_tensor_4d,
            weight_tensor,
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
            std::move(dtype),
            std::move(bias_tensor),
            conv_config,
            compute_config,
            memory_config,
            std::nullopt,
            true,
            true));

    if (return_output_dim && return_weights_and_bias) {
        return Result(std::tuple(output_tensor, std::get<0>(output_dimensions), weights_and_bias));
    } else if (return_output_dim) {
        return Result(std::tuple(output_tensor, std::get<0>(output_dimensions)));
    } else if (return_weights_and_bias) {
        return Result(std::tuple(output_tensor, weights_and_bias));
    } else {
        return Result(output_tensor);
    };
}
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
    const std::optional<const DataType>& dtype,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv1dConfig>& conv_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_config,
    const std::optional<const MemoryConfig>& memory_config,
    bool return_output_dim,
    bool return_weights_and_bias) {
    return conv1d(
        queue_id,
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
        std::move(dtype),
        std::move(bias_tensor),
        std::move(conv_config),
        std::move(compute_config),
        memory_config,
        return_output_dim,
        return_weights_and_bias);
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
    const std::optional<const DataType>& dtype,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv1dConfig>& conv_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_config,
    const std::optional<const MemoryConfig>& memory_config,
    bool return_output_dim,
    bool return_weights_and_bias) {
    return conv1d(
        queue_id,
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
        std::move(dtype),
        std::move(bias_tensor),
        std::move(conv_config),
        std::move(compute_config),
        memory_config,
        return_output_dim,
        return_weights_and_bias);
}

}  // namespace conv1d
}  // namespace operations::conv
}  // namespace ttnn
