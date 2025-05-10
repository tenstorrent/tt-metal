// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <stdexcept>
#include <tuple>
#include <variant>

#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/conv/conv_types.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"

namespace ttnn {

namespace operations::conv {
namespace conv2d {

using OutputHeight = uint32_t;
using OutputWidth = uint32_t;
using Result = std::tuple<ttnn::Tensor, OutputHeight, OutputWidth, ttnn::Tensor, std::optional<ttnn::Tensor>>;
using ResultWithOptions = std::variant<
    ttnn::Tensor,
    std::tuple<ttnn::Tensor, std::tuple<OutputHeight, OutputWidth>>,
    std::tuple<ttnn::Tensor, std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>>>,
    std::tuple<
        ttnn::Tensor,
        std::tuple<OutputHeight, OutputWidth>,
        std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>>>>;

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
    const std::optional<const ttnn::Tensor>& bias_tensor = std::nullopt,
    const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt);

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
    const std::optional<const ttnn::Tensor>& bias_tensor = std::nullopt,
    const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
    const std::optional<const MemoryConfig>& memory_config_ = std::nullopt,
    const Conv2dSliceConfig& dram_slice_config_ = Conv2dSliceConfig{
        .slice_type = Conv2dSliceConfig::SliceType::WIDTH, .num_slices = 0});

template <typename T>
ResultWithOptions conv2d(
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
    const std::optional<const ttnn::Tensor>& bias_tensor = std::nullopt,
    const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
    const std::optional<const MemoryConfig>& memory_config_ = std::nullopt,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_ = std::nullopt,
    bool return_output_dim = false,
    bool return_weights_and_bias = false);

struct Conv2dOperation {
    static ResultWithOptions invoke(
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
        std::array<uint32_t, 2> stride = std::array<uint32_t, 2>{1, 1},
        std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding = std::array<uint32_t, 2>{0, 0},
        std::array<uint32_t, 2> dilation = std::array<uint32_t, 2>{1, 1},
        uint32_t groups = 1,
        const std::optional<const ttnn::Tensor>& bias_tensor = std::nullopt,
        const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
        const std::optional<const MemoryConfig>& memory_config_ = std::nullopt,
        const std::optional<const Conv2dSliceConfig>& dram_slice_config_ = std::nullopt,
        bool return_output_dim = false,
        bool return_weights_and_bias = false);

    static ResultWithOptions invoke(
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
        std::array<uint32_t, 2> stride = std::array<uint32_t, 2>{1, 1},
        std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding = std::array<uint32_t, 2>{0, 0},
        std::array<uint32_t, 2> dilation = std::array<uint32_t, 2>{1, 1},
        uint32_t groups = 1,
        const std::optional<const ttnn::Tensor>& bias_tensor = std::nullopt,
        const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
        const std::optional<const MemoryConfig>& memory_config_ = std::nullopt,
        const std::optional<const Conv2dSliceConfig>& dram_slice_config_ = std::nullopt,
        bool return_output_dim = false,
        bool return_weights_and_bias = false);
};
}  // namespace conv2d
}  // namespace operations::conv
}  // namespace ttnn

namespace ttnn {
constexpr auto conv2d = ttnn::register_operation<"ttnn::conv2d", operations::conv::conv2d::Conv2dOperation>();
}
