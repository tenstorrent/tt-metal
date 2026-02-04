// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <variant>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"

namespace ttnn::operations::conv::conv_transpose2d {

using ttnn::prim::Conv2dConfig;
using ttnn::prim::Conv2dSliceConfig;

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
struct ConvTranpose2dOperation {
    static ResultWithOptions invoke(
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
        std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding = std::array<uint32_t, 4>{0, 0, 0, 0},
        std::array<uint32_t, 2> output_padding = std::array<uint32_t, 2>{0, 0},
        std::array<uint32_t, 2> dilation = std::array<uint32_t, 2>{1, 1},
        uint32_t groups = 1,
        const std::optional<const DataType>& dtype = std::nullopt,
        const std::optional<const ttnn::Tensor>& bias_tensor = std::nullopt,
        const std::optional<const Conv2dConfig>& conv_config_ = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig>& compute_config_ = std::nullopt,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const Conv2dSliceConfig>& dram_slice_config_ = std::nullopt,
        bool mirror_kernel = true,
        bool return_output_dim = false,
        bool return_weights_and_bias = false);
};

std::unique_ptr<op_slicing::OpSliceAttr> get_conv_transpose2d_slice_attr(
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t in_channels,
    uint32_t out_channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 4> padding_n4,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    Layout input_layout,
    DataType input_dtype,
    DataType conv_output_dtype,
    Tensor& weight_tensor,
    std::optional<std::reference_wrapper<Tensor>> bias_tensor,
    const Conv2dConfig& conv_config_,
    const DeviceComputeKernelConfig& compute_config,
    MeshDevice* device,
    bool mirror_kernel);

}  // namespace ttnn::operations::conv::conv_transpose2d
namespace ttnn {
constexpr auto conv_transpose2d =
    ttnn::register_operation<"ttnn::conv_transpose2d", operations::conv::conv_transpose2d::ConvTranpose2dOperation>();
}
