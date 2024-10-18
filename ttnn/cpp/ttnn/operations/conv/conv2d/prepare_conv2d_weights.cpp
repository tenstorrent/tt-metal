// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv2d.hpp"
#include <sys/types.h>
#include <cstdint>

#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/pool/downsample/device/downsample_op.hpp"
#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/tensor/tensor.hpp"


using namespace tt;
namespace ttnn {
namespace operations::conv {
using sliding_window::SlidingWindowConfig;
using sliding_window::ParallelConfig;

namespace conv2d {

void validate_weight_tensor(const ttnn::Tensor& weight_tensor) {
    TT_ASSERT(!ttnn::has_storage_type_of(weight_tensor, ttnn::DEVICE_STORAGE_TYPE));
    TT_ASSERT(weight_tensor.get_layout() == Layout::ROW_MAJOR);
    TT_ASSERT(weight_tensor.get_shape().rank() == 4);
}

void validate_bias_tensor(const ttnn::Tensor& bias_tensor) {
    TT_ASSERT(!ttnn::has_storage_type_of(bias_tensor, ttnn::DEVICE_STORAGE_TYPE));
    TT_ASSERT(bias_tensor.get_shape().rank() == 4);
    TT_ASSERT(bias_tensor.get_layout() == Layout::ROW_MAJOR);
}

void validate_weights_format(std::string weights_format) {
    TT_FATAL(weights_format.size() == 4, "weights_format must have exactly 4 characters");
    TT_ASSERT(weights_format.find("O") != string::npos, "weights_format must contain \"O\"");
    TT_ASSERT(weights_format.find("I") != string::npos, "weights_format must contain \"I\"");
    TT_ASSERT(weights_format.find("H") != string::npos, "weights_format must contain \"H\"");
    TT_ASSERT(weights_format.find("W") != string::npos, "weights_format must contain \"W\"");
    TT_ASSERT(weights_format == "OIHW", "Conv2d weights format must be \"OIHW\"");
}

template <typename T>
ttnn::Tensor prepare_conv_weights_for_ttnn(
    const ttnn::Tensor& weight_tensor,
    std::string weights_format,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    T *device,
    std::optional<const Conv2dConfig> conv_config_) {

    TT_FATAL(!ttnn::is_tensor_on_device_or_multidevice(weight_tensor), "Error: weight tensor must be on host for preparation.");

    const Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    auto [opt_conv_op_block_config, shard_layout] = get_opt_conv_op_block_config_and_shard_layout(in_channels, out_channels, batch_size, input_height, input_width, kernel_size, stride, padding, dilation, groups, conv_config, device);

    uint32_t weight_block_h_ntiles = opt_conv_op_block_config.act_block_w_ntiles;
    uint32_t weight_block_w_ntiles = opt_conv_op_block_config.out_subblock_w_ntiles;
    uint32_t act_block_h_ntiles = opt_conv_op_block_config.act_block_h_ntiles;

    validate_weight_tensor(weight_tensor);
    ttnn::Tensor weight_tensor_ = weight_tensor;  // tensor to return

    // Permute to OIHW layout as thats what the preparation expects
    validate_weights_format(weights_format);

    auto original_weights_shape = weight_tensor_.get_shape();
    uint32_t original_weights_out_channels = original_weights_shape[0];
    uint32_t original_weights_in_channels = original_weights_shape[1];
    uint32_t original_weights_window_h = original_weights_shape[2];
    uint32_t original_weights_window_w = original_weights_shape[3];

    bool is_conv1d = original_weights_window_w == 1 && input_width == 1;
    bool is_depthwise_conv = groups == original_weights_out_channels && original_weights_in_channels == 1;

    // Convert weight tensor to 0 padded shape if groups > 1
    if (!is_conv1d and groups > 1) {
        weight_tensor_ = tt::tt_metal::convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, groups, conv_config.weights_dtype);
    }
    else if (is_conv1d and groups > 1) {
        if (is_depthwise_conv) {
            weight_tensor_ = convert_conv_weight_tensor_to_depthwise_layout(weight_tensor_, act_block_h_ntiles, conv_config.weights_dtype);
            weight_block_h_ntiles = act_block_h_ntiles;
        }
        else{
           weight_tensor_ = tt::tt_metal::convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, groups, conv_config.weights_dtype);
        }
    }

    uint32_t out_channel_padding = tt::round_up(out_channels, 32) - out_channels;
    tt::tt_metal::LegacyShape weights_channels_padded_shape = tt::tt_metal::LegacyShape(std::array<uint32_t, 4>(
        {tt::round_up(out_channels, 32), tt::round_up(in_channels, conv_config.input_channels_alignment), kernel_size[0], kernel_size[1]}));
    if (conv_config.weights_dtype == DataType::BFLOAT8_B) {
        TT_ASSERT(weight_tensor_.get_dtype() == DataType::FLOAT32);
    } else {
        // TODO: fix the need to check this. We should be able to accept any datatype and convert
        TT_ASSERT(weight_tensor_.get_dtype() == conv_config.weights_dtype);
    }
    weight_tensor_ = ttnn::pad(weight_tensor_, weights_channels_padded_shape.to_array_4D(), tt::tt_metal::Array4D({0, 0, 0, 0}), 0);

    // for conv op, pad the weights to block shape
    if (shard_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        weight_tensor_ = tt::tt_metal::convert_conv_weight_tensor_to_special_padding_tiled_layout(
            weight_tensor_, weight_block_h_ntiles, weight_block_w_ntiles, conv_config.weights_dtype);
    } else {
        weight_tensor_ = tt::tt_metal::convert_conv_weight_tensor_to_tiled_layout(
            weight_tensor_, weight_block_h_ntiles, weight_block_w_ntiles, conv_config.weights_dtype);
    }

    uint32_t weight_matrix_height = in_channels * kernel_size[0] * kernel_size[1];
    int32_t weight_matrix_height_padding = weight_tensor_.shape()[2] - weight_matrix_height;
    TT_FATAL(weight_matrix_height_padding >= 0," Matrix Height Padding can't be negative");

    // convert_conv_weight_tensor adds the padding to the base shape.
    // Reshape the weights to remove padding from the base shape.
    weight_tensor_.set_shape(
        ttnn::Shape(std::array<uint32_t,4>{1, 1, weight_matrix_height, out_channels},
        std::array<std::array<uint32_t, 2>, 4>{
            std::array<uint32_t, 2>{0, 0},
            std::array<uint32_t, 2>{0, 0},
            std::array<uint32_t, 2>{0, weight_matrix_height_padding},
            std::array<uint32_t, 2>{0, out_channel_padding}
    }));
    return weight_tensor_;



}

template <typename T>
ttnn::Tensor prepare_conv_bias_for_ttnn(
    const ttnn::Tensor& bias_tensor,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    T *device,
    std::optional<const Conv2dConfig> conv_config_) {

    TT_FATAL(!ttnn::is_tensor_on_device_or_multidevice(bias_tensor), "Error: bias tensor must be on host for preparation.");

    const Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    auto [opt_conv_op_block_config, _] = get_opt_conv_op_block_config_and_shard_layout(in_channels, out_channels, batch_size, input_height, input_width, kernel_size, stride, padding, dilation, groups, conv_config, device);

    uint32_t weight_block_w_ntiles = opt_conv_op_block_config.out_subblock_w_ntiles;
    validate_bias_tensor(bias_tensor);

    ttnn::Tensor bias_tensor_;
    bias_tensor_ = bias_tensor;
    auto bias_shape = bias_tensor_.get_shape();
    TT_ASSERT(bias_shape[3] == out_channels && bias_shape[0] == 1 && bias_shape[1] == 1 && bias_shape[2] == 1);
    tt::tt_metal::LegacyShape bias_channels_padded_shape = tt::tt_metal::LegacyShape(
        std::array<uint32_t, 4>({1, 1, 32, tt::round_up(out_channels, weight_block_w_ntiles * 32)}));
    bias_tensor_ = ttnn::pad(bias_tensor_, bias_channels_padded_shape.to_array_4D(), tt::tt_metal::Array4D({0, 0, 0, 0}), 0);
    bias_tensor_ = ttnn::to_layout(
        bias_tensor_, Layout::TILE, std::nullopt, std::nullopt, (T*)nullptr);
    if (bias_tensor_.get_dtype() != conv_config.weights_dtype) {
        bias_tensor_ = ttnn::to_dtype(bias_tensor_, conv_config.weights_dtype);
    }
    return bias_tensor_;
}

template ttnn::Tensor prepare_conv_weights_for_ttnn<Device>(
    const ttnn::Tensor& weight_tensor,
    std::string weights_format,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    Device *device,
    std::optional<const Conv2dConfig> conv_config_);

template ttnn::Tensor prepare_conv_weights_for_ttnn<MeshDevice>(
    const ttnn::Tensor& weight_tensor,
    std::string weights_format,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    MeshDevice *device,
    std::optional<const Conv2dConfig> conv_config_);

template ttnn::Tensor prepare_conv_bias_for_ttnn<Device>(
    const ttnn::Tensor& bias_tensor,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    Device *device,
    std::optional<const Conv2dConfig> conv_config_);

template ttnn::Tensor prepare_conv_bias_for_ttnn<MeshDevice>(
    const ttnn::Tensor& bias_tensor,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    MeshDevice *device,
    std::optional<const Conv2dConfig> conv_config_);

}  // namespace conv2d
}  // namespace operations
}  // namespace ttnn
