// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prepare_conv2d_weights.hpp"
#include "conv2d_utils.hpp"
#include <sys/types.h>
#include <cstdint>

using namespace tt;
namespace ttnn {
namespace operations::conv {
using sliding_window::SlidingWindowConfig;
using sliding_window::ParallelConfig;

namespace conv2d {

void validate_weight_and_bias_tensors(
    const ttnn::Tensor& weight_tensor, std::optional<const ttnn::Tensor>& bias_tensor) {
    TT_ASSERT(!ttnn::has_storage_type_of(weight_tensor, ttnn::DEVICE_STORAGE_TYPE));
    TT_ASSERT(weight_tensor.get_layout() == Layout::ROW_MAJOR);
    TT_ASSERT(weight_tensor.get_shape().rank() == 4);
    // TODO: enable this assert
    // TT_ASSERT(weight_tensor.get_shape() == weight_tensor.get_legacy_shape());
    if (bias_tensor.has_value()) {
        TT_ASSERT(!ttnn::has_storage_type_of(bias_tensor.value(), ttnn::DEVICE_STORAGE_TYPE));
        TT_ASSERT(bias_tensor.value().get_shape().rank() == 4);
        TT_ASSERT(bias_tensor.value().get_layout() == Layout::ROW_MAJOR);
        // TODO: enable this assert
        // TT_ASSERT(bias_tensor.value().get_shape() == bias_tensor.value().get_legacy_shape());
    }
}


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

void validate_weights_format(const std::string& weights_format) {
    TT_FATAL(weights_format.size() == 4, "weights_format must have exactly 4 characters");
    TT_ASSERT(weights_format.find("O") != string::npos, "weights_format must contain \"O\"");
    TT_ASSERT(weights_format.find("I") != string::npos, "weights_format must contain \"I\"");
    TT_ASSERT(weights_format.find("H") != string::npos, "weights_format must contain \"H\"");
    TT_ASSERT(weights_format.find("W") != string::npos, "weights_format must contain \"W\"");
    TT_ASSERT(weights_format == "OIHW", "Conv2d weights format must be \"OIHW\"");
}

template <typename T>
OptimizedConvBlockConfig get_opt_block_config(
    bool mm_conv,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t batch_size,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    T *device,
    Conv2dConfig& conv_config,
    Layout input_tensor_layout,
    const MemoryConfig& input_memory_config) {
    auto compute_grid_size = device->compute_with_storage_grid_size();

    adjust_conv_op_config_for_auto_shard_if_necessary(
        mm_conv,
        batch_size,
        in_channels,
        out_channels,
        output_height,
        output_width,
        kernel_size[1],
        input_width,
        device->compute_with_storage_grid_size(),
        conv_config,
        input_tensor_layout,
        input_memory_config);

    ShardOrientation shard_orientation =
        conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;

    bool use_non_tile_height = conv_config.shard_layout.value() == TensorMemoryLayout::HEIGHT_SHARDED && out_channels <= 256 && conv_config.act_block_h_override == 0 &&
        (conv_config.dtype == DataType::BFLOAT16 || conv_config.dtype == DataType::FLOAT32) && conv_config.output_layout == Layout::ROW_MAJOR;
    use_non_tile_height = use_non_tile_height && conv_config.input_channels_alignment != 16;

    ParallelConfig parallel_config = determine_parallel_config(
        conv_config.shard_layout.value(),
        batch_size,
        in_channels,
        output_height,
        output_width,
        out_channels,
        device->compute_with_storage_grid_size(),
        shard_orientation,
        !use_non_tile_height);

    auto output_parallel_config = parallel_config;
    if(conv_config.shard_layout.value() == ttnn::TensorMemoryLayout::WIDTH_SHARDED && !mm_conv) {
        uint32_t max_num_cores = compute_grid_size.x * compute_grid_size.y;
        output_parallel_config = {
            .grid = num_cores_to_corerangeset( find_closest_largest_divisor(tt::div_up(out_channels, tt::constants::TILE_WIDTH),max_num_cores), compute_grid_size, true),
            .shard_scheme = ttnn::TensorMemoryLayout::WIDTH_SHARDED,
            .shard_orientation = parallel_config.shard_orientation
        };
        log_debug(tt::LogOp, "Changing width sharded output grid to  {}",output_parallel_config.grid);
    }

    uint32_t round_up_size = !use_non_tile_height ? tt::constants::TILE_HEIGHT : 1;
    auto conv_out_memory_config = create_sharded_memory_config_from_parallel_config(
        ttnn::Shape(std::array<uint32_t, 4>{1, 1, batch_size * output_height * output_width, tt::round_up(out_channels, 32)}),
        output_parallel_config, round_up_size);
    auto largest_parallel_config = output_parallel_config.grid.num_cores() > parallel_config.grid.num_cores() ? output_parallel_config : parallel_config;
    auto opt_conv_op_parallel_config = determine_conv_op_parallel_config_from_conv_output_mem_config(
        conv_out_memory_config, get_num_cores_nhw_from_parallel_config(largest_parallel_config),
        get_num_cores_channels_from_parallel_config(parallel_config));

    uint32_t in_channels_padded = tt::round_up(
        in_channels,
        get_num_cores_channels_from_parallel_config(parallel_config) * conv_config.input_channels_alignment);

    uint32_t nhw_out_padded_ntile = get_num_cores_nhw_from_parallel_config(output_parallel_config) *
        conv_out_memory_config.shard_spec.value().shape[0] / tt::constants::TILE_HEIGHT;

    return determine_per_core_conv_block_config(
        parallel_config,
        opt_conv_op_parallel_config,
        in_channels_padded,
        nhw_out_padded_ntile,
        conv_config.act_block_h_override,
        conv_config.act_block_w_div,
        kernel_size[0],
        kernel_size[1],
        conv_config.fp32_dest_acc_enabled,
        conv_config.enable_split_reader);
}



template <typename T>
std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_and_move_to_device(
    const ttnn::Tensor& weight_tensor,
    std::optional<const ttnn::Tensor>& bias_tensor,
    uint32_t input_channels_alignment,
    DataType weights_bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const ParallelConfig& parallel_config,
    T * device,
    uint32_t groups,
    uint32_t act_block_h_ntiles,
    uint32_t input_width,
    const bool parameters_on_device) {

    validate_weight_and_bias_tensors(weight_tensor, bias_tensor);
    ttnn::Tensor weight_tensor_;  // tensor to return
    ttnn::Tensor bias_tensor_;

    auto original_weights_shape = weight_tensor.get_shape();
    uint32_t original_weights_out_channels = original_weights_shape[0];
    uint32_t original_weights_in_channels = original_weights_shape[1];
    uint32_t original_weights_window_h = original_weights_shape[2];
    uint32_t original_weights_window_w = original_weights_shape[3];

    bool is_conv1d = original_weights_window_w == 1 && input_width == 1;
    bool is_depthwise_conv = groups == original_weights_out_channels && original_weights_in_channels == 1;

    weight_tensor_ = weight_tensor;

    // Convert weight tensor to 0 padded shape if groups > 1
    if (!is_conv1d and groups > 1) {
        weight_tensor_ = tt::tt_metal::convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, groups, weights_bias_dtype);
    }
    else if (is_conv1d and groups > 1) {
        if (is_depthwise_conv) {
            weight_tensor_ = convert_conv_weight_tensor_to_depthwise_layout(weight_tensor_, act_block_h_ntiles, weights_bias_dtype);
            weight_block_h_ntiles = act_block_h_ntiles;
        }
        else{
           weight_tensor_ = tt::tt_metal::convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, groups, weights_bias_dtype);
        }
    }

    auto weights_shape = weight_tensor_.get_shape();
    uint32_t out_channels = weights_shape[0];
    uint32_t in_channels = weights_shape[1];
    uint32_t window_h = weights_shape[2];
    uint32_t window_w = weights_shape[3];

    uint32_t num_cores_channels = get_num_cores_channels_from_parallel_config(parallel_config);
    uint32_t out_channels_padded = tt::round_up(out_channels, num_cores_channels * tt::constants::TILE_WIDTH);
    uint32_t in_channels_padded = tt::round_up(in_channels, num_cores_channels * input_channels_alignment);
    uint32_t out_channel_padding = out_channels_padded - out_channels;

    tt::tt_metal::LegacyShape weights_channels_padded_shape = tt::tt_metal::LegacyShape(std::array<uint32_t, 4>(
        {out_channels_padded, in_channels_padded, window_h, window_w}));
    if (weights_bias_dtype == DataType::BFLOAT8_B) {
        TT_ASSERT(weight_tensor_.get_dtype() == DataType::FLOAT32);
        if (bias_tensor.has_value()) {
            TT_ASSERT(bias_tensor.value().get_dtype() == DataType::FLOAT32);
        }
    } else {
        // TODO: fix the need to check this. We should be able to accept any datatype and convert
        TT_ASSERT(weight_tensor_.get_dtype() == weights_bias_dtype);
        if (bias_tensor.has_value()) {
            TT_ASSERT(bias_tensor.value().get_dtype() == weights_bias_dtype);
        }
    }
    weight_tensor_ = ttnn::pad(weight_tensor_, weights_channels_padded_shape.to_array_4D(), tt::tt_metal::Array4D({0, 0, 0, 0}), 0);

    // for conv op, pad the weights to block shape
    if (parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
        weight_tensor_ = tt::tt_metal::convert_conv_weight_tensor_to_special_padding_tiled_layout(
            weight_tensor_, weight_block_h_ntiles, weight_block_w_ntiles, weights_bias_dtype);
    } else {
        weight_tensor_ = tt::tt_metal::convert_conv_weight_tensor_to_tiled_layout(
            weight_tensor_, weight_block_h_ntiles, weight_block_w_ntiles, weights_bias_dtype);
    }

    uint32_t weight_matrix_height = in_channels * window_h * window_w;
    int32_t weight_matrix_height_padding = weight_tensor_.shape()[2] - weight_matrix_height;
    TT_FATAL(weight_matrix_height_padding >= 0," Matrix Height Padding can't be negative");

    auto target_shape = ttnn::Shape(std::array<uint32_t,4>{1, 1, weight_matrix_height, out_channels},
        std::array<std::array<uint32_t, 2>, 4>{
            std::array<uint32_t, 2>{0, 0},
            std::array<uint32_t, 2>{0, 0},
            std::array<uint32_t, 2>{0, weight_matrix_height_padding},
            std::array<uint32_t, 2>{0, out_channel_padding}
    });
    weight_tensor_ = ttnn::reshape(weight_tensor_, target_shape);

    if(parameters_on_device)
        weight_tensor_ = ttnn::operations::core::to_device(weight_tensor_, device, std::nullopt);

    if (bias_tensor.has_value()) {
        bias_tensor_ = bias_tensor.value();
        auto bias_shape = bias_tensor_.get_shape();
        TT_ASSERT(bias_shape[3] == out_channels && bias_shape[0] == 1 && bias_shape[1] == 1 && bias_shape[2] == 1);
        tt::tt_metal::LegacyShape bias_channels_padded_shape = tt::tt_metal::LegacyShape(
            std::array<uint32_t, 4>({1, 1, 32, tt::round_up(out_channels, weight_block_w_ntiles * 32)}));
        bias_tensor_ = ttnn::pad(bias_tensor_, bias_channels_padded_shape.to_array_4D(), tt::tt_metal::Array4D({0, 0, 0, 0}), 0);
        bias_tensor_ = ttnn::to_layout(
            bias_tensor_, Layout::TILE, std::nullopt, std::nullopt, (T*)nullptr);
        if (bias_tensor_.get_dtype() != weights_bias_dtype) {
            bias_tensor_ = ttnn::to_dtype(bias_tensor_, weights_bias_dtype);
        }
        if(parameters_on_device)
            bias_tensor_ = ttnn::operations::core::to_device(bias_tensor_, device, std::nullopt);
    }

    return {weight_tensor_, bias_tensor.has_value() ? bias_tensor_ : std::optional<ttnn::Tensor>()};
}

template <typename T>
ttnn::Tensor prepare_conv_weights(
    const ttnn::Tensor& weight_tensor,
    const ttnn::MemoryConfig &input_memory_config,
    Layout input_tensor_layout,
    const std::string& weights_format,
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
    const std::optional<const Conv2dConfig>& conv_config_) {
    TT_FATAL(!ttnn::is_tensor_on_device_or_multidevice(weight_tensor), "Error: weight tensor must be on host for preparation.");
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    const bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding, dilation, groups);
    const uint32_t output_height = ((input_height - kernel_size[0] - ((kernel_size[0] - 1 ) * (dilation[0] - 1)) + 2 * padding[0]) / stride[0]) + 1;
    const uint32_t output_width =
        ((input_width - kernel_size[1] - ((kernel_size[0] - 1) * (dilation[0] - 1)) + 2 * padding[1]) / stride[1]) + 1;
    auto opt_conv_op_block_config = get_opt_block_config(
        mm_conv,
        in_channels,
        out_channels,
        output_height,
        output_width,
        batch_size,
        input_width,
        kernel_size,
        stride,
        device,
        conv_config,
        input_tensor_layout,
        input_memory_config
    );

    ShardOrientation shard_orientation =
        conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;

    bool use_non_tile_height = conv_config.shard_layout.value() == TensorMemoryLayout::HEIGHT_SHARDED && out_channels <= 256 && conv_config.act_block_h_override == 0 &&
        (conv_config.dtype == DataType::BFLOAT16 || conv_config.dtype == DataType::FLOAT32) && conv_config.output_layout == Layout::ROW_MAJOR;
    use_non_tile_height = use_non_tile_height && conv_config.input_channels_alignment != 16;

    ParallelConfig parallel_config = determine_parallel_config(
        conv_config.shard_layout.value(),
        batch_size,
        in_channels,
        output_height,
        output_width,
        out_channels,
        device->compute_with_storage_grid_size(),
        shard_orientation,
        !use_non_tile_height);

    std::optional<const ttnn::Tensor> bias_tensor = std::nullopt;
    ttnn::Tensor weight_tensor_on_device = weight_tensor;
    std::optional<ttnn::Tensor> bias_tensor_on_device = bias_tensor;
    tie(weight_tensor_on_device, bias_tensor_on_device) = prepare_conv_weights_biases_and_move_to_device(
        weight_tensor,
        bias_tensor,
        conv_config.input_channels_alignment,
        conv_config.weights_dtype,
        opt_conv_op_block_config.act_block_w_ntiles,
        opt_conv_op_block_config.out_subblock_w_ntiles,
        parallel_config,
        device,
        groups,
        opt_conv_op_block_config.act_block_h_ntiles,
        input_width,
        false);

    return weight_tensor_on_device;
}

template <typename T>
ttnn::Tensor prepare_conv_bias(
    const ttnn::Tensor& bias_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_tensor_layout,
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
    const std::optional<const Conv2dConfig>& conv_config_) {

    TT_FATAL(!ttnn::is_tensor_on_device_or_multidevice(bias_tensor), "Error: bias tensor must be on host for preparation.");

    const bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding, dilation, groups);
    const uint32_t output_height = ((input_height - kernel_size[0] - ((kernel_size[0] - 1 ) * (dilation[0] - 1)) + 2 * padding[0]) / stride[0]) + 1;
    const uint32_t output_width =
        ((input_width - kernel_size[1] - ((kernel_size[0] - 1) * (dilation[0] - 1)) + 2 * padding[1]) / stride[1]) + 1;

    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    auto opt_conv_op_block_config = get_opt_block_config(
        mm_conv,
        in_channels,
        out_channels,
        output_height,
        output_width,
        batch_size,
        input_width,
        kernel_size,
        stride,
        device,
        conv_config,
        input_tensor_layout,
        input_memory_config
    );

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

template OptimizedConvBlockConfig get_opt_block_config<Device>(
    bool mm_conv,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t batch_size,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    Device *device,
    Conv2dConfig& conv_config,
    Layout input_tensor_layout,
    const ttnn::MemoryConfig& input_memory_config);

template OptimizedConvBlockConfig get_opt_block_config<MeshDevice>(
    bool mm_conv,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t batch_size,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    MeshDevice *device,
    Conv2dConfig& conv_config,
    Layout input_tensor_layout,
    const ttnn::MemoryConfig& input_memory_config);

template ttnn::Tensor prepare_conv_weights<Device>(
    const ttnn::Tensor& weight_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_tensor_layout,
    const std::string& weights_format,
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
    const std::optional<const Conv2dConfig>& conv_config_);

template ttnn::Tensor prepare_conv_weights<MeshDevice>(
    const ttnn::Tensor& weight_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_tensor_layout,
    const std::string& weights_format,
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
    const std::optional<const Conv2dConfig>& conv_config_);

template std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_and_move_to_device<Device>(
    const ttnn::Tensor& weight_tensor,
    std::optional<const ttnn::Tensor>& bias_tensor,
    uint32_t input_channels_alignment,
    DataType weights_bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const ParallelConfig& parallel_config,
    Device* device,
    uint32_t groups,
    uint32_t act_block_h_ntiles,
    uint32_t input_width,
    const bool parameters_on_device);

template std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_and_move_to_device<MeshDevice>(
    const ttnn::Tensor& weight_tensor,
    std::optional<const ttnn::Tensor>& bias_tensor,
    uint32_t input_channels_alignment,
    DataType weights_bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const ParallelConfig& parallel_config,
    MeshDevice* device,
    uint32_t groups,
    uint32_t act_block_h_ntiles,
    uint32_t input_width,
    const bool parameters_on_device);

template ttnn::Tensor prepare_conv_bias<Device>(
    const ttnn::Tensor& bias_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_tensor_layout,
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
    const std::optional<const Conv2dConfig>& conv_config_);

template ttnn::Tensor prepare_conv_bias<MeshDevice>(
    const ttnn::Tensor& bias_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_tensor_layout,
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
    const std::optional<const Conv2dConfig>& conv_config_);

}  // namespace conv2d
}  // namespace operations
}  // namespace ttnn
