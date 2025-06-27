// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <utility>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/conv/conv_transpose2d/conv_transpose2d.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv_transpose2d/prepare_conv_transpose2d_weights.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn {
namespace operations::conv {

using namespace tt;
using sliding_window::ParallelConfig;
using sliding_window::SlidingWindowConfig;

namespace conv_transpose2d {

template <typename T>
Result conv_transpose2d(
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
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    const std::optional<const DataType>& dtype,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config,
    bool mirror_kernel,
    bool return_output_dim,
    bool return_weights_and_bias) {
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    const DataType output_dtype = dtype.value_or(input_tensor.dtype());
    DeviceComputeKernelConfig compute_config = compute_config_.value_or(get_conv_default_compute_kernel_config(device));

    // Inverse of sliding_window.get_output_shape()
    SlidingWindowConfig sliding_window_config = SlidingWindowConfig{
        .batch_size = batch_size,
        .input_hw = {input_height, input_width},
        .window_hw = {kernel_size[0], kernel_size[1]},
        .stride_hw = {stride[0], stride[1]},
        .padding = sliding_window::get_pair_n4_padding(padding),
        .output_pad_hw = {output_padding[0], output_padding[1]},
        .dilation_hw = {dilation[0], dilation[1]},
        .is_transpose = true};

    // ConvTranspose2d is implemented via the Conv2d u_op with flipped weights.
    // The input tensor is first passed to the halo op that paddeds the input.
    // In the scenario, where stride > 1, the halo op will add interleaved 0s to the input tensor.
    // The Conv2d u_op is then called with stride = 1, padding = 0.
    // SlidingWindowConfig has a is_transpose flag that is set to true to indicate that the Conv2d u_op & Halo u_op is
    // being called for ConvTranspose2d.
    uint32_t output_height =
        (input_height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1;
    uint32_t output_width =
        (input_width - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1;

    // Dimensions of Input to Conv u_op
    uint32_t full_input_height = output_height + dilation[0] * (kernel_size[0] - 1);
    uint32_t full_input_width = output_width + dilation[1] * (kernel_size[1] - 1);

    // Size of input after adding interleaved 0s.
    uint32_t strided_input_height = (input_height - 1) * stride[0] + 1;
    uint32_t strided_input_width = (input_width - 1) * stride[1] + 1;

    uint32_t input_pad_top = (full_input_height - strided_input_height) / 2;
    uint32_t input_pad_bottom = full_input_height - strided_input_height - input_pad_top;

    uint32_t input_pad_left = (full_input_width - strided_input_width) / 2;
    uint32_t input_pad_right = full_input_width - strided_input_width - input_pad_left;

    log_debug(LogOp, "Input : {}x{}", input_height, input_width);
    log_debug(LogOp, "Output : {}x{}", output_height, output_width);

    log_debug(LogOp, "Conv Op Input : {}x{}", full_input_height, full_input_width);
    log_debug(LogOp, "Strided Input : {}x{}", strided_input_height, strided_input_width);

    log_debug(LogOp, "Padding : ({},{}) ({},{})", input_pad_top, input_pad_bottom, input_pad_left, input_pad_right);

    const bool mm_conv = use_matmul_for_1x1_conv(
        kernel_size,
        {1, 1},
        {input_pad_top + input_pad_bottom, input_pad_left + input_pad_right},
        dilation,
        groups,
        conv_config);

    const auto compute_grid_size = device->compute_with_storage_grid_size();

    bool auto_shard = false;
    if (!input_tensor.is_sharded() && !conv_config.shard_layout.has_value()) {
        if (!conv_config.weights_dtype.has_value()) {
            conv_config.weights_dtype = weight_tensor.dtype();
        }
        // In this case we deduce the shard layout.
        conv_config = determine_conv_config_for_auto_shard(
            conv_config,
            mm_conv,
            batch_size,
            in_channels,
            out_channels,
            output_height,
            output_width,
            weight_tensor.logical_shape()[3],
            full_input_height,
            full_input_width,
            compute_grid_size,
            input_tensor.layout(),
            input_tensor.dtype(),
            output_dtype,
            tt::tt_metal::is_device_tensor(input_tensor) ? std::make_optional(input_tensor.memory_config())
                                                         : std::nullopt,
            kernel_size,
            groups,
            bias_tensor.has_value(),
            compute_config);
        auto_shard = true;
    }

    // Call Halo Transpose
    auto [input_tensor_post_tm, parallel_config, output_parallel_config] = shard_or_reshard_tensor_if_required(
        device,
        input_tensor,
        conv_config,
        batch_size,
        output_height,
        output_width,
        in_channels,
        out_channels,
        mm_conv,
        auto_shard);

    uint32_t round_up_size = tt::constants::TILE_HEIGHT;

    Tensor halo_output;
    if (!mm_conv) {
        sliding_window_config.num_cores_nhw = get_num_cores_nhw_from_parallel_config(parallel_config);
        sliding_window_config.core_range_set = input_tensor_post_tm.memory_config().shard_spec().value().grid;
        sliding_window_config.snap_to_tile = true;

        halo_output = ttnn::halo(
            DefaultQueueId,
            input_tensor_post_tm,
            sliding_window_config,
            0,
            false,
            parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
            input_tensor_post_tm.memory_config());

        if (conv_config.deallocate_activation) {
            input_tensor_post_tm.deallocate(/*force*/ true);
            log_debug(tt::LogOp, "Deallocate Input Tensor");
        }

        if (conv_config.reallocate_halo_output) {
            halo_output = ttnn::move(halo_output);
            log_debug(tt::LogOp, "Reallocate Halo Output");
        }
    }

    // Call Conv2d u_op with Stride = 1, Padding = 0.
    auto conv_out_memory_config = create_sharded_memory_config_from_parallel_config(
        ttnn::Shape({1, 1, batch_size * output_height * output_width, tt::round_up(out_channels, 32)}),
        output_parallel_config,
        round_up_size);

    auto largest_parallel_config = output_parallel_config.grid.num_cores() > parallel_config.grid.num_cores()
                                       ? output_parallel_config
                                       : parallel_config;

    auto opt_conv_op_parallel_config = determine_conv_op_parallel_config_from_conv_output_mem_config(
        conv_out_memory_config,
        get_num_cores_nhw_from_parallel_config(largest_parallel_config),
        get_num_cores_channels_from_parallel_config(largest_parallel_config));

    const uint32_t input_channels_alignment = get_input_channels_alignment(
        input_tensor_post_tm.memory_config().memory_layout(),
        input_tensor.layout(),
        mm_conv,
        input_tensor_post_tm.memory_config());
    uint32_t in_channels_padded = tt::round_up(
        in_channels, get_num_cores_channels_from_parallel_config(parallel_config) * input_channels_alignment);
    uint32_t nhw_out_padded_ntile = get_num_cores_nhw_from_parallel_config(output_parallel_config) *
                                    conv_out_memory_config.shard_spec().value().shape[0] / tt::constants::TILE_HEIGHT;
    auto opt_conv_op_block_config = determine_per_core_conv_block_config(
        parallel_config,
        opt_conv_op_parallel_config,
        in_channels_padded,
        nhw_out_padded_ntile,
        conv_config.act_block_h_override,
        conv_config.act_block_w_div,
        kernel_size[0],
        kernel_size[1],
        get_fp32_dest_acc_en(compute_config),
        conv_config.enable_split_reader,
        conv_config.full_inner_dim);

    bool weight_is_on_device = tt::tt_metal::is_device_tensor(weight_tensor);
    ttnn::Tensor weight_tensor_on_device = weight_tensor;
    std::optional<ttnn::Tensor> bias_tensor_on_device = bias_tensor;
    if (!weight_is_on_device) {
        // prepare weights in desired layout and move to device
        Conv2dWeightsBiasPrepConfig params(
            input_channels_alignment,
            conv_config.weights_dtype,
            opt_conv_op_block_config.act_block_w_ntiles,
            opt_conv_op_block_config.out_subblock_w_ntiles,
            parallel_config,
            output_parallel_config,
            groups,
            opt_conv_op_block_config.act_block_h_ntiles,
            input_width,
            bias_tensor.has_value());
        tie(weight_tensor_on_device, bias_tensor_on_device) = prepare_conv_weights_biases_and_move_to_device(
            transform_weights_for_conv_transpose2d(weight_tensor, mirror_kernel), bias_tensor, params, device);
    }
    if (mm_conv) {
        input_tensor_post_tm =
            ttnn::to_layout(input_tensor_post_tm, Layout::TILE, output_dtype, input_tensor_post_tm.memory_config());
        std::optional<ttnn::operations::matmul::MatmulProgramConfig> program_config = std::nullopt;
        std::optional<MemoryConfig> mm_output_memory_config = std::nullopt;

        if (input_tensor_post_tm.is_sharded()) {
            uint32_t num_cores_c = get_num_cores_channels_from_parallel_config(parallel_config);
            program_config = determine_matmul_op_config_from_conv_op_config(
                opt_conv_op_parallel_config,
                opt_conv_op_block_config,
                parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED,
                conv_config.activation,
                parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
                num_cores_c);
            mm_output_memory_config = conv_out_memory_config;
        }
        Tensor matmul_output = ttnn::linear(
            input_tensor_post_tm,
            weight_tensor_on_device,
            bias_tensor_on_device,
            false,
            false,
            mm_output_memory_config,
            std::nullopt,
            program_config);

        if (memory_config.has_value() && memory_config.value() != matmul_output.memory_config()) {
            matmul_output = ttnn::to_memory_config(matmul_output, memory_config.value(), std::nullopt);
        }
        if (return_output_dim && return_weights_and_bias) {
            return std::tuple(
                matmul_output,
                std::tuple(output_height, output_width),
                std::tuple(weight_tensor_on_device, bias_tensor_on_device));
        } else if (return_output_dim) {
            return std::tuple(matmul_output, std::tuple(output_height, output_width));
        } else if (return_weights_and_bias) {
            return std::tuple(matmul_output, std::tuple(weight_tensor_on_device, bias_tensor_on_device));
        }
        return matmul_output;
    }
    // call conv micro op
    auto conv_output = optimized_conv_new(
        halo_output,
        weight_tensor_on_device,
        bias_tensor_on_device,
        sliding_window_config,
        out_channels,
        groups,
        conv_config.output_layout == Layout::ROW_MAJOR,
        conv_config.activation,
        opt_conv_op_parallel_config,
        opt_conv_op_block_config,
        conv_out_memory_config,
        output_dtype,
        {batch_size, input_height, input_width, in_channels},
        compute_config,
        conv_config.enable_act_double_buffer,
        conv_config.enable_weights_double_buffer,
        conv_config.full_inner_dim,
        conv_config.enable_split_reader,
        conv_config.enable_subblock_padding);
    if (memory_config.has_value() && memory_config.value() != conv_output.memory_config()) {
        conv_output = ttnn::to_memory_config(conv_output, memory_config.value(), std::nullopt);
    }
    if (return_output_dim && return_weights_and_bias) {
        return std::tuple(
            conv_output,
            std::tuple(output_height, output_width),
            std::tuple(weight_tensor_on_device, bias_tensor_on_device));
    } else if (return_output_dim) {
        return std::tuple(conv_output, std::tuple(output_height, output_width));
    } else if (return_weights_and_bias) {
        return std::tuple(conv_output, std::tuple(weight_tensor_on_device, bias_tensor_on_device));
    }
    return conv_output;
}

Result ConvTranpose2dOperation::invoke(
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
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    const std::optional<const DataType>& dtype,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config,
    bool mirror_kernel,
    bool return_output_dim,
    bool return_weights_and_bias) {
    return conv_transpose2d(
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
        output_padding,
        dilation,
        groups,
        std::move(dtype),
        std::move(bias_tensor),
        std::move(conv_config_),
        std::move(compute_config_),
        std::move(memory_config),
        mirror_kernel,
        return_output_dim,
        return_weights_and_bias);
}

Result ConvTranpose2dOperation::invoke(
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
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    const std::optional<const DataType>& dtype,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config,
    bool mirror_kernel,
    bool return_output_dim,
    bool return_weights_and_bias) {
    return conv_transpose2d(
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
        output_padding,
        dilation,
        groups,
        std::move(dtype),
        std::move(bias_tensor),
        std::move(conv_config_),
        std::move(compute_config_),
        std::move(memory_config),
        mirror_kernel,
        return_output_dim,
        return_weights_and_bias);
}

}  // namespace conv_transpose2d
}  // namespace operations::conv
}  // namespace ttnn
