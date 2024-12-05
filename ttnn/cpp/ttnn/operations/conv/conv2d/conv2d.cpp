// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv2d.hpp"
#include "conv2d_utils.hpp"
#include "prepare_conv2d_weights.hpp"
#include <sys/types.h>
#include <cstdint>
#include <optional>
#include <utility>

#include "common/constants.hpp"
#include "common/math.hpp"
#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/pool/downsample/device/downsample_op.hpp"
#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

using namespace tt;
namespace ttnn {
namespace operations::conv {
using sliding_window::SlidingWindowConfig;
using sliding_window::ParallelConfig;

namespace conv2d {

using OutputHeight = uint32_t;
using OutputWidth = uint32_t;
using Result = std::tuple<ttnn::Tensor, OutputHeight, OutputWidth, ttnn::Tensor, std::optional<ttnn::Tensor>>;

template <typename T>
Result conv2d(
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
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const MemoryConfig>& memory_config) {
    const bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding, dilation, groups);
    const uint32_t output_height = ((input_height - kernel_size[0] - ((kernel_size[0] - 1 ) * (dilation[0] - 1)) + 2 * padding[0]) / stride[0]) + 1;
    const uint32_t output_width =
        ((input_width - kernel_size[1] - ((kernel_size[0] - 1) * (dilation[0] - 1)) + 2 * padding[1]) / stride[1]) + 1;

    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    const auto compute_grid_size = device->compute_with_storage_grid_size();

    if (!input_tensor.is_sharded() && !conv_config.shard_layout.has_value()) {
        // In this case we deduce the shard layout.
        adjust_conv_op_config_for_auto_shard_if_necessary(
            mm_conv,
            batch_size,
            in_channels,
            out_channels,
            output_height,
            output_width,
            weight_tensor.get_shape()[3],
            input_width,
            compute_grid_size,
            conv_config,
            input_tensor.layout(),
            ttnn::is_tensor_on_device_or_multidevice(input_tensor) ? std::make_optional(input_tensor.memory_config()) : std::nullopt);
    }

    auto [input_tensor_post_tm, parallel_config, output_parallel_config, tensor_manipulated, use_non_tile_height] = shard_or_reshard_tensor_if_required(
        device, input_tensor, conv_config, batch_size, output_height, output_width, in_channels, out_channels, mm_conv);
    if (tensor_manipulated) {
        if (conv_config.deallocate_activation) {
            ttnn::Tensor input_tensor_ = input_tensor;  // TODO: allow in place modification of inputs to the op
            input_tensor_.deallocate();
            // ttnn::operations::core::deallocate(input_tensor_);
        }
        conv_config.deallocate_activation = true;
    }

    uint32_t round_up_size = !use_non_tile_height ? tt::constants::TILE_HEIGHT : 1;
    uint32_t nhw_out = batch_size * output_height * output_width;
    uint32_t out_channels_padded = tt::round_up(
        out_channels,
        get_num_cores_channels_from_parallel_config(output_parallel_config) * tt::constants::TILE_WIDTH);
    MemoryConfig conv_out_memory_config = create_sharded_memory_config_from_parallel_config(
        ttnn::Shape(std::array<uint32_t, 4>{1, 1, nhw_out, out_channels_padded}),
        output_parallel_config,
        round_up_size);
    ParallelConfig largest_parallel_config = output_parallel_config.grid.num_cores() > parallel_config.grid.num_cores() ? output_parallel_config : parallel_config;

    OptimizedConvParallelizationConfig opt_conv_op_parallel_config = determine_conv_op_parallel_config_from_conv_output_mem_config(
        conv_out_memory_config,
        get_num_cores_nhw_from_parallel_config(largest_parallel_config),
        get_num_cores_channels_from_parallel_config(largest_parallel_config));

    uint32_t in_channels_padded = tt::round_up(
        in_channels,
        get_num_cores_channels_from_parallel_config(parallel_config) * conv_config.input_channels_alignment);

    uint32_t nhw_out_padded_ntile = get_num_cores_nhw_from_parallel_config(output_parallel_config) *
                                    conv_out_memory_config.shard_spec.value().shape[0] / tt::constants::TILE_HEIGHT;

    OptimizedConvBlockConfig opt_conv_op_block_config = determine_per_core_conv_block_config(
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
    bool weight_is_on_device = ttnn::is_tensor_on_device_or_multidevice(weight_tensor);
    ttnn::Tensor weight_tensor_on_device = weight_tensor;
    std::optional<ttnn::Tensor> bias_tensor_on_device = bias_tensor;
    if (!weight_is_on_device) {
        // prepare weights in desired layout and move to device
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
            input_width);
    }
    // if 1x1 conv w/ stride 1, convert input tensor to tile layout if required
    if (mm_conv) {
        Tensor input_tensor_post_tm_out = ttnn::to_layout(
            input_tensor_post_tm, Layout::TILE, conv_config.dtype, input_tensor_post_tm.memory_config(), device);
        if (conv_config.deallocate_activation) {
            input_tensor_post_tm.deallocate();
            // ttnn::operations::core::deallocate(input_tensor_post_tm);
        }
        input_tensor_post_tm = input_tensor_post_tm_out;
    }
    // call optimized conv op or matmul micro op
    bool input_is_on_device = ttnn::is_tensor_on_device_or_multidevice(input_tensor_post_tm);
    TT_ASSERT(input_is_on_device);
    DeviceComputeKernelConfig compute_kernel_config = ttnn::init_device_compute_kernel_config(
        device->arch(),
        std::nullopt,
        conv_config.math_fidelity,
        conv_config.math_approx_mode_enabled,
        conv_config.fp32_dest_acc_enabled,
        conv_config.packer_l1_accum_enabled);

    if (!mm_conv) {
        // call halo op
        SlidingWindowConfig sliding_window_config = SlidingWindowConfig{
            .batch_size = batch_size,
            .input_hw = {input_height, input_width},
            .window_hw = {kernel_size[0], kernel_size[1]},
            .stride_hw = {stride[0], stride[1]},
            .pad_hw = {padding[0], padding[1]},
            .dilation_hw = {dilation[0], dilation[1]},
            .num_cores_nhw = opt_conv_op_parallel_config.num_cores_nhw,
            .core_range_set = input_tensor_post_tm.memory_config().shard_spec.value().grid,
            .snap_to_tile = !use_non_tile_height,
        };

        bool bypass_halo =
            (parallel_config.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED &&
             sliding_window_config.pad_hw.first == 0 && sliding_window_config.pad_hw.second == 0);

        if (bypass_halo) {
            if (input_tensor_post_tm.layout() == Layout::TILE) {
                input_tensor_post_tm = ttnn::to_layout(
                    input_tensor_post_tm, Layout::ROW_MAJOR, std::nullopt, std::nullopt, device);
            }
        } else {
            Tensor halo_output = ttnn::halo(
                DefaultQueueId,
                input_tensor_post_tm,
                sliding_window_config,
                0,
                false,
                parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
                0,
                input_tensor_post_tm.memory_config(),
                !use_non_tile_height);

            if (conv_config.deallocate_activation) {
                ttnn::operations::core::deallocate(input_tensor_post_tm);
            }

            if (conv_config.reallocate_halo_output) {
                auto move_output = ttnn::operations::core::reallocate(halo_output, input_tensor_post_tm.memory_config());
                ttnn::operations::core::deallocate(halo_output);
                halo_output = move_output;
            }
            input_tensor_post_tm = halo_output;
        }

        // call conv micro op
        auto conv_output = optimized_conv_new(
            input_tensor_post_tm,
            weight_tensor_on_device,
            bias_tensor_on_device,
            sliding_window_config,
            out_channels,
            groups,
            conv_config.output_layout == Layout::ROW_MAJOR,
            conv_config.activation == "relu",
            conv_config.math_fidelity,
            opt_conv_op_parallel_config,
            opt_conv_op_block_config,
            conv_out_memory_config,
            conv_config.dtype,
            {batch_size, input_height, input_width, in_channels},
            conv_config.input_channels_alignment == 16,
            compute_kernel_config,
            conv_config.enable_act_double_buffer,
            conv_config.enable_weights_double_buffer,
            conv_config.enable_split_reader,
            conv_config.enable_subblock_padding,
            use_non_tile_height);

        if (memory_config.has_value() && memory_config.value() != conv_output.memory_config()) {
            conv_output = ttnn::to_memory_config(conv_output, memory_config.value(), std::nullopt);
        }
        return {conv_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
    } else {
        // run conv as matmul
        uint32_t num_cores_c = get_num_cores_channels_from_parallel_config(parallel_config);
        auto matmul_program_config = determine_matmul_op_config_from_conv_op_config(
            opt_conv_op_parallel_config,
            opt_conv_op_block_config,
            parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED,
            conv_config.activation,
            parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
            num_cores_c);
        Tensor matmul_input = input_tensor_post_tm;
        if (stride[0] > 1) {
            // run downsample
            matmul_input = ttnn::operations::downsample::downsample(
                input_tensor_post_tm, {batch_size, input_height, input_width, stride[0], stride[1]});
            if (conv_config.deallocate_activation) {
                ttnn::operations::core::deallocate(input_tensor_post_tm);
            }
        }
        auto matmul_output = ttnn::operations::matmul::matmul(
            matmul_input,
            weight_tensor_on_device,
            bias_tensor_on_device,
            ttnn::operations::matmul::Matmul{
            matmul_program_config,
            /*bcast_batch=*/std::nullopt,
            conv_out_memory_config,
            conv_config.dtype,
            compute_kernel_config});
        if (conv_config.deallocate_activation) {
            ttnn::operations::core::deallocate(matmul_input);
        }

        if (memory_config.has_value() && memory_config.value() != matmul_output.memory_config()) {
            matmul_output = ttnn::to_memory_config(matmul_output, memory_config.value(), std::nullopt);
        }

        return {matmul_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
    }
}

Result Conv2dOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    Device * device,
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
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const MemoryConfig>& memory_config){
    return conv2d(input_tensor, weight_tensor, device, in_channels, out_channels, batch_size, input_height, input_width, kernel_size, stride, padding, dilation, groups, std::move(bias_tensor), std::move(conv_config_), memory_config);
}

Result Conv2dOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    MeshDevice * device,
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
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const MemoryConfig>& memory_config){
    return conv2d(input_tensor, weight_tensor, device, in_channels, out_channels, batch_size, input_height, input_width, kernel_size, stride, padding, dilation, groups, std::move(bias_tensor), std::move(conv_config_), memory_config);
}

}  // namespace conv2d
}  // namespace operations
}  // namespace ttnn
