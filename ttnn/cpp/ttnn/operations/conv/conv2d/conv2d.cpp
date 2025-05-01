// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include <tt-metalium/buffer_types.hpp>

#include "tt-metalium/assert.hpp"
#include "tt-metalium/logger.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/experimental/slice_write/slice_write.hpp"
#include "ttnn/types.hpp"
namespace ttnn {
namespace operations::conv {
using namespace tt;
using sliding_window::ParallelConfig;
using sliding_window::SlidingWindowConfig;

namespace conv2d {

ResultWithOptions result_to_result_with_options(
    const Result& result, const bool return_output_dim, const bool return_weights_and_bias) {
    if (return_output_dim && return_weights_and_bias) {
        return std::make_tuple(
            std::get<0>(result),
            std::make_tuple(std::get<1>(result), std::get<2>(result)),
            std::make_tuple(std::get<3>(result), std::get<4>(result)));
    } else if (return_output_dim) {
        return std::make_tuple(std::get<0>(result), std::make_tuple(std::get<1>(result), std::get<2>(result)));
    } else if (return_weights_and_bias) {
        return std::make_tuple(std::get<0>(result), std::make_tuple(std::get<3>(result), std::get<4>(result)));
    }
    return std::get<0>(result);
}

// GCC refuses to create a symbol for this function if it is defined in the source file, leading to linker errors.
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
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config_,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_,
    bool return_output_dim,
    bool return_weights_and_bias) {
    if (dram_slice_config_.has_value()) {
        return result_to_result_with_options(
            conv2d_DRAM(
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
                dram_slice_config_.value()),
            return_output_dim,
            return_weights_and_bias);
    } else {
        return result_to_result_with_options(
            conv2d_L1(
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
                memory_config_),
            return_output_dim,
            return_weights_and_bias);
    }
}

// This function is used for DRAM Slicing
// It divides the output tensor into slices, and calculates the corresponding input slices.
// Uses ttnn::slice to slice the input tensor and bring it to L1.
// Calls conv2d_L1 to perform the convolution on the sliced input tensor.
// Finally, it uses ttnn::experimental::slice_write to write the output tensor back to DRAM.
// The function is called in a loop for each slice of the output tensor.
// The Conv2dSliceConfig is used to determine the slicing configuration. The dimension along which it is sliced, and the
// number of such slices.
// Conv2dConfig does not control the final output, but rather the conv2d_L1 function that is called internally.
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
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config_,
    const Conv2dSliceConfig& dram_slice_config) {
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    std::array<uint32_t, 4> padding_n4 = sliding_window::get_pair_n4_padding(padding);
    const bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding_n4, dilation, groups, conv_config);
    auto [output_height, output_width] =
        calculate_output_image_size({input_height, input_width}, kernel_size, stride, padding_n4, dilation);

    const uint32_t input_sliced_dim =
        dram_slice_config.slice_type == Conv2dSliceConfig::SliceType::HEIGHT ? input_height : input_width;
    const uint32_t output_sliced_dim =
        dram_slice_config.slice_type == Conv2dSliceConfig::SliceType::HEIGHT ? output_height : output_width;
    TT_FATAL(dram_slice_config.num_slices > 1, " Number of slices should be greater than 1 for Conv2D DRAM Slicing");
    TT_FATAL(
        dram_slice_config.num_slices < output_sliced_dim,
        " Number of slices should be less than the dimension being sliced in Conv2D DRAM Slicing");

    ttnn::Tensor input_tensor_on_device;
    if (!is_device_tensor(input_tensor)) {
        input_tensor_on_device = ttnn::operations::core::to_device(input_tensor, device, ttnn::DRAM_MEMORY_CONFIG);
    } else {
        input_tensor_on_device = input_tensor;
    }
    DeviceComputeKernelConfig compute_config = compute_config_.value_or(get_conv_default_compute_kernel_config(device));
    const auto compute_grid_size = device->compute_with_storage_grid_size();

    ttnn::Tensor weight_tensor_on_device;
    std::optional<ttnn::Tensor> bias_tensor_on_device;
    TT_FATAL(input_tensor_on_device.memory_config().is_dram(), "Conv DRAM expects the input tensor to be in DRAM.");
    TT_FATAL(conv_config.dtype != tt::tt_metal::DataType::BFLOAT8_B, "Conv DRAM currently doesn't support BFLOAT8_B");
    TT_FATAL(
        input_tensor_on_device.dtype() == tt_metal::DataType::BFLOAT16, "Input Tensor to Conv DRAM should be BFLOAT16");
    TT_FATAL(
        input_tensor_on_device.layout() == tt_metal::Layout::ROW_MAJOR,
        "Input Tensor to Conv DRAM should be in Row Major Layout");
    TT_FATAL(
        input_tensor_on_device.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Input Tensor to Conv DRAM should be in Interleaved Memory Layout");

    Tensor dram_output_tensor = tt_metal::create_device_tensor(
        TensorSpec(
            ttnn::Shape({batch_size, output_height, output_width, out_channels}),
            tt_metal::TensorLayout(
                conv_config.dtype,
                tt_metal::PageConfig(tt_metal::Layout::ROW_MAJOR),
                MemoryConfig{
                    .memory_layout = TensorMemoryLayout::INTERLEAVED,
                    .buffer_type = BufferType::DRAM,
                })),
        device);

    bool first_run = true;

    const uint32_t min_output_slice_size = output_sliced_dim / dram_slice_config.num_slices;
    const uint32_t output_slice_rem = output_sliced_dim % dram_slice_config.num_slices;

    uint32_t slice_index = 0;
    uint32_t output_slice_dim_start = 0;

    while ((output_slice_dim_start < output_sliced_dim) && (slice_index < dram_slice_config.num_slices)) {
        const uint32_t output_slice_size = min_output_slice_size + ((slice_index < output_slice_rem) ? 1 : 0);
        const uint32_t output_slice_dim_end = std::min(output_sliced_dim, output_slice_dim_start + output_slice_size);
        const uint32_t this_output_slice_dim = output_slice_dim_end - output_slice_dim_start;

        if (this_output_slice_dim == 0) {
            continue;
        }

        uint32_t output_slice_height_start, output_slice_height_end, input_slice_height_start, input_slice_height_end;
        uint32_t output_slice_width_start, output_slice_width_end, input_slice_width_start, input_slice_width_end;
        int pad_top, pad_bottom, pad_left, pad_right;
        if (dram_slice_config.slice_type == Conv2dSliceConfig::SliceType::HEIGHT) {
            output_slice_height_start = output_slice_dim_start;
            output_slice_height_end = output_slice_dim_end;
            output_slice_width_start = 0;
            output_slice_width_end = output_width;
            input_slice_height_start = (output_slice_height_start * stride[0]) - padding_n4[0];
            input_slice_height_end = ((output_slice_height_end - 1) * stride[0]) - padding_n4[0] +
                                     ((kernel_size[0] - 1) * (dilation[0] - 1)) + kernel_size[0];
            input_slice_width_start = 0;
            input_slice_width_end = input_width;
            pad_top = std::max<int>(0, -input_slice_height_start);
            pad_bottom = std::max<int>(0, input_slice_height_end - input_height);
            pad_left = padding_n4[2];
            pad_right = padding_n4[3];

            input_slice_height_start = std::max<int>(0, input_slice_height_start);
            input_slice_height_end = std::min<int>(input_height, input_slice_height_end);
            if (input_slice_height_start >= input_slice_height_end) {
                continue;
            }
        } else {
            output_slice_height_start = 0;
            output_slice_height_end = output_height;
            output_slice_width_start = output_slice_dim_start;
            output_slice_width_end = output_slice_dim_end;
            input_slice_height_start = 0;
            input_slice_height_end = input_height;
            input_slice_width_start = (output_slice_width_start * stride[1]) - padding_n4[2];
            input_slice_width_end = ((output_slice_width_end - 1) * stride[1]) - padding_n4[2] +
                                    ((kernel_size[1] - 1) * (dilation[1] - 1)) + kernel_size[1];

            pad_top = padding_n4[0];
            pad_bottom = padding_n4[1];
            pad_left = std::max<int>(0, -input_slice_width_start);
            pad_right = std::max<int>(0, input_slice_width_end - input_width);
            input_slice_width_start = std::max<int>(0, input_slice_width_start);
            input_slice_width_end = std::min<int>(input_width, input_slice_width_end);

            if (input_slice_width_start >= input_slice_width_end) {
                continue;
            }
        }

        const uint32_t output_slice_height = output_slice_height_end - output_slice_height_start;
        const uint32_t output_slice_width = output_slice_width_end - output_slice_width_start;

        const uint32_t input_slice_height = input_slice_height_end - input_slice_height_start;
        const uint32_t input_slice_width = input_slice_width_end - input_slice_width_start;

        if (!conv_config.shard_layout.has_value()) {
            conv_config = determine_conv_config_for_auto_shard(
                conv_config,
                mm_conv,
                batch_size,
                in_channels,
                out_channels,
                output_slice_height,
                output_slice_width,
                weight_tensor.get_logical_shape()[3],
                input_slice_height,
                input_slice_width,
                compute_grid_size,
                input_tensor_on_device.layout(),
                std::make_optional(input_tensor_on_device.memory_config()),
                kernel_size,
                groups,
                bias_tensor.has_value(),
                compute_config);
        }
        if (!first_run) {
            // After the first run, never preprocess weights.
            TT_ASSERT(
                conv_config.shard_layout.has_value(),
                " Conv2D DRAM Slicing must have fixed a shard layout after the first run.");
            conv_config.always_preprocess_weights = false;
        }
        auto sliced_input_tensor = ttnn::slice(
            queue_id,
            input_tensor_on_device,
            std::array<uint32_t, 4>{0, input_slice_height_start, input_slice_width_start, 0},  // Start
            std::array<uint32_t, 4>{batch_size, input_slice_height_end, input_slice_width_end, in_channels},
            std::array<uint32_t, 4>{
                1,
                1,
                1,
                1,
            }  // Step,
        );
        auto conv_config_l1 = conv_config;
        ttnn::Tensor sliced_output_tensor;
        std::tie(sliced_output_tensor, std::ignore, std::ignore, weight_tensor_on_device, bias_tensor_on_device) =
            conv2d_L1(
                queue_id,
                sliced_input_tensor,
                // TODO: Add check to ensure that the shard_layout and memory_config are the same as the last slice to
                // re-use the weights tensor.
                // TODO: Add caching mechanism for multiple weights tensors, depending on the memory configs.
                first_run ? weight_tensor : weight_tensor_on_device,
                device,
                in_channels,
                out_channels,
                batch_size,
                input_slice_height,
                input_slice_width,
                kernel_size,
                stride,
                std::array<uint32_t, 4>({pad_top, pad_bottom, pad_left, pad_right}),
                dilation,
                groups,
                first_run ? bias_tensor : (std::optional<const ttnn::Tensor>)(bias_tensor_on_device),
                conv_config_l1,
                compute_config_,
                memory_config_);
        if (sliced_output_tensor.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED &&
            sliced_output_tensor.memory_config().memory_layout != TensorMemoryLayout::BLOCK_SHARDED) {
            sliced_output_tensor = ttnn::to_memory_config(
                sliced_output_tensor,
                MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::L1});
        }
        if (sliced_output_tensor.layout() != Layout::ROW_MAJOR) {
            sliced_output_tensor =
                ttnn::to_layout(sliced_output_tensor, Layout::ROW_MAJOR, std::nullopt, std::nullopt, device);
        }
        sliced_output_tensor = ttnn::reshape(
            sliced_output_tensor, ttnn::Shape({batch_size, output_slice_height, output_slice_width, out_channels}));
        ttnn::experimental::slice_write(
            queue_id,
            sliced_output_tensor,
            dram_output_tensor,
            std::array<uint32_t, 4>{0, output_slice_height_start, output_slice_width_start, 0},
            std::array<uint32_t, 4>{batch_size, output_slice_height_end, output_slice_width_end, out_channels},
            std::array<uint32_t, 4>{1, 1, 1, 1});

        first_run = false;
        output_slice_dim_start += output_slice_size;
        slice_index++;
    }

    return {dram_output_tensor, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
}

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
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config) {
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    std::array<uint32_t, 4> padding_n4 = sliding_window::get_pair_n4_padding(padding);
    const bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding_n4, dilation, groups, conv_config);
    auto [output_height, output_width] =
        calculate_output_image_size({input_height, input_width}, kernel_size, stride, padding_n4, dilation);

    DeviceComputeKernelConfig compute_config = compute_config_.value_or(get_conv_default_compute_kernel_config(device));

    const auto compute_grid_size = device->compute_with_storage_grid_size();

    bool auto_shard = false;
    if (!input_tensor.is_sharded() && !conv_config.shard_layout.has_value()) {
        // In this case we deduce the shard layout.
        conv_config = determine_conv_config_for_auto_shard(
            conv_config,
            mm_conv,
            batch_size,
            in_channels,
            out_channels,
            output_height,
            output_width,
            kernel_size[1],
            input_height,
            input_width,
            compute_grid_size,
            input_tensor.layout(),
            tt::tt_metal::is_device_tensor(input_tensor) ? std::make_optional(input_tensor.memory_config())
                                                         : std::nullopt,
            kernel_size,
            groups,
            bias_tensor.has_value(),
            compute_config);
        auto_shard = true;
    }

    ShardOrientation shard_orientation =
        conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;

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

    auto [opt_conv_op_parallel_config, opt_conv_op_block_config, conv_out_memory_config] = get_conv_configs(
        conv_config,
        compute_config,
        parallel_config,
        output_parallel_config,
        in_channels,
        out_channels,
        batch_size,
        output_height,
        output_width,
        kernel_size,
        compute_grid_size);

    bool weight_is_on_device = tt::tt_metal::is_device_tensor(weight_tensor);
    ttnn::Tensor weight_tensor_on_device = weight_tensor;
    std::optional<ttnn::Tensor> bias_tensor_on_device = bias_tensor;
    if (!weight_is_on_device || conv_config.always_preprocess_weights) {
        // prepare weights in desired layout and move to device

        // TODO: Implement heuristic to decide if weights should be preprocessed on device.
        if (!conv_config.preprocess_weights_on_device) {
            tie(weight_tensor_on_device, bias_tensor_on_device) = prepare_conv_weights_biases_and_move_to_device(
                weight_tensor,
                bias_tensor,
                conv_config.input_channels_alignment,
                conv_config.weights_dtype,
                opt_conv_op_block_config.act_block_w_ntiles,
                opt_conv_op_block_config.out_subblock_w_ntiles,
                parallel_config,
                output_parallel_config,
                device,
                groups,
                opt_conv_op_block_config.act_block_h_ntiles,
                input_width,
                bias_tensor.has_value(),
                true);
        } else {
            tie(weight_tensor_on_device, bias_tensor_on_device) = prepare_conv_weights_biases_on_device(
                weight_tensor,
                bias_tensor,
                conv_config.input_channels_alignment,
                conv_config.weights_dtype,
                opt_conv_op_block_config.act_block_w_ntiles,
                opt_conv_op_block_config.out_subblock_w_ntiles,
                parallel_config,
                output_parallel_config,
                device,
                groups,
                opt_conv_op_block_config.act_block_h_ntiles,
                input_width,
                bias_tensor.has_value());
        }
    }

    // call optimized conv op or matmul micro op
    bool input_is_on_device = tt::tt_metal::is_device_tensor(input_tensor_post_tm);
    TT_ASSERT(input_is_on_device);

    if (!mm_conv) {
        // call halo op
        SlidingWindowConfig sliding_window_config = SlidingWindowConfig{
            .batch_size = batch_size,
            .input_hw = {input_height, input_width},
            .window_hw = {kernel_size[0], kernel_size[1]},
            .stride_hw = {stride[0], stride[1]},
            .padding = {{padding_n4[0], padding_n4[1], padding_n4[2], padding_n4[3]}},
            .dilation_hw = {dilation[0], dilation[1]},
            .num_cores_nhw = opt_conv_op_parallel_config.num_cores_nhw,
            .core_range_set = input_tensor_post_tm.memory_config().shard_spec.value().grid,
            .snap_to_tile = true,
        };

        bool bypass_halo =
            (parallel_config.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED &&
             sliding_window_config.get_pad_h() == 0 && sliding_window_config.get_pad_w() == 0);

        if (bypass_halo) {
            if (input_tensor_post_tm.layout() == Layout::TILE) {
                // Reshape is used as a workaround to an issue in to_layout mentioned here :
                // https://github.com/tenstorrent/tt-metal/issues/16330
                input_tensor_post_tm = ttnn::reshape(input_tensor_post_tm, input_tensor_post_tm.get_padded_shape());
                input_tensor_post_tm =
                    ttnn::to_layout(input_tensor_post_tm, Layout::ROW_MAJOR, std::nullopt, std::nullopt, device);
            }
        } else {
            Tensor halo_output = ttnn::halo(
                queue_id,
                input_tensor_post_tm,
                sliding_window_config,
                0,
                false,
                parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
                input_tensor_post_tm.memory_config(),
                true,
                conv_config.in_place);

            if (conv_config.deallocate_activation) {
                input_tensor_post_tm.deallocate(/*force*/ true);
            }

            input_tensor_post_tm = std::move(halo_output);

            if (conv_config.reallocate_halo_output) {
                input_tensor_post_tm = ttnn::move(input_tensor_post_tm);
            }
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
            conv_config.activation,
            opt_conv_op_parallel_config,
            opt_conv_op_block_config,
            conv_out_memory_config,
            conv_config.dtype,
            {batch_size, input_height, input_width, in_channels},
            compute_config,
            conv_config.enable_act_double_buffer,
            conv_config.enable_weights_double_buffer,
            conv_config.enable_split_reader);

        if (memory_config.has_value() && memory_config.value() != conv_output.memory_config()) {
            conv_output = ttnn::to_memory_config(conv_output, memory_config.value(), std::nullopt);
        }
        return {conv_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
    } else {
        if (input_tensor_post_tm.layout() != Layout::TILE) {
            input_tensor_post_tm =
                ttnn::to_layout(input_tensor_post_tm, Layout::TILE, std::nullopt, std::nullopt, device);
        }

        // run conv as matmul
        std::optional<ttnn::operations::matmul::MatmulProgramConfig> program_config = std::nullopt;
        std::optional<MemoryConfig> mm_output_memory_config = std::nullopt;
        std::optional<std::string> activation = std::nullopt;

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
        } else {
            if (!conv_config.activation.empty()) {
                activation = conv_config.activation;
            }
        }
        Tensor matmul_output = ttnn::linear(
            input_tensor_post_tm,
            weight_tensor_on_device,
            bias_tensor_on_device,
            false,
            false,
            mm_output_memory_config,
            conv_config.dtype,
            program_config,
            activation);

        if (memory_config.has_value() && memory_config.value() != matmul_output.memory_config()) {
            matmul_output = ttnn::to_memory_config(matmul_output, memory_config.value(), std::nullopt);
        }

        return {matmul_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
    }
}

template ResultWithOptions conv2d<IDevice>(
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
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config_,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_,
    bool return_output_dim,
    bool return_weights_and_bias);

template ResultWithOptions conv2d<MeshDevice>(
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
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config_,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_,
    bool return_output_dim,
    bool return_weights_and_bias);

ResultWithOptions Conv2dOperation::invoke(
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
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config_,
    const std::optional<const Conv2dSliceConfig>& slice_config_,
    bool return_output_dim,
    bool return_weights_and_bias) {
    return conv2d(
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
        std::move(bias_tensor),
        std::move(conv_config_),
        std::move(compute_config_),
        std::move(memory_config_),
        std::move(slice_config_),
        return_output_dim,
        return_weights_and_bias);
}

ResultWithOptions Conv2dOperation::invoke(
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
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const Conv2dSliceConfig>& slice_config_,
    bool return_output_dim,
    bool return_weights_and_bias) {
    return conv2d(
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
        std::move(bias_tensor),
        std::move(conv_config_),
        std::move(compute_config_),
        std::move(memory_config),
        std::move(slice_config_),
        return_output_dim,
        return_weights_and_bias);
}

}  // namespace conv2d
}  // namespace operations::conv
}  // namespace ttnn
