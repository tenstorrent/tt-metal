// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include <tt-metalium/buffer_types.hpp>

#include "tt-metalium/assert.hpp"
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/math.hpp"
#include <tt_stl/small_vector.hpp>
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/tensor/enum_types.hpp"
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
#include "ttnn/operations/data_movement/fold/fold.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/experimental/slice_write/slice_write.hpp"
#include "ttnn/operations/experimental/padded_slice/padded_slice.hpp"
#include "ttnn/operations/creation.hpp"
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
    const std::optional<const DataType>& dtype,
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
                dtype,
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
                dtype,
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
    const std::optional<const DataType>& dtype,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config_,
    const Conv2dSliceConfig& dram_slice_config) {
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    const DataType output_dtype = dtype.value_or(input_tensor.dtype());
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

    const auto unflattened_input_shape = ttnn::Shape{batch_size, input_height, input_width, in_channels};
    input_tensor_on_device = ttnn::reshape(input_tensor_on_device, unflattened_input_shape, unflattened_input_shape);

    DeviceComputeKernelConfig compute_config = compute_config_.value_or(get_conv_default_compute_kernel_config(device));
    const auto compute_grid_size = device->compute_with_storage_grid_size();

    ttnn::Tensor weight_tensor_on_device;
    std::optional<ttnn::Tensor> bias_tensor_on_device;
    TT_FATAL(!memory_config_.has_value(), "Setting Memory config for Conv2D with DRAM Slicing is not supported.");
    TT_FATAL(input_tensor_on_device.memory_config().is_dram(), "Conv DRAM expects the input tensor to be in DRAM.");
    TT_FATAL(
        !(conv_config.output_layout == Layout::ROW_MAJOR && output_dtype == DataType::BFLOAT8_B),
        "Conv output can't be in Row Major if output dtype is BFloat8_B.");

    TT_FATAL(
        input_tensor_on_device.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input Tensor to Conv DRAM should be in Interleaved Memory Layout");

    Tensor dram_output_tensor = tt_metal::create_device_tensor(
        TensorSpec(
            ttnn::Shape({batch_size, output_height, output_width, out_channels}),
            tt_metal::TensorLayout(
                output_dtype,
                tt_metal::PageConfig(conv_config.output_layout),
                MemoryConfig{
                    TensorMemoryLayout::INTERLEAVED,
                    BufferType::DRAM,
                })),
        device);

    uint32_t slice_rounding_value = 1;
    if (conv_config.output_layout == tt_metal::Layout::TILE) {
        // In Conv2d DRAM with Outputs in Tile layout, we need to round the slice size to a multiple of TILE_HEIGHT.
        slice_rounding_value = tt::constants::TILE_HEIGHT;
    }

    bool first_run = true;
    const uint32_t min_output_slice_size =
        tt::div_up(output_sliced_dim, slice_rounding_value) / dram_slice_config.num_slices;
    const uint32_t output_slice_rem =
        tt::div_up(output_sliced_dim, slice_rounding_value) % dram_slice_config.num_slices;

    uint32_t slice_index = 0;
    uint32_t output_slice_dim_start = 0;

    uint32_t additional_padded_width = 0;

    while ((output_slice_dim_start < output_sliced_dim) && (slice_index < dram_slice_config.num_slices)) {
        const uint32_t output_slice_size =
            slice_rounding_value * (min_output_slice_size + ((slice_index < output_slice_rem) ? 1 : 0));
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

        log_trace(
            LogOp,
            "Conv2d DRAM Slicing: Slice {}: Output Slice Start: ({}, {}), End: ({}, {})",
            slice_index,
            output_slice_height_start,
            output_slice_width_start,
            output_slice_height_end,
            output_slice_width_end);
        log_trace(
            LogOp,
            "Conv2d DRAM Slicing: Slice {}: Input Slice Start: ({}, {}), End: ({}, {})",
            slice_index,
            input_slice_height_start,
            input_slice_width_start,
            input_slice_height_end,
            input_slice_width_end);

        const uint32_t input_slice_height = input_slice_height_end - input_slice_height_start;
        const uint32_t input_slice_width = input_slice_width_end - input_slice_width_start;

        const uint32_t output_slice_height = output_slice_height_end - output_slice_height_start;

        uint32_t output_slice_width = output_slice_width_end - output_slice_width_start;
        if (output_slice_width % slice_rounding_value != 0) {
            additional_padded_width = slice_rounding_value - (output_slice_width % slice_rounding_value);
            log_trace(
                LogOp,
                "Conv2d DRAM Slicing: Slice {}: Additional padding of {} added to the right side.",
                slice_index,
                additional_padded_width);
            pad_right += additional_padded_width * stride[1];
            output_slice_width += additional_padded_width;
        }

        log_debug(
            tt::LogOp,
            "Input Slice : {} x {}, Output Slice {} x {}",
            input_slice_height,
            input_slice_width,
            output_slice_height,
            output_slice_width);

        if (!conv_config.shard_layout.has_value()) {
            if (!conv_config.weights_dtype.has_value()) {
                conv_config.weights_dtype = weight_tensor.dtype();
            }
            conv_config = determine_conv_config_for_auto_shard(
                conv_config,
                mm_conv,
                batch_size,
                in_channels,
                out_channels,
                output_slice_height,
                output_slice_width,
                weight_tensor.logical_shape()[3],
                input_slice_height,
                input_slice_width,
                compute_grid_size,
                input_tensor_on_device.layout(),
                input_tensor_on_device.dtype(),
                output_dtype,
                std::make_optional(input_tensor_on_device.memory_config()),
                kernel_size,
                groups,
                bias_tensor.has_value(),
                compute_config);
        }

        TT_FATAL(conv_config.shard_layout.has_value(), " Conv2D DRAM Slicing must have a shard layout set.");

        Tensor sliced_input_tensor;
        if (conv_config.shard_layout.value() == TensorMemoryLayout::WIDTH_SHARDED) {
            sliced_input_tensor = ttnn::slice(
                queue_id,
                input_tensor_on_device,
                ttnn::SmallVector<uint32_t>{0, input_slice_height_start, input_slice_width_start, 0},  // Start
                ttnn::SmallVector<uint32_t>{batch_size, input_slice_height_end, input_slice_width_end, in_channels},
                ttnn::SmallVector<uint32_t>{1, 1, 1, 1}  // Step
            );
        } else {
            auto sliced_input_tensor_memory_config = std::get<1>(determine_input_memory_config(
                conv_config,
                batch_size,
                ttnn::Shape({batch_size, input_slice_height, input_slice_width, in_channels}),
                ttnn::Shape({batch_size, output_slice_height, output_slice_width, out_channels}),
                mm_conv,
                device,
                // Setting layout to TILE forces input_channels_alignment to 32.
                //  The padded_slice op needs aligned reads from L1.
                Layout::TILE));

            sliced_input_tensor = ttnn::experimental::padded_slice(
                queue_id,
                input_tensor_on_device,
                ttnn::SmallVector<uint32_t>{0, input_slice_height_start, input_slice_width_start, 0},  // Start
                ttnn::SmallVector<uint32_t>{batch_size, input_slice_height_end, input_slice_width_end, in_channels},
                ttnn::SmallVector<uint32_t>{1, 1, 1, 1},  // Step
                sliced_input_tensor_memory_config);
        }
        auto conv_config_l1 = conv_config;
        conv_config_l1.deallocate_activation = true;
        conv_config_l1.reallocate_halo_output = true;

        // Force Conv2d_L1 to always output tiled layout to reduce CB Memory usage.
        conv_config_l1.output_layout = Layout::TILE;

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
                output_dtype,
                first_run ? bias_tensor : (std::optional<const ttnn::Tensor>)(bias_tensor_on_device),
                conv_config_l1,
                compute_config_,
                memory_config_);

        // slice_write supports all sharding layouts for tiled inputs. For row major, height & block sharding are
        // supported.
        if (sliced_output_tensor.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED &&
            sliced_output_tensor.memory_config().memory_layout() != TensorMemoryLayout::BLOCK_SHARDED &&
            dram_output_tensor.layout() == Layout::ROW_MAJOR) {
            sliced_output_tensor = ttnn::to_memory_config(
                sliced_output_tensor, MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1});
        }
        if (sliced_output_tensor.layout() != Layout::ROW_MAJOR && conv_config.output_layout == Layout::ROW_MAJOR) {
            sliced_output_tensor = ttnn::untilize(sliced_output_tensor);
        }
        if (sliced_output_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED) {
            // slice_write expects the output tensor to be correctly shaped when its in interleaved memory layout.
            sliced_output_tensor = ttnn::reshape(
                sliced_output_tensor,
                ttnn::Shape({batch_size, output_slice_height, output_slice_width, out_channels}),
                ttnn::Shape(
                    {batch_size, output_slice_height, output_slice_width, sliced_output_tensor.get_padded_shape()[3]}));
        }
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

    if (conv_config.deallocate_activation) {
        input_tensor_on_device.deallocate(true);
    }
    const auto flattened_output_shape = flatten_4d_shape(dram_output_tensor.logical_shape());
    const auto flattened_padded_output_shape = flatten_4d_shape(dram_output_tensor.padded_shape());

    dram_output_tensor = ttnn::reshape(dram_output_tensor, flattened_output_shape, flattened_padded_output_shape);

    return {dram_output_tensor, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
}

template <typename T>
Result conv2d_L1(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor_,
    const ttnn::Tensor& weight_tensor_,
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
    const std::optional<const DataType>& dtype,
    const std::optional<const ttnn::Tensor>& bias_tensor_,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config) {
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    const DataType output_dtype = dtype.value_or(input_tensor_.dtype());
    std::array<uint32_t, 4> padding_n4 = sliding_window::get_pair_n4_padding(padding);
    auto input_tensor = input_tensor_;
    const auto& weight_tensor = weight_tensor_;
    std::optional<ttnn::Tensor> bias_tensor = bias_tensor_;
    bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding_n4, dilation, groups, conv_config);
    // Store the original stride size for weight folding
    auto orig_stride = stride;
    if (conv_config.enable_kernel_stride_folding) {
        auto folding_result = compute_kernel_stride_folding_params(
            input_height, input_width, in_channels, kernel_size, stride, padding_n4, conv_config);
        input_tensor = fold_tensor(input_tensor, device, stride, kernel_size, padding_n4);
        if (conv_config.deallocate_activation) {
            Tensor input_tensor_pre_folded = input_tensor_;
            input_tensor_pre_folded.deallocate(true);
        }

        // Update the input tensor parameters to the folding result
        input_height = folding_result.input_height;
        input_width = folding_result.input_width;
        in_channels = folding_result.in_channels;
        stride = folding_result.stride;
        kernel_size = folding_result.kernel_size;
        mm_conv = folding_result.mm_conv;
    }
    auto [output_height, output_width] =
        calculate_output_image_size({input_height, input_width}, kernel_size, stride, padding_n4, dilation);

    DeviceComputeKernelConfig compute_config = compute_config_.value_or(get_conv_default_compute_kernel_config(device));

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
            kernel_size[1],
            input_height,
            input_width,
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

    const uint32_t input_channels_alignment = get_input_channels_alignment(
        input_tensor_post_tm.memory_config().memory_layout(),
        input_tensor.layout(),
        mm_conv,
        input_tensor_post_tm.memory_config());
    const uint32_t in_channels_padded = tt::round_up(
        in_channels, get_num_cores_channels_from_parallel_config(parallel_config) * input_channels_alignment);

    auto [opt_conv_op_parallel_config, opt_conv_op_block_config, conv_out_memory_config] = get_conv_configs(
        conv_config,
        compute_config,
        parallel_config,
        output_parallel_config,
        in_channels_padded,
        out_channels,
        batch_size,
        output_height,
        output_width,
        kernel_size,
        compute_grid_size);

    ttnn::Tensor weight_tensor_on_device = weight_tensor;
    std::optional<ttnn::Tensor> bias_tensor_on_device = bias_tensor;

    // Configure weight and bias preparation parameters
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
        bias_tensor.has_value(),
        true,  // parameters_on_device
        conv_config.enable_kernel_stride_folding,
        kernel_size,
        orig_stride,
        padding_n4);

    // Prepare weights and move to device if necessary
    if (!is_device_tensor(weight_tensor)) {
        log_debug(tt::LogOp, "conv2d: Preprocessing weights on host and moving to device.");
        std::tie(weight_tensor_on_device, bias_tensor_on_device) =
            prepare_conv_weights_biases_and_move_to_device(weight_tensor, bias_tensor, params, device);
    } else {
        // Check if device weights are properly prepared
        if (is_valid_device_conv_weights(
                weight_tensor_on_device, in_channels, out_channels, conv_config.weights_dtype)) {
            log_debug(tt::LogOp, "conv2d: Using preprocessed weights from device.");
        } else {
            log_warning(
                tt::LogOp,
                "conv2d: Device weights not properly prepared, pulling back to host and trying to reprocess.");
            // Pull weights back to host, prepare them, and push back to device
            ttnn::Tensor host_weight_tensor = ttnn::operations::core::from_device(weight_tensor_on_device);
            std::tie(weight_tensor_on_device, bias_tensor_on_device) =
                prepare_conv_weights_biases_and_move_to_device(host_weight_tensor, bias_tensor, params, device);
        }
    }

    // Prepare bias tensor if it exists and is not yet on device
    if (bias_tensor_on_device.has_value()) {
        if (!is_device_tensor(bias_tensor_on_device.value())) {
            bias_tensor_on_device = prepare_conv_bias_internal(
                bias_tensor_on_device, out_channels, params, weight_tensor_on_device.dtype(), device);
        } else {
            // Check if device bias is properly prepared
            if (is_valid_device_conv_bias(bias_tensor_on_device.value(), out_channels, conv_config.weights_dtype)) {
                log_debug(tt::LogOp, "conv2d: Using preprocessed bias from device.");
            } else {
                log_warning(
                    tt::LogOp, "conv2d: Device bias not properly prepared, pulling back to host and reprocessing.");
                // Pull bias back to host, prepare it, and push back to device
                ttnn::Tensor host_bias_tensor = ttnn::operations::core::from_device(bias_tensor_on_device.value());
                bias_tensor_on_device = prepare_conv_bias_internal(
                    std::optional<const ttnn::Tensor>(host_bias_tensor),
                    out_channels,
                    params,
                    weight_tensor_on_device.dtype(),
                    device);
            }
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
            .core_range_set = input_tensor_post_tm.memory_config().shard_spec().value().grid,
            .snap_to_tile = true,
        };

        bool bypass_halo =
            (parallel_config.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED &&
             sliding_window_config.get_pad_h() == 0 && sliding_window_config.get_pad_w() == 0);

        if (bypass_halo) {
            if (input_tensor_post_tm.layout() == Layout::TILE) {
                // Reshape is used as a workaround to an issue in to_layout mentioned here :
                // https://github.com/tenstorrent/tt-metal/issues/16330
                input_tensor_post_tm = ttnn::reshape(input_tensor_post_tm, input_tensor_post_tm.padded_shape());
                input_tensor_post_tm = ttnn::to_layout(input_tensor_post_tm, Layout::ROW_MAJOR);
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

        bool enable_split_reader = conv_config.enable_split_reader;
        if (enable_split_reader && opt_conv_op_block_config.act_block_h_ntiles == 1) {
            // If the activation block height is 1, we can't enable split reader.
            enable_split_reader = false;
            log_warning(
                tt::LogOp,
                "Conv2D: Split reader was requested by the user, but it can't be support with just one tile per core "
                "in activation matrix height.");
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
            output_dtype,
            {batch_size, input_height, input_width, in_channels},
            compute_config,
            conv_config.enable_act_double_buffer,
            conv_config.enable_weights_double_buffer,
            enable_split_reader);

        if (memory_config.has_value() && memory_config.value() != conv_output.memory_config()) {
            conv_output = ttnn::to_memory_config(conv_output, memory_config.value(), std::nullopt);
        }
        return {conv_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
    } else {
        if (input_tensor_post_tm.layout() != Layout::TILE) {
            Tensor input_tensor_post_tm_tilized =
                ttnn::to_layout(input_tensor_post_tm, Layout::TILE);
            if (conv_config.deallocate_activation) {
                input_tensor_post_tm.deallocate(/*force*/ true);
                input_tensor_post_tm_tilized = ttnn::move(input_tensor_post_tm_tilized);
            }
            input_tensor_post_tm = input_tensor_post_tm_tilized;
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
            output_dtype,
            program_config,
            activation,
            compute_config);

        if (conv_config.deallocate_activation) {
            input_tensor_post_tm.deallocate(/*force*/ true);
        }
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
    const std::optional<const DataType>& dtype,
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
    const std::optional<const DataType>& dtype,
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
    const std::optional<const DataType>& dtype,
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
        std::move(dtype),
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
    const std::optional<const DataType>& dtype,
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
        std::move(dtype),
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
