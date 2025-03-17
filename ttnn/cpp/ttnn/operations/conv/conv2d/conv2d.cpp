// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <utility>

#include "tt-metalium/buffer_constants.hpp"
#include "tt-metalium/logger.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/slice_write/slice_write.hpp"
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
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/types.hpp"
namespace ttnn {
namespace operations::conv {
using namespace tt;
using sliding_window::ParallelConfig;
using sliding_window::SlidingWindowConfig;

namespace conv2d {

using OutputHeight = uint32_t;
using OutputWidth = uint32_t;
using Result = std::tuple<ttnn::Tensor, OutputHeight, OutputWidth, ttnn::Tensor, std::optional<ttnn::Tensor>>;

template <typename T>
Result conv2d(
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
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config_,
    const std::optional<const ConvSliceConfig>& slice_config_) {
    if (slice_config_.has_value()) {
        return conv2d_DRAM(
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
            slice_config_.value());
    } else {
        return conv2d_L1(
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
            memory_config_);
    }
}

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
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config_,
    const ConvSliceConfig& dram_slice_config) {
    uint32_t output_height =
        ((input_height - kernel_size[0] - ((kernel_size[0] - 1) * (dilation[0] - 1)) + 2 * padding[0]) / stride[0]) + 1;
    uint32_t output_width =
        ((input_width - kernel_size[1] - ((kernel_size[0] - 1) * (dilation[0] - 1)) + 2 * padding[1]) / stride[1]) + 1;

    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    ttnn::Tensor input_tensor_on_device;
    if (!is_tensor_on_device_or_multidevice(input_tensor)) {
        input_tensor_on_device = ttnn::operations::core::to_device(input_tensor, device, std::nullopt);
    } else {
        input_tensor_on_device = input_tensor;
    }
    DeviceComputeKernelConfig compute_config = compute_config_.value_or(get_conv_default_compute_kernel_config(device));
    const auto compute_grid_size = device->compute_with_storage_grid_size();

    ttnn::Tensor weight_tensor_on_device;
    std::optional<ttnn::Tensor> bias_tensor_on_device;
    TT_FATAL(input_tensor_on_device.memory_config().is_dram(), "Conv DRAM expects the input tensor to be in DRAM.");
    TT_FATAL(conv_config.dtype != tt::tt_metal::DataType::BFLOAT8_B, "Conv DRAM currently doesn't support BFLOAT8_B");

    // Tensor dram_output_tensor = ttnn::zeros(
    //         ttnn::Shape({batch_size, output_height, output_width, out_channels}),
    //         conv_config.dtype,
    //         tt_metal::Layout::ROW_MAJOR,
    //         *device,
    //         MemoryConfig{
    //             .memory_layout = TensorMemoryLayout::INTERLEAVED,
    //             .buffer_type = BufferType::DRAM,
    //         });

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
    bool auto_shard = false;
    std::optional<MemoryConfig> input_memory_config = std::nullopt;
    const bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding, dilation, groups, conv_config);
    if (dram_slice_config.slice_output_height) {
        for (int batch_index = 0; batch_index < batch_size; batch_index++) {
            for (uint32_t output_slice_height_start = 0; output_slice_height_start < output_height;
                 output_slice_height_start += dram_slice_config.output_slice_size) {
                uint32_t output_slice_height_end =
                    std::min(output_height, output_slice_height_start + dram_slice_config.output_slice_size);
                uint32_t output_slice_height = output_slice_height_end - output_slice_height_start;

                if (output_slice_height == 0) {
                    continue;
                }

                int input_slice_height_start = (output_slice_height_start * stride[0]) - padding[0];
                int input_slice_height_end = ((output_slice_height_end - 1) * stride[0]) - padding[0] +
                                             ((kernel_size[0] - 1) * (dilation[0] - 1)) + kernel_size[0];
                int pad_top = std::max(0, -input_slice_height_start);
                int pad_bottom = std::max<int>(0, input_slice_height_end - input_height);
                input_slice_height_start = std::max(0, input_slice_height_start);
                input_slice_height_end = std::min<int>(input_height, input_slice_height_end);
                uint32_t input_slice_height = input_slice_height_end - input_slice_height_start;

                if (input_slice_height_start >= input_slice_height_end) {
                    continue;
                }
                if (!conv_config.shard_layout.has_value()) {
                    conv_config = determine_conv_config_for_auto_shard(
                        conv_config,
                        mm_conv,
                        batch_size,
                        in_channels,
                        out_channels,
                        output_height,
                        output_width,
                        weight_tensor.get_logical_shape()[3],
                        input_slice_height,
                        input_width,
                        compute_grid_size,
                        input_tensor.layout(),
                        std::make_optional(input_tensor.memory_config()),
                        kernel_size,
                        groups,
                        bias_tensor.has_value(),
                        compute_config);
                    input_memory_config = get_input_memory_config(
                        conv_config,
                        in_channels,
                        out_channels,
                        batch_size,
                        input_slice_height,
                        input_width,
                        output_slice_height,
                        output_width,
                        compute_grid_size,
                        Layout::ROW_MAJOR,
                        mm_conv);
                    auto_shard = true;
                    tt::log_info(
                        tt::LogOp,
                        "DRAM Width slicing using {}, Input Memory Config = {} ",
                        conv_config.shard_layout.value(),
                        input_memory_config.value());
                }
                auto sliced_input_tensor = ttnn::slice(
                    queue_id,
                    input_tensor,
                    std::array<uint32_t, 4>{batch_index, input_slice_height_start, 0, 0},  // Start
                    std::array<uint32_t, 4>{batch_index + 1, input_slice_height_end, input_width, in_channels},
                    std::array<uint32_t, 4>{
                        1,
                        1,
                        1,
                        1,
                    },  // Step,
                    input_memory_config);
                log_debug(tt::LogOp, "Sliced input tensor shape: {}", sliced_input_tensor.get_logical_shape());
                if (pad_top > 0 || pad_bottom > 0) {
                    auto pad_top_tensor = ttnn::pad(
                        queue_id,
                        sliced_input_tensor,
                        std::vector<std::pair<uint32_t, uint32_t>>{{0, 0}, {pad_top, pad_bottom}, {0, 0}, {0, 0}},
                        0,
                        true,
                        std::nullopt);
                    sliced_input_tensor = pad_top_tensor;
                }
                log_debug(tt::LogOp, "Padded sliced input tensor shape: {}", sliced_input_tensor.get_logical_shape());
                auto conv_config_l1 = conv_config;
                conv_config_l1.reshard_if_not_optimal = true;
                conv_config_l1.output_layout = Layout::TILE;
                ttnn::Tensor sliced_output_tensor;
                std::tie(
                    sliced_output_tensor, std::ignore, std::ignore, weight_tensor_on_device, bias_tensor_on_device) =
                    conv2d_L1(
                        queue_id,
                        sliced_input_tensor,
                        first_run ? weight_tensor : weight_tensor_on_device,
                        device,
                        in_channels,
                        out_channels,
                        1,
                        input_slice_height + pad_top + pad_bottom,
                        input_width,
                        kernel_size,
                        stride,
                        {0, padding[1]},
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
                sliced_output_tensor =
                    ttnn::to_layout(sliced_output_tensor, Layout::ROW_MAJOR, std::nullopt, std::nullopt, device);
                sliced_output_tensor = ttnn::reshape(
                    sliced_output_tensor, ttnn::Shape({1, output_slice_height, output_width, out_channels}));
                ttnn::slice_write(
                    queue_id,
                    sliced_output_tensor,
                    dram_output_tensor,
                    std::array<uint32_t, 4>{batch_index, output_slice_height_start, 0, 0},
                    std::array<uint32_t, 4>{batch_index + 1, output_slice_height_end, output_width, out_channels},
                    std::array<uint32_t, 4>{1, 1, 1, 1});

                log_debug(tt::LogOp, "Dram output tensor shape: {}", dram_output_tensor.get_logical_shape());
                first_run = false;
            }
        }
    } else {
        for (uint32_t output_slice_width_start = 0; output_slice_width_start < output_width;
             output_slice_width_start += dram_slice_config.output_slice_size) {
            uint32_t output_slice_width_end =
                std::min(output_width, output_slice_width_start + dram_slice_config.output_slice_size);
            uint32_t output_slice_width = output_slice_width_end - output_slice_width_start;

            if (output_slice_width == 0) {
                continue;
            }
            int input_slice_width_start = (output_slice_width_start * stride[1]) - padding[1];
            int input_slice_width_end = ((output_slice_width_end - 1) * stride[1]) - padding[1] +
                                        ((kernel_size[1] - 1) * (dilation[1] - 1)) + kernel_size[1];

            int pad_left = std::max(0, -input_slice_width_start);
            int pad_right = std::max<int>(0, input_slice_width_end - input_width);
            input_slice_width_start = std::max(0, input_slice_width_start);
            input_slice_width_end = std::min<int>(input_width, input_slice_width_end);
            uint32_t input_slice_width = input_slice_width_end - input_slice_width_start;

            if (input_slice_width_start >= input_slice_width_end) {
                continue;
            }

            if (!conv_config.shard_layout.has_value()) {
                conv_config = determine_conv_config_for_auto_shard(
                    conv_config,
                    mm_conv,
                    batch_size,
                    in_channels,
                    out_channels,
                    output_height,
                    output_width,
                    weight_tensor.get_logical_shape()[3],
                    input_height,
                    input_slice_width,
                    compute_grid_size,
                    input_tensor.layout(),
                    std::make_optional(input_tensor.memory_config()),
                    kernel_size,
                    groups,
                    bias_tensor.has_value(),
                    compute_config);
                input_memory_config = get_input_memory_config(
                    conv_config,
                    in_channels,
                    out_channels,
                    batch_size,
                    input_height,
                    input_slice_width,
                    output_height,
                    output_slice_width,
                    compute_grid_size,
                    Layout::ROW_MAJOR,
                    mm_conv);
                auto_shard = true;
                tt::log_info(
                    tt::LogOp,
                    "DRAM Width slicing using {}, Input Memory Config = {} ",
                    conv_config.shard_layout.value(),
                    input_memory_config.value());
            }
            auto sliced_input_tensor = ttnn::slice(
                queue_id,
                input_tensor,
                std::array<uint32_t, 4>{0, 0, input_slice_width_start, 0},  // Start
                std::array<uint32_t, 4>{batch_size, input_height, input_slice_width_end, in_channels},
                std::array<uint32_t, 4>{
                    1,
                    1,
                    1,
                    1,
                },  // Step
                input_memory_config);
            log_debug(tt::LogOp, "Sliced input tensor shape: {}", sliced_input_tensor.get_logical_shape());
            if (pad_left > 0 || pad_right > 0) {
                auto pad_top_tensor = ttnn::pad(
                    queue_id,
                    sliced_input_tensor,
                    std::vector<std::pair<uint32_t, uint32_t>>{{0, 0}, {0, 0}, {pad_left, pad_right}, {0, 0}},
                    0,
                    true,
                    std::nullopt);
                sliced_input_tensor = pad_top_tensor;
            }
            log_debug(tt::LogOp, "Padded sliced input tensor shape: {}", sliced_input_tensor.get_logical_shape());
            auto conv_config_l1 = conv_config;
            conv_config_l1.reshard_if_not_optimal = true;
            conv_config_l1.output_layout = Layout::TILE;
            ttnn::Tensor sliced_output_tensor;
            std::tie(sliced_output_tensor, std::ignore, std::ignore, weight_tensor_on_device, bias_tensor_on_device) =
                conv2d_L1(
                    queue_id,
                    sliced_input_tensor,
                    first_run ? weight_tensor : weight_tensor_on_device,
                    device,
                    in_channels,
                    out_channels,
                    batch_size,
                    input_height,
                    input_slice_width + pad_left + pad_right,
                    kernel_size,
                    stride,
                    {padding[0], 0},
                    dilation,
                    groups,
                    first_run ? bias_tensor : (std::optional<const ttnn::Tensor>)(bias_tensor_on_device),
                    conv_config_l1,
                    compute_config_,
                    memory_config_);
            tt::log_info(LogOp, "Output Memory Config: {}", sliced_output_tensor.memory_config());
            if (sliced_output_tensor.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED &&
                sliced_output_tensor.memory_config().memory_layout != TensorMemoryLayout::BLOCK_SHARDED) {
                sliced_output_tensor = ttnn::to_memory_config(
                    sliced_output_tensor,
                    MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::L1});
            }
            sliced_output_tensor =
                ttnn::to_layout(sliced_output_tensor, Layout::ROW_MAJOR, std::nullopt, std::nullopt, device);
            sliced_output_tensor = ttnn::reshape(
                queue_id,
                sliced_output_tensor,
                ttnn::Shape({batch_size, output_height, output_slice_width, out_channels}));

            ttnn::slice_write(
                queue_id,
                sliced_output_tensor,
                dram_output_tensor,
                std::array<uint32_t, 4>{0, 0, output_slice_width_start, 0},
                std::array<uint32_t, 4>{batch_size, output_height, output_slice_width_end, out_channels},
                std::array<uint32_t, 4>{1, 1, 1, 1});
            log_debug(tt::LogOp, "Dram output tensor shape: {}", dram_output_tensor.get_logical_shape());
            first_run = false;
        }
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
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config) {
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    const bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding, dilation, groups, conv_config);
    const uint32_t output_height =
        ((input_height - kernel_size[0] - ((kernel_size[0] - 1) * (dilation[0] - 1)) + 2 * padding[0]) / stride[0]) + 1;
    const uint32_t output_width =
        ((input_width - kernel_size[1] - ((kernel_size[0] - 1) * (dilation[0] - 1)) + 2 * padding[1]) / stride[1]) + 1;

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
            weight_tensor.get_logical_shape()[3],
            input_height,
            input_width,
            compute_grid_size,
            input_tensor.layout(),
            ttnn::is_tensor_on_device_or_multidevice(input_tensor) ? std::make_optional(input_tensor.memory_config())
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

    bool weight_is_on_device = ttnn::is_tensor_on_device_or_multidevice(weight_tensor);
    ttnn::Tensor weight_tensor_on_device = weight_tensor;
    std::optional<ttnn::Tensor> bias_tensor_on_device = bias_tensor;
    if (!weight_is_on_device || conv_config.always_preprocess_weights) {
        // prepare weights in desired layout and move to device

        // TODO: Implement heuristic to decide if weights should be preprocessed on device.
        if (conv_config.preprocess_weights_on_device == false) {
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
                true);
        }
    }
    // if 1x1 conv w/ stride 1, convert input tensor to tile layout if required
    if (mm_conv) {
        input_tensor_post_tm = ttnn::to_layout(
            input_tensor_post_tm, Layout::TILE, conv_config.dtype, input_tensor_post_tm.memory_config(), device);
    }
    // call optimized conv op or matmul micro op
    bool input_is_on_device = ttnn::is_tensor_on_device_or_multidevice(input_tensor_post_tm);
    TT_ASSERT(input_is_on_device);

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
            .snap_to_tile = true,
        };

        bool bypass_halo =
            (parallel_config.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED &&
             sliding_window_config.pad_hw.first == 0 && sliding_window_config.pad_hw.second == 0);

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
                0,
                input_tensor_post_tm.memory_config(),
                true);

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
            conv_config.input_channels_alignment == 16,
            compute_config,
            conv_config.enable_act_double_buffer,
            conv_config.enable_weights_double_buffer,
            conv_config.enable_split_reader);

        if (memory_config.has_value() && memory_config.value() != conv_output.memory_config()) {
            conv_output = ttnn::to_memory_config(conv_output, memory_config.value(), std::nullopt);
        }
        return {conv_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
    } else {
        // run conv as matmul
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

        return {matmul_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
    }
}

Result Conv2dOperation::invoke(
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
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config_,
    const std::optional<const ConvSliceConfig>& slice_config_) {
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
        std::move(slice_config_));
}

Result Conv2dOperation::invoke(
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
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const ConvSliceConfig>& slice_config_) {
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
        std::move(slice_config_));
}

}  // namespace conv2d
}  // namespace operations::conv
}  // namespace ttnn
