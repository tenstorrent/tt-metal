// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include <tt-metalium/buffer_types.hpp>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

#include "tt-metalium/constants.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/sliding_window/op_slicing/op_slicing.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/experimental/padded_slice/padded_slice.hpp"
#include "ttnn/operations/experimental/slice_write/slice_write.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::operations::conv::conv2d {

using ttnn::Conv2dResult;
using ttnn::Conv2dResultWithOptions;
using Result = Conv2dResult;
using ResultWithOptions = Conv2dResultWithOptions;

Result conv2d_L1(
    const ttnn::Tensor& input_tensor_,
    const ttnn::Tensor& weight_tensor_,
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
    const std::optional<const ttnn::Tensor>& bias_tensor_,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config) {
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    const DataType output_dtype = dtype.value_or(input_tensor_.dtype());
    std::array<uint32_t, 4> padding_n4 = sliding_window::get_pair_n4_padding(padding);
    const auto& weight_tensor = weight_tensor_;
    std::optional<ttnn::Tensor> bias_tensor = bias_tensor_;
    bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding_n4, dilation, groups, conv_config);
    // Store the original stride size for weight folding
    auto orig_stride = stride;

    auto input_tensor = fold_input_tensor_if_required(
        input_tensor_,
        device,
        batch_size,
        input_height,
        input_width,
        in_channels,
        kernel_size,
        stride,
        dilation,
        padding_n4,
        mm_conv,
        conv_config);

    if (conv_config.enable_activation_reuse) {
        if (conv_config.enable_act_double_buffer) {
            conv_config.enable_act_double_buffer = false;
            log_debug(
                tt::LogOp,
                "Activation double buffering is currently not supported when activation reuse optimization is enabled, "
                "disabling double buffering.");
        }

        if (conv_config.enable_weights_double_buffer) {
            conv_config.enable_weights_double_buffer = false;
            log_debug(
                tt::LogOp,
                "Weights are already fully buffered when activation reuse optimization is enabled, disabling weights "
                "double buffering.");
        }
    }
    auto [output_height, output_width] =
        calculate_output_image_size({input_height, input_width}, kernel_size, stride, padding_n4, dilation);

    // Use weights_dtype from config if set, otherwise use weight tensor's dtype
    DataType weight_dtype = conv_config.weights_dtype.value_or(weight_tensor_.dtype());
    DeviceComputeKernelConfig compute_config =
        compute_config_.value_or(get_conv_default_compute_kernel_config(device, input_tensor_.dtype(), weight_dtype));

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
            ttnn::is_device_tensor(input_tensor) ? std::make_optional(input_tensor.memory_config()) : std::nullopt,
            kernel_size,
            stride,
            dilation,
            padding_n4,
            groups,
            bias_tensor.has_value(),
            compute_config);
        auto_shard = true;
    }
    const bool should_deallocate_act = conv_config.deallocate_activation && !input_tensor.memory_config().is_dram();
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
        input_tensor_post_tm.layout(),
        false,
        mm_conv,
        input_tensor_post_tm.memory_config());
    const uint32_t in_channels_padded = tt::round_up(
        in_channels, get_num_cores_channels_from_parallel_config(parallel_config) * input_channels_alignment);

    // mm_conv (1x1, stride 1, no pad) runs as ttnn::linear below; the depthwise flags must not
    // reshape its block config or bias/weight preparation. has_bias is otherwise accepted by the
    // depthwise path (the factory folds bias on the last kernel tap).
    const bool conv_is_1d_depthwise =
        !mm_conv &&
        is_1d_depthwise_conv(groups, in_channels, out_channels, kernel_size[0], input_height, bias_tensor.has_value());
    const bool coalesce_1d_depthwise_kw_reads = should_coalesce_1d_depthwise_conv_reads(
        conv_is_1d_depthwise,
        parallel_config.shard_scheme,
        in_channels_padded,
        kernel_size[1],
        dilation[1],
        input_tensor_post_tm.dtype());

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
        compute_grid_size,
        conv_is_1d_depthwise,
        coalesce_1d_depthwise_kw_reads);

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
        input_height,
        input_width,
        mm_conv && auto_shard,
        out_channels,
        bias_tensor.has_value(),
        conv_config.enable_kernel_stride_folding.value(),
        conv_config.full_inner_dim,
        conv_config.enable_activation_reuse,
        coalesce_1d_depthwise_kw_reads,
        orig_stride,
        mm_conv);

    // Prepare weights and move to device if necessary
    if (!is_device_tensor(weight_tensor)) {
        log_trace(tt::LogOp, "conv2d: Preprocessing weights on host and moving to device.");
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
            log_trace(tt::LogOp, "conv2d: Preprocessing bias on host and moving to device.");

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

    // call conv op or matmul micro op
    bool input_is_on_device = ttnn::is_device_tensor(input_tensor_post_tm);
    TT_ASSERT(input_is_on_device);

    if (!mm_conv) {
        // call halo op
        sliding_window::SlidingWindowConfig sliding_window_config = sliding_window::SlidingWindowConfig{
            .batch_size = batch_size,
            .input_hw = {input_height, input_width},
            .window_hw = {kernel_size[0], kernel_size[1]},
            .stride_hw = {stride[0], stride[1]},
            .padding = {{padding_n4[0], padding_n4[1], padding_n4[2], padding_n4[3]}},
            .dilation_hw = {dilation[0], dilation[1]},
            .num_cores_nhw = opt_conv_op_parallel_config.num_cores_nhw,
            .core_range_set = input_tensor_post_tm.memory_config().shard_spec().value().grid,
            .snap_to_tile = true,
            .padding_mode = conv_config.padding_mode,
        };

        if (parallel_config.shard_scheme != TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor_post_tm.layout() != Layout::ROW_MAJOR || sliding_window_config.get_pad_h() != 0 ||
            sliding_window_config.get_pad_w() != 0) {
            ttnn::Tensor halo_output = ttnn::halo(
                input_tensor_post_tm,
                sliding_window_config,
                compute_config,
                0,
                false,
                parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
                true,
                conv_config.config_tensors_in_dram);

            // In cases where input tensor is in DRAM and it gets sharded, we need to deallocate the sharded input
            // tensor at this point (it will be deallocated automatically because nothing is using it, but reallocating
            // halo output will be affected so we need to deallocate it manually before reallocating halo output)
            if (conv_config.deallocate_activation && !input_tensor_post_tm.memory_config().is_dram()) {
                input_tensor_post_tm.deallocate(/*force*/ true);
            }

            input_tensor_post_tm = std::move(halo_output);

            if (conv_config.reallocate_halo_output) {
                input_tensor_post_tm = ttnn::move(input_tensor_post_tm);
            }
        }

        const std::array<std::uint32_t, 4> input_tensor_shape = {
            batch_size,
            input_height,
            input_width,
            in_channels,
        };

        // call conv micro op
        auto conv_output = ttnn::prim::conv2d(
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
            input_tensor_shape,
            compute_config,
            conv_config.enable_act_double_buffer,
            conv_config.enable_weights_double_buffer,
            conv_config.full_inner_dim,
            conv_config.enable_activation_reuse,
            conv_config.config_tensors_in_dram,
            conv_config.force_split_reader);

        if (memory_config.has_value() && memory_config.value() != conv_output.memory_config()) {
            conv_output = ttnn::to_memory_config(conv_output, memory_config.value(), std::nullopt);
        }
        return {conv_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
    }  // Matmul expects inputs to be in Tile Layout
    tilize_with_optional_deallocation(input_tensor_post_tm, should_deallocate_act);

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

    ttnn::Tensor matmul_output = ttnn::linear(
        input_tensor_post_tm,
        weight_tensor_on_device,
        bias_tensor_on_device,
        false,
        false,
        mm_output_memory_config,
        output_dtype,
        program_config,
        // for sharded input, activation is set on program config
        input_tensor_post_tm.is_sharded() ? std::nullopt : conv_config.activation,
        compute_config);

    if (should_deallocate_act) {
        input_tensor_post_tm.deallocate(/*force*/ true);
    }
    if (memory_config.has_value() && memory_config.value() != matmul_output.memory_config()) {
        matmul_output = ttnn::to_memory_config(matmul_output, memory_config.value(), std::nullopt);
    }

    return {matmul_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
}

ResultWithOptions result_to_result_with_options(
    const Result& result, const bool return_output_dim, const bool return_weights_and_bias) {
    if (return_output_dim && return_weights_and_bias) {
        return std::make_tuple(
            std::get<0>(result),
            std::make_tuple(std::get<1>(result), std::get<2>(result)),
            std::make_tuple(std::get<3>(result), std::get<4>(result)));
    }
    if (return_output_dim) {
        return std::make_tuple(std::get<0>(result), std::make_tuple(std::get<1>(result), std::get<2>(result)));
    }
    if (return_weights_and_bias) {
        return std::make_tuple(std::get<0>(result), std::make_tuple(std::get<3>(result), std::get<4>(result)));
    }
    return std::get<0>(result);
}

class Conv2dSliceAttr : public ttnn::operations::op_slicing::OpSliceAttr {
    using OptionalRefTensor = std::optional<std::reference_wrapper<ttnn::Tensor>>;

    Conv2dConfig auto_slice_conv_config;
    uint32_t batch_size;
    IOShape input_shape;
    uint32_t input_channels;
    uint32_t output_channels;
    std::array<uint32_t, 2> kernel_size;
    std::array<uint32_t, 2> stride;
    std::array<uint32_t, 4> padding_n4;
    std::array<uint32_t, 2> dilation;
    uint32_t groups;
    tt::tt_metal::Layout input_layout;
    tt::tt_metal::DataType input_dtype;
    tt::tt_metal::DataType output_dtype;
    Tensor& weight_tensor;
    OptionalRefTensor bias_tensor;
    Conv2dConfig conv_config;
    DeviceComputeKernelConfig compute_config;
    MeshDevice* device;

public:
    Conv2dSliceAttr(
        uint32_t batch_size,
        IOShape input_shape,
        uint32_t input_channels,
        uint32_t output_channels,
        std::array<uint32_t, 2> kernel_size,
        std::array<uint32_t, 2> stride,
        std::array<uint32_t, 4> padding_n4,
        std::array<uint32_t, 2> dilation,
        uint32_t groups,
        tt::tt_metal::Layout input_layout,
        tt::tt_metal::DataType input_dtype,
        tt::tt_metal::DataType output_dtype,
        Tensor& weight_tensor,
        OptionalRefTensor bias_tensor,
        const Conv2dConfig& conv_config,
        const DeviceComputeKernelConfig& compute_config,
        MeshDevice* device) :
        batch_size(batch_size),
        input_shape(input_shape),
        input_channels(input_channels),
        output_channels(output_channels),
        kernel_size(kernel_size),
        stride(stride),
        padding_n4(padding_n4),
        dilation(dilation),
        groups(groups),
        input_layout(input_layout),
        input_dtype(input_dtype),
        output_dtype(output_dtype),
        weight_tensor(weight_tensor),
        bias_tensor(bias_tensor),
        conv_config(conv_config),
        compute_config(compute_config),
        device(device) {}

    std::tuple<std::tuple<IOShape, IOShape>, std::array<uint32_t, 4>> get_input_slice_and_padding(
        const IOShape& output_slice_start, const IOShape& output_slice_end) const {
        auto [output_slice_height_start, output_slice_width_start] = output_slice_start;
        auto [output_slice_height_end, output_slice_width_end] = output_slice_end;
        auto [input_height, input_width] = input_shape;

        // Calculate required input slice range based on output slice
        // Formula: input_start = (output_start * stride) - padding
        // Formula: input_end = ((output_end - 1) * stride) - padding + dilated_kernel_size
        int input_slice_height_start = (output_slice_height_start * stride[0]) - padding_n4[0];
        int input_slice_height_end = ((output_slice_height_end - 1) * stride[0]) - padding_n4[0] +
                                     ((kernel_size[0] - 1) * (dilation[0] - 1)) + kernel_size[0];
        int input_slice_width_start = (output_slice_width_start * stride[1]) - padding_n4[2];
        int input_slice_width_end = ((output_slice_width_end - 1) * stride[1]) - padding_n4[2] +
                                    ((kernel_size[1] - 1) * (dilation[1] - 1)) + kernel_size[1];

        // Calculate padding needed if input slice extends beyond input tensor
        uint32_t pad_top = std::max<int>(0, -input_slice_height_start);
        uint32_t pad_bottom = std::max<int>(0, input_slice_height_end - input_height);
        uint32_t pad_left = std::max<int>(0, -input_slice_width_start);
        uint32_t pad_right = std::max<int>(0, input_slice_width_end - input_width);

        // Clamp input slice to valid input tensor bounds
        input_slice_height_start = std::max<int>(0, input_slice_height_start);
        input_slice_height_end = std::min<int>(input_height, input_slice_height_end);
        input_slice_width_start = std::max<int>(0, input_slice_width_start);
        input_slice_width_end = std::min<int>(input_width, input_slice_width_end);

        // Calculate full output dimensions
        auto [output_height, output_width] = calculate_output_image_size(
            std::array<uint32_t, 2>{input_height, input_width}, kernel_size, stride, padding_n4, dilation);

        // Special handling for edges: if output slice starts/ends at tensor boundary,
        // use the full original padding and reset input slice to tensor boundary
        if (output_slice_height_start == 0) {
            pad_top = padding_n4[0];
            input_slice_height_start = 0;
        }
        if (output_slice_height_end == output_height) {
            pad_bottom = padding_n4[1];
            input_slice_height_end = input_height;
        }
        if (output_slice_width_start == 0) {
            pad_left = padding_n4[2];
            input_slice_width_start = 0;
        }
        if (output_slice_width_end == output_width) {
            pad_right = padding_n4[3];
            input_slice_width_end = input_width;
        }
        uint32_t input_slice_height = input_slice_height_end - input_slice_height_start;
        uint32_t input_slice_width = input_slice_width_end - input_slice_width_start;
        uint32_t output_slice_width = output_slice_width_end - output_slice_width_start;
        // Apply width rounding and adjust right padding if necessary
        uint32_t width_rounding_value =
            (conv_config.output_layout == tt::tt_metal::Layout::TILE) ? tt::constants::TILE_HEIGHT : 1;

        bool single_slice =
            (input_slice_height == std::get<0>(input_shape)) && (input_slice_width == std::get<1>(input_shape));

        if (output_slice_width % width_rounding_value != 0 && !single_slice) {
            uint32_t additional_padded_width = width_rounding_value - (output_slice_width % width_rounding_value);
            log_trace(
                tt::LogOp,
                "Conv2d DRAM Slicing: Additional padding of {} added to the right side.",
                additional_padded_width);
            pad_right += additional_padded_width * stride[1];  // Adjust right padding
        }

        return {
            {{input_slice_height_start, input_slice_width_start}, {input_slice_height_end, input_slice_width_end}},
            {pad_top, pad_bottom, pad_left, pad_right}};
    }

    std::tuple<IOShape, IOShape> get_input_slice(
        const IOShape& output_slice_start, const IOShape& output_slice_end) const override {
        return std::get<0>(get_input_slice_and_padding(output_slice_start, output_slice_end));
    }

    uint32_t get_L1_usage(
        const IOShape& output_slice_start,
        const IOShape& output_slice_end,
        const op_slicing::Op2DSliceConfig& slice_config) const override {
        // Remove this->conv_config from scope so that for each slice, conv_config can be calculated independently.
        auto conv_config = this->conv_config;
        bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding_n4, dilation, groups, conv_config);
        TT_FATAL(!mm_conv, "Conv2D DRAM with matmul should never use the slicing code path.");

        auto [input_slicing, slice_padding] = get_input_slice_and_padding(output_slice_start, output_slice_end);
        auto [input_slice_start, input_slice_end] = input_slicing;
        auto [input_slice_height_start, input_slice_width_start] = input_slice_start;
        auto [input_slice_height_end, input_slice_width_end] = input_slice_end;
        auto input_slice_height = input_slice_height_end - input_slice_height_start;
        auto input_slice_width = input_slice_width_end - input_slice_width_start;

        auto [output_slice_height, output_slice_width] = calculate_output_image_size(
            {input_slice_height, input_slice_width}, kernel_size, stride, slice_padding, dilation);
        auto compute_grid = device->compute_with_storage_grid_size();
        log_trace(
            tt::LogOp,
            "Conv2D DRAM Auto Slice Max Input Size : {}x{}, Max Output Size : {}x{}",
            input_slice_height,
            input_slice_width,
            output_slice_height,
            output_slice_width);

        auto sliced_input_tensor_memory_config = get_input_memory_config(output_slice_start, output_slice_end);
        if (!conv_config.shard_layout.has_value()) {
            conv_config.shard_layout = sliced_input_tensor_memory_config.memory_layout();
        }
        auto conv_L1_usage = calculate_L1_usage_for_conv_op(
            batch_size,
            input_channels,
            output_channels,
            input_slice_height,
            input_slice_width,
            output_slice_height,
            output_slice_width,
            kernel_size,
            stride,
            slice_padding,
            dilation,
            groups,
            bias_tensor.has_value(),
            input_dtype,
            output_dtype,
            input_layout,
            compute_grid,
            false,
            conv_config.shard_layout.value(),
            compute_config,
            conv_config,
            sliced_input_tensor_memory_config);

        log_trace(
            tt::LogOp,
            "Conv DRAM Auto slicing: num_slices = {}, input_memory_config = {}, L1 usage = {}",
            slice_config.num_slices,
            sliced_input_tensor_memory_config,
            conv_L1_usage);
        return std::max(conv_L1_usage.halo_input_size + conv_L1_usage.halo_output_size, conv_L1_usage.total_size);
    }

    tt::tt_metal::MemoryConfig get_input_memory_config(
        const IOShape& output_slice_start, const IOShape& output_slice_end) const override {
        auto compute_grid_size = device->compute_with_storage_grid_size();
        auto conv_config = this->conv_config;

        auto [input_slicing, slice_padding] = get_input_slice_and_padding(output_slice_start, output_slice_end);
        auto [input_start, input_end] = input_slicing;
        uint32_t input_slice_height = std::get<0>(input_end) - std::get<0>(input_start);
        uint32_t input_slice_width = std::get<1>(input_end) - std::get<1>(input_start);
        // Use padded output dimensions to match what the halo op actually produces.
        // The halo output is tile-aligned, so edge slices get additional padding
        // (e.g., output width 4 pads to 32). Without this, the shard spec is computed
        // for the unpadded dimensions, leading to L1 underestimation.
        auto [output_slice_height, output_slice_width] = calculate_output_image_size(
            {input_slice_height, input_slice_width}, kernel_size, stride, slice_padding, dilation);

        bool single_slice =
            (input_slice_height == std::get<0>(input_shape)) && (input_slice_width == std::get<1>(input_shape));

        if (!conv_config.shard_layout.has_value()) {
            if (!conv_config.weights_dtype.has_value()) {
                conv_config.weights_dtype = weight_tensor.dtype();
            }
            conv_config = determine_conv_config_for_auto_shard(
                conv_config,
                false,
                batch_size,
                input_channels,
                output_channels,
                output_slice_height,
                output_slice_width,
                weight_tensor.logical_shape()[3],
                input_slice_height,
                input_slice_width,
                device->compute_with_storage_grid_size(),
                input_layout,
                input_dtype,
                output_dtype,
                std::nullopt,
                kernel_size,
                stride,
                dilation,
                padding_n4,
                groups,
                bias_tensor.has_value(),
                compute_config);
        }
        TT_FATAL(conv_config.shard_layout.has_value(), " Conv2D DRAM Slicing must have a shard layout set.");

        ShardOrientation shard_orientation =
            conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;
        auto sliced_input_tensor_memory_config = std::get<1>(determine_input_memory_config(
            conv_config.shard_layout.value(),
            shard_orientation,
            batch_size,
            ttnn::Shape({batch_size, input_slice_height, input_slice_width, input_channels}),
            ttnn::Shape({batch_size, output_slice_height, output_slice_width, output_channels}),
            false,
            compute_grid_size,
            input_layout,
            single_slice ? BufferType::L1 : BufferType::DRAM));
        return sliced_input_tensor_memory_config;
    }

    std::string name() const override { return "Conv2D"; }

    std::vector<ttnn::Tensor> run_L1_op(
        const ttnn::Tensor& sliced_input_tensor,
        const IOShape& output_slice_start,
        const IOShape& output_slice_end) override {
        // Use helper function to calculate slice bounds and padding
        auto [input_slicing, this_op_padding] = get_input_slice_and_padding(output_slice_start, output_slice_end);
        auto [input_slice_start, input_slice_end] = input_slicing;
        auto [input_slice_height_start, input_slice_width_start] = input_slice_start;
        auto [input_slice_height_end, input_slice_width_end] = input_slice_end;
        // Calculate dimensions directly from result
        uint32_t input_slice_height = input_slice_height_end - input_slice_height_start;
        uint32_t input_slice_width = input_slice_width_end - input_slice_width_start;

        if (!conv_config.shard_layout.has_value() && sliced_input_tensor.is_sharded()) {
            conv_config.shard_layout = sliced_input_tensor.memory_config().memory_layout();
        }
        auto conv_config_l1 = conv_config;

        conv_config_l1.deallocate_activation = true;
        conv_config_l1.reallocate_halo_output = true;

        // Force Conv2d_L1 to always output tiled layout to reduce CB Memory usage.
        conv_config_l1.output_layout = Layout::TILE;

        auto conv2d_result = conv2d_L1(
            sliced_input_tensor,
            weight_tensor,
            device,
            input_channels,
            output_channels,
            batch_size,
            input_slice_height,
            input_slice_width,
            kernel_size,
            stride,
            this_op_padding,
            dilation,
            groups,
            output_dtype,
            bias_tensor,
            conv_config_l1,
            compute_config,
            std::nullopt);
        weight_tensor = std::get<3>(conv2d_result);
        if (bias_tensor.has_value()) {
            bias_tensor->get() = std::get<4>(conv2d_result).value();
        }
        return {std::get<0>(conv2d_result)};
    }
};

// Channel-chunk decision for grouped convs: a grouped conv can be split exactly along
// the channel/group axis (chunk k owns groups [k*G/K, (k+1)*G/K)) with no cross-chunk
// dependency, so when a single call cannot satisfy the per-core L1 limit even with
// maximum spatial slicing, the op can run as K channel chunks instead. Returns the
// number of channel chunks to use (1 = do not chunk). Two weight forms are chunkable:
//  - Host weights in OIHW layout: each chunk re-prepares its own weight slice on host.
//  - Prepared device weights in the 1D-depthwise layout [1, 1, act_block_h*32*K, C_padded]
//    (TILE, see convert_conv_weight_tensor_to_depthwise_layout): each chunk slices its
//    channel range out of the last dim directly in TILE (chunk ends are TILE_WIDTH-aligned,
//    so the slice stays in TILE with no ROW_MAJOR round-trip and no host re-preparation)
//    and the chunk conv consumes the TILE slice as-is (valid device weights), so the path
//    stays safe under trace capture. The chunk conv_config pins act_block_h to the prepared
//    tap-slab height so the kernel reads each kernel tap's slab at the boundaries the
//    preparation laid out.
static uint32_t channel_chunk_count_if_needed(
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    MeshDevice* device,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t in_channels,
    uint32_t out_channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 4> padding_n4,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    tt::tt_metal::Layout input_layout,
    tt::tt_metal::DataType input_dtype,
    tt::tt_metal::DataType output_dtype,
    const Conv2dConfig& conv_config,
    const DeviceComputeKernelConfig& compute_config,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_) {
    // Channel-chunk stitching is ROW_MAJOR; compressed TILE-only dtypes cannot be stored there.
    if (output_dtype == DataType::BFLOAT8_B || output_dtype == DataType::BFLOAT4_B) {
        return 1;
    }
    // use_matmul_for_1x1_conv ignores groups. A grouped 1x1 still becomes a matmul in
    // conv2d_DRAM (early-return above the chunk loop). Never probe get_L1_usage for mm_conv:
    // Conv2dSliceAttr::get_L1_usage fatals on mm_conv, and the L1_FULL estimate was unguarded.
    const bool mm_conv =
        use_matmul_for_1x1_conv(kernel_size, stride, padding_n4, dilation, groups, conv_config);
    if (mm_conv) {
        return 1;
    }

    const uint32_t kernel_taps = kernel_size[0] * kernel_size[1];
    const bool weight_is_host = !ttnn::is_device_tensor(weight_tensor);
    const bool conv_is_1d_depthwise = is_1d_depthwise_conv(
        groups, in_channels, out_channels, kernel_size[0], input_height, bias_tensor.has_value());
    const bool shard_is_height =
        conv_config.shard_layout.has_value() && conv_config.shard_layout.value() == TensorMemoryLayout::HEIGHT_SHARDED;
    // Prepared device weights are the 1D-depthwise layout only. is_device_tensor alone is not
    // enough: prepared grouped weights share the [1,1,H,C] shape class with the depthwise form.
    // do not infer depthwise from the weight shape.
    const bool weight_is_prepared_depthwise =
        conv_is_1d_depthwise && !weight_is_host && shard_is_height &&
        is_valid_device_conv_weights(weight_tensor, in_channels, out_channels, conv_config.weights_dtype);
    // Device bias chunks only alongside prepared depthwise weights: that bias is TILE with the
    // channel range in the last dim (prepare_conv_bias pads to round_up(C, TILE_WIDTH)), so a
    // TILE_WIDTH-aligned chunk end slices it in TILE exactly like the weights. Any other device
    // bias (host weights, or grouped-layout weights) is refused: it would need a host round-trip
    // per chunk.
    const bool bias_is_chunkable_device_bias =
        bias_tensor.has_value() && ttnn::is_device_tensor(bias_tensor.value()) && weight_is_prepared_depthwise &&
        is_valid_device_conv_bias(bias_tensor.value(), out_channels, conv_config.weights_dtype);
    const bool can_chunk_channels =
        groups > 1 && in_channels % groups == 0 && out_channels % groups == 0 &&
        (weight_is_host || weight_is_prepared_depthwise) &&
        (!bias_tensor.has_value() || !ttnn::is_device_tensor(bias_tensor.value()) || bias_is_chunkable_device_bias);
    if (!can_chunk_channels) {
        return 1;
    }

    Conv2dConfig estimate_conv_config = conv_config;
    if (weight_is_prepared_depthwise) {
        estimate_conv_config.act_block_h_override = weight_tensor.logical_shape()[2] / kernel_taps;
    }

    auto [output_height, output_width] =
        calculate_output_image_size({input_height, input_width}, kernel_size, stride, padding_n4, dilation);
    const uint32_t l1_available_bytes =
        device->allocator()->get_statistics(tt::tt_metal::BufferType::L1).total_free_bytes;
    auto single_call_l1_usage = [&](uint32_t est_in_channels, uint32_t est_out_channels, uint32_t est_groups) {
        Tensor weight_for_estimate = weight_tensor;
        std::optional<Tensor> bias_for_estimate = bias_tensor;
        auto attr = Conv2dSliceAttr(
            batch_size,
            {input_height, input_width},
            est_in_channels,
            est_out_channels,
            kernel_size,
            stride,
            padding_n4,
            dilation,
            est_groups,
            input_layout,
            input_dtype,
            output_dtype,
            weight_for_estimate,
            bias_for_estimate.has_value() ? std::make_optional(std::ref(bias_for_estimate.value())) : std::nullopt,
            estimate_conv_config,
            compute_config,
            device);
        return attr.get_L1_usage(
            {0, 0},
            {output_height, output_width},
            ttnn::operations::op_slicing::Op2DSliceConfig{
                .slice_type = ttnn::operations::op_slicing::Op2DSliceConfig::SliceType::DRAM_WIDTH,
                .num_slices = 1});
    };

    bool needs_channel_chunking = false;
    if (dram_slice_config_.has_value() &&
        dram_slice_config_.value().slice_type ==
            ttnn::operations::op_slicing::Op2DSliceConfig::SliceType::L1_FULL &&
        dram_slice_config_.value().num_slices <= 1) {
        needs_channel_chunking = single_call_l1_usage(in_channels, out_channels, groups) > l1_available_bytes;
    } else if (!dram_slice_config_.has_value() || dram_slice_config_.value().num_slices == 0) {
        Tensor weight_for_estimate = weight_tensor;
        std::optional<Tensor> bias_for_estimate = bias_tensor;
        auto attr = Conv2dSliceAttr(
            batch_size,
            {input_height, input_width},
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding_n4,
            dilation,
            groups,
            input_layout,
            input_dtype,
            output_dtype,
            weight_for_estimate,
            bias_for_estimate.has_value() ? std::make_optional(std::ref(bias_for_estimate.value())) : std::nullopt,
            estimate_conv_config,
            compute_config,
            device);
        auto maybe = ttnn::operations::op_slicing::try_determine_slice_config(
            &attr,
            ttnn::Shape({batch_size, input_height, input_width, in_channels}),
            ttnn::Shape({batch_size, output_height, output_width, out_channels}),
            std::nullopt,
            estimate_conv_config.output_layout,
            device);
        needs_channel_chunking = !maybe.has_value();
    }
    if (!needs_channel_chunking) {
        return 1;
    }
    // Smallest divisor of groups whose per-chunk channel counts are TILE_WIDTH-aligned
    // (slice_write RM interleaved copies the padded last dim; C=96/4=24 padded-to-32
    // overwrites; C=96/3=32 is legal and was missed by a power-of-two-only search)
    // and whose per-chunk L1 estimate fits.
    uint32_t num_channel_chunks = 1;
    for (uint32_t candidate_chunks = 2; candidate_chunks <= groups; ++candidate_chunks) {
        if (groups % candidate_chunks != 0) {
            continue;
        }
        const uint32_t chunk_out = out_channels / candidate_chunks;
        const uint32_t chunk_in = in_channels / candidate_chunks;
        if (chunk_out % tt::constants::TILE_WIDTH != 0 || chunk_in % tt::constants::TILE_WIDTH != 0) {
            continue;
        }
        if (single_call_l1_usage(chunk_in, chunk_out, groups / candidate_chunks) <= l1_available_bytes) {
            num_channel_chunks = candidate_chunks;
            break;
        }
    }
    if (num_channel_chunks > 1) {
        log_debug(
            tt::LogOp,
            "Conv2D DRAM: grouped conv with groups={} does not fit in available L1 ({} bytes) even with maximum "
            "spatial slicing; running as {} channel chunks of {} input channels each{}.",
            groups,
            l1_available_bytes,
            num_channel_chunks,
            in_channels / num_channel_chunks,
            weight_is_prepared_depthwise ? " (prepared device weights, chunked on device)" : "");
    } else {
        log_debug(
            tt::LogOp,
            "Conv2D DRAM: grouped conv with groups={} needs channel chunking but no TILE_WIDTH-aligned "
            "chunk count fits available L1 ({} bytes).",
            groups,
            l1_available_bytes);
    }
    return num_channel_chunks;
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
Result conv2d_DRAM(
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
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_) {
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    const DataType output_dtype = dtype.value_or(input_tensor.dtype());
    std::array<uint32_t, 4> padding_n4 = sliding_window::get_pair_n4_padding(padding);
    bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding_n4, dilation, groups, conv_config);
    // Use weights_dtype from config if set, otherwise use weight tensor's dtype
    DataType weight_dtype = conv_config.weights_dtype.value_or(weight_tensor.dtype());
    DeviceComputeKernelConfig compute_config =
        compute_config_.value_or(get_conv_default_compute_kernel_config(device, input_tensor.dtype(), weight_dtype));
    TT_FATAL(
        !conv_config.override_output_sharding_config,
        "Conv2D DRAM slicing doesn't support override_output_sharding_config.");

    // Fold the input tensor if required - this may update mm_conv after folding
    ttnn::Tensor input_tensor_on_device = fold_input_tensor_if_required(
        input_tensor,
        device,
        batch_size,
        input_height,
        input_width,
        in_channels,
        kernel_size,
        stride,
        dilation,
        padding_n4,
        mm_conv,
        conv_config);
    if (!is_device_tensor(input_tensor_on_device)) {
        input_tensor_on_device =
            ttnn::operations::core::to_device(input_tensor_on_device, device, ttnn::DRAM_MEMORY_CONFIG);
    }

    // After folding, check if this can be implemented as matmul and delegate to conv2d_L1
    // Note: mm_conv may have been updated by fold_input_tensor_if_required
    if (mm_conv) {
        return conv2d_L1(
            input_tensor_on_device,
            weight_tensor,
            device,
            in_channels,
            out_channels,
            batch_size,
            input_height,
            input_width,
            kernel_size,
            stride,
            padding_n4,
            dilation,
            groups,
            output_dtype,
            bias_tensor,
            conv_config,
            compute_config_,
            memory_config_);
    }

    // DRAM slicing path - only executed when mm_conv is false
    const bool should_deallocate_act = conv_config.deallocate_activation && !input_tensor.memory_config().is_dram();
    ttnn::Tensor weight_tensor_on_device;
    std::optional<ttnn::Tensor> bias_tensor_on_device;
    if (memory_config_.has_value()) {
        log_warning(
            tt::LogOp,
            "Conv2D DRAM doesn't support specifying memory config, as the output will always be DRAM Interleaved");
    }

    TT_FATAL(
        !(conv_config.output_layout == Layout::ROW_MAJOR && output_dtype == DataType::BFLOAT8_B),
        "Conv output can't be in Row Major if output dtype is BFloat8_B.");

    auto [output_height, output_width] =
        calculate_output_image_size({input_height, input_width}, kernel_size, stride, padding_n4, dilation);

    if (!conv_config.weights_dtype.has_value()) {
        conv_config.weights_dtype = weight_tensor.dtype();
    }

    const auto unflattened_input_shape = ttnn::Shape{batch_size, input_height, input_width, in_channels};
    input_tensor_on_device = ttnn::reshape(input_tensor_on_device, unflattened_input_shape, unflattened_input_shape);
    TT_FATAL(input_tensor_on_device.memory_config().is_dram(), "Conv DRAM expects the input tensor to be in DRAM.");
    TT_FATAL(
        input_tensor_on_device.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input Tensor to Conv DRAM should be in Interleaved Memory Layout");

    // Grouped convolutions that cannot satisfy the per-core L1 limit even with maximum
    // spatial DRAM slicing run as channel chunks instead (see
    // channel_chunk_count_if_needed for the decision and the two supported weight forms).
    // One decision, post-fold (fold mutates height/width/channels/conv_config). conv2d()
    // no longer probes this; L1_FULL grouped DRAM inputs enter here and bounce back to
    // conv2d_L1 when chunking is not needed so the L1 output contract is preserved.
    const uint32_t num_channel_chunks = channel_chunk_count_if_needed(
        weight_tensor,
        bias_tensor,
        device,
        batch_size,
        input_height,
        input_width,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding_n4,
        dilation,
        groups,
        input_tensor.layout(),
        input_tensor.dtype(),
        output_dtype,
        conv_config,
        compute_config,
        dram_slice_config_);

    if (num_channel_chunks == 1 && dram_slice_config_.has_value() &&
        dram_slice_config_->slice_type == Conv2dSliceConfig::SliceType::L1_FULL &&
        dram_slice_config_->num_slices <= 1) {
        return conv2d_L1(
            input_tensor_on_device,
            weight_tensor,
            device,
            in_channels,
            out_channels,
            batch_size,
            input_height,
            input_width,
            kernel_size,
            stride,
            padding_n4,
            dilation,
            groups,
            output_dtype,
            bias_tensor,
            conv_config,
            compute_config_,
            memory_config_);
    }

    // The channel-chunked path stitches chunk outputs into a ROW_MAJOR DRAM tensor:
    // slice_write's RM interleaved program factory is the only one that supports a
    // non-zero start offset on the last (channel) dimension. Compressed dtypes that
    // cannot be stored ROW_MAJOR (BFloat8_B / BFloat4_B) therefore cannot be chunked;
    // fail fast instead of constructing an invalid ROW_MAJOR allocation below.
    TT_FATAL(
        !(num_channel_chunks > 1 &&
          (output_dtype == DataType::BFLOAT8_B || output_dtype == DataType::BFLOAT4_B)),
        "Channel-chunked conv stitches its output through a ROW_MAJOR tensor; BFloat8_B / BFloat4_B output "
        "dtype is not supported on this path.");

    ttnn::Tensor dram_output_tensor = ttnn::create_device_tensor(
        tt::tt_metal::TensorSpec(
            ttnn::Shape({batch_size, output_height, output_width, out_channels}),
            tt::tt_metal::TensorLayout(
                output_dtype,
                tt::tt_metal::PageConfig(
                    num_channel_chunks > 1 ? tt::tt_metal::Layout::ROW_MAJOR : conv_config.output_layout),
                MemoryConfig{
                    TensorMemoryLayout::INTERLEAVED,
                    BufferType::DRAM,
                })),
        device);

    if (num_channel_chunks > 1) {
        const uint32_t chunk_in_channels = in_channels / num_channel_chunks;
        const uint32_t chunk_out_channels = out_channels / num_channel_chunks;
        const uint32_t chunk_groups = groups / num_channel_chunks;

        // Host OIHW weights: ttnn::prim::slice fatals on HOST tensors and round-tripping
        // each chunk through the device would serialize the loop on the host anyway.
        // Slice each chunk on the host with Tensor::unpad (host-only API); the chunk's own
        // weight preparation inside conv2d_L1 prepares that host slice exactly as the
        // unchunked path prepares the full weight.
        // Prepared depthwise device weights ([1, 1, act_block_h*32*K, C_padded], TILE):
        // slice each chunk's channel range (last dim) directly in TILE — TILE_WIDTH-aligned
        // chunk ends land on tile boundaries so SliceTileProgramFactory applies without
        // untilize/re-tilize or host re-preparation, and the chunk conv consumes the TILE
        // slice as-is (valid device weights), which keeps the path safe under trace capture.
        // act_block_h is pinned to the prepared tap slab height so the kernel reads each
        // kernel tap's slab at its prepared boundaries.
        const bool chunk_prepared_device_weights =
            is_1d_depthwise_conv(
                groups, in_channels, out_channels, kernel_size[0], input_height, bias_tensor.has_value()) &&
            ttnn::is_device_tensor(weight_tensor);
        Conv2dConfig chunk_conv_config = conv_config;
        if (chunk_prepared_device_weights) {
            const uint32_t prepared_tap_slab_rows =
                weight_tensor.logical_shape()[2] / (kernel_size[0] * kernel_size[1]);
            chunk_conv_config.act_block_h_override = prepared_tap_slab_rows;
            log_debug(
                tt::LogOp,
                "Conv2D DRAM channel chunking: prepared device weights with {} rows per kernel tap slab.",
                prepared_tap_slab_rows);
        }

        for (uint32_t chunk_index = 0; chunk_index < num_channel_chunks; chunk_index++) {
            const uint32_t in_channels_begin = chunk_index * chunk_in_channels;
            const uint32_t out_channels_begin = chunk_index * chunk_out_channels;

            Tensor chunk_weight;
            if (chunk_prepared_device_weights) {
                chunk_weight = ttnn::slice(
                    weight_tensor,
                    ttsl::SmallVector<uint32_t>{0, 0, 0, out_channels_begin},
                    ttsl::SmallVector<uint32_t>{1, 1, weight_tensor.logical_shape()[2], out_channels_begin + chunk_out_channels},
                    ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
            } else {
                chunk_weight = weight_tensor.unpad(
                    tt::tt_metal::Shape({out_channels_begin, 0, 0, 0}),
                    tt::tt_metal::Shape(
                        {out_channels_begin + chunk_out_channels,
                         weight_tensor.logical_shape()[1],
                         weight_tensor.logical_shape()[2],
                         weight_tensor.logical_shape()[3]}));
            }
            std::optional<Tensor> chunk_bias = std::nullopt;
            if (bias_tensor.has_value()) {
                const auto& bias_shape = bias_tensor.value().logical_shape();
                if (chunk_prepared_device_weights) {
                    // Prepared depthwise bias is TILE with the channel range in the last dim
                    // ([1, 1, 32, C_padded]); TILE_WIDTH-aligned chunk ends slice it in TILE exactly
                    // like the weights above, with no host round-trip.
                    chunk_bias = ttnn::slice(
                        bias_tensor.value(),
                        ttsl::SmallVector<uint32_t>{0, 0, 0, out_channels_begin},
                        ttsl::SmallVector<uint32_t>{1, 1, bias_shape[2], out_channels_begin + chunk_out_channels},
                        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
                } else {
                    chunk_bias = bias_tensor.value().unpad(
                        tt::tt_metal::Shape({0, 0, 0, out_channels_begin}),
                        tt::tt_metal::Shape(
                            {bias_shape[0], bias_shape[1], bias_shape[2], out_channels_begin + chunk_out_channels}));
                }
            }
            auto chunk_attr = Conv2dSliceAttr(
                batch_size,
                {input_height, input_width},
                chunk_in_channels,
                chunk_out_channels,
                kernel_size,
                stride,
                padding_n4,
                dilation,
                chunk_groups,
                input_tensor.layout(),
                input_tensor.dtype(),
                output_dtype,
                chunk_weight,
                chunk_bias.has_value() ? std::make_optional(std::ref(chunk_bias.value())) : std::nullopt,
                chunk_conv_config,
                compute_config,
                device);

            // Isolate this chunk's channel window with ttnn::slice (the output shard's
            // row size equals chunk_in_channels, so the reader copies exactly the
            // chunk's channels and never a neighbour's), then padded_slice places the
            // isolated window into the sharded input config the chunk's conv expects.
            // padded_slice must start at channel 0 of the isolated tensor: its C++
            // pad_value parameter is ignored (the pad writer only zeroes what it fills
            // when the shard row is wider than the input row), so slicing the full
            // tensor with in_channels_begin != 0 would copy real neighbour channels
            // into the pad region instead of zeros.
            auto chunk_input_memory_config = chunk_attr.get_input_memory_config({0, 0}, {output_height, output_width});
            const Tensor chunk_input_window = ttnn::slice(
                input_tensor_on_device,
                ttsl::SmallVector<uint32_t>{0, 0, 0, in_channels_begin},
                ttsl::SmallVector<uint32_t>{
                    batch_size, input_height, input_width, in_channels_begin + chunk_in_channels},
                ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
            const Tensor chunk_input_tensor = ttnn::experimental::padded_slice(
                chunk_input_window,
                ttsl::SmallVector<uint32_t>{0, 0, 0, 0},
                ttsl::SmallVector<uint32_t>{batch_size, input_height, input_width, chunk_in_channels},
                ttsl::SmallVector<uint32_t>{1, 1, 1, 1},
                chunk_input_memory_config,
                std::nullopt,
                0.f);

            auto chunk_output_tensors = chunk_attr.run_L1_op(chunk_input_tensor, {0, 0}, {output_height, output_width});
            TT_FATAL(
                chunk_output_tensors.size() == 1, "Channel-chunked conv must produce exactly one output tensor.");
            Tensor chunk_output_tensor = chunk_output_tensors[0];

            // Bring the chunk output to ROW_MAJOR interleaved: slice_write's RM
            // interleaved factory is the only one that supports a non-zero start on the
            // last (channel) dimension, which channel stitching requires.
            if (chunk_output_tensor.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED) {
                chunk_output_tensor = ttnn::to_memory_config(
                    chunk_output_tensor, MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1});
            }
            if (chunk_output_tensor.layout() != Layout::ROW_MAJOR) {
                chunk_output_tensor = ttnn::untilize(chunk_output_tensor);
            }
            chunk_output_tensor = ttnn::reshape(
                chunk_output_tensor,
                ttnn::Shape({batch_size, output_height, output_width, chunk_out_channels}),
                ttnn::Shape({batch_size, output_height, output_width, chunk_output_tensor.padded_shape()[3]}));
            TT_FATAL(
                chunk_output_tensor.padded_shape()[3] == chunk_out_channels,
                "Channel-chunk slice_write requires padded last dim == logical chunk_out_channels (padded={}, "
                "logical={}). Candidates whose out_channels/chunks is not a multiple of TILE_WIDTH are rejected.",
                chunk_output_tensor.padded_shape()[3],
                chunk_out_channels);
            ttnn::experimental::slice_write(
                chunk_output_tensor,
                dram_output_tensor,
                ttsl::SmallVector<uint32_t>{0, 0, 0, out_channels_begin},
                ttsl::SmallVector<uint32_t>{
                    batch_size, output_height, output_width, out_channels_begin + chunk_out_channels},
                ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
        }

        if (should_deallocate_act) {
            input_tensor_on_device.deallocate(true);
        }
        // Stitching happened through a ROW_MAJOR intermediate because slice_write needs
        // it; convert back to the caller-requested layout so chunked calls honor
        // conv_config.output_layout exactly like the stock DRAM path. Values are
        // unchanged - this is a layout hop only.
        if (conv_config.output_layout != tt::tt_metal::Layout::ROW_MAJOR) {
            dram_output_tensor = ttnn::to_layout(dram_output_tensor, conv_config.output_layout);
        }
        const auto flattened_output_shape = flatten_4d_shape(dram_output_tensor.logical_shape());
        const auto flattened_padded_output_shape = flatten_4d_shape(dram_output_tensor.padded_shape());
        dram_output_tensor = ttnn::reshape(dram_output_tensor, flattened_output_shape, flattened_padded_output_shape);

        // return_weights_and_bias contract on the chunked path: with host weights, every
        // chunk prepares (and discards) its own channel-sliced device weights, so there is
        // no single prepared device tensor to hand back; the caller's original host
        // weight/bias are returned so a repeated call re-runs preparation identically.
        // Prepared device weights are returned unchanged and stay reusable.
        return {dram_output_tensor, output_height, output_width, weight_tensor, bias_tensor};
    }

    weight_tensor_on_device = weight_tensor;
    bias_tensor_on_device = bias_tensor;
    auto slice_attr = Conv2dSliceAttr(
        batch_size,
        {input_height, input_width},
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding_n4,
        dilation,
        groups,
        input_tensor.layout(),
        input_tensor.dtype(),
        output_dtype,
        std::ref(weight_tensor_on_device),
        bias_tensor_on_device.has_value() ? std::make_optional(std::ref(bias_tensor_on_device.value())) : std::nullopt,
        conv_config,
        compute_config,
        device);

    std::vector<std::reference_wrapper<Tensor>> output_tensors = {std::ref(dram_output_tensor)};
    ttnn::operations::op_slicing::run_sliced_op(
        input_tensor_on_device, output_tensors, &slice_attr, dram_slice_config_);

    if (should_deallocate_act) {
        input_tensor_on_device.deallocate(true);
    }
    const auto flattened_output_shape = flatten_4d_shape(dram_output_tensor.logical_shape());
    const auto flattened_padded_output_shape = flatten_4d_shape(dram_output_tensor.padded_shape());

    dram_output_tensor = ttnn::reshape(dram_output_tensor, flattened_output_shape, flattened_padded_output_shape);

    return {dram_output_tensor, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
}

}  // namespace ttnn::operations::conv::conv2d

namespace ttnn {

Conv2dResultWithOptions conv2d(
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
    using namespace operations::conv::conv2d;
    using operations::conv::Conv2dExecutionPath;
    using operations::conv::determine_conv2d_execution_path;
    // Determine execution path based on configuration and input properties
    Conv2dExecutionPath path = determine_conv2d_execution_path(input_tensor, slice_config_);

    // Grouped L1_FULL over a DRAM interleaved input: send to conv2d_DRAM so the
    // chunk decision runs once, post-fold. conv2d_DRAM bounces back to conv2d_L1
    // when chunking is not needed (L1 output preserved). Ungrouped L1_FULL is
    // unchanged. Do not probe channel_chunk_count_if_needed here (pre-fold vs
    // post-fold disagreed; dual L1 search).
    if (path == Conv2dExecutionPath::L1 && groups > 1 && slice_config_.has_value() &&
        slice_config_->slice_type == Conv2dSliceConfig::SliceType::L1_FULL && slice_config_->num_slices <= 1 &&
        ttnn::is_device_tensor(input_tensor) &&
        input_tensor.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM &&
        input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        log_debug(
            tt::LogOp,
            "Conv2D L1_FULL grouped conv: routing to the DRAM path for a post-fold channel-chunk decision.");
        path = Conv2dExecutionPath::DRAM;
    }

    // Execute L1 path
    if (path == Conv2dExecutionPath::L1) {
        log_trace(tt::LogOp, "Conv2d L1 {}", slice_config_.has_value() ? "with slice config" : "without slice config");
        return result_to_result_with_options(
            conv2d_L1(
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
                memory_config),
            return_output_dim,
            return_weights_and_bias);
    }

    // Execute DRAM path
    log_trace(tt::LogOp, "Conv2d DRAM {}", slice_config_.has_value() ? "with slice config" : "without slice config");
    return result_to_result_with_options(
        conv2d_DRAM(
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
            memory_config,
            slice_config_),
        return_output_dim,
        return_weights_and_bias);
}

}  // namespace ttnn

namespace ttnn::operations::conv::conv2d {

std::unique_ptr<op_slicing::OpSliceAttr> get_conv2d_slice_attr(
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t in_channels,
    uint32_t out_channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 4> padding_n4,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    Layout input_layout,
    DataType input_dtype,
    DataType conv_output_dtype,
    Tensor& weight_tensor,
    std::optional<std::reference_wrapper<Tensor>> bias_tensor,
    const Conv2dConfig& conv_config_,
    const DeviceComputeKernelConfig& compute_config,
    MeshDevice* device) {
    return std::unique_ptr<op_slicing::OpSliceAttr>(new Conv2dSliceAttr(
        batch_size,
        {input_height, input_width},
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding_n4,
        dilation,
        groups,
        input_layout,
        input_dtype,
        conv_output_dtype,
        weight_tensor,
        bias_tensor,
        conv_config_,
        compute_config,
        device));
}
}  // namespace ttnn::operations::conv::conv2d
