// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
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

#include "tt-metalium/math.hpp"
#include "ttnn/operations/sliding_window/op_slicing/op_slicing.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::operations::conv::conv2d {

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
            tt::tt_metal::is_device_tensor(input_tensor) ? std::make_optional(input_tensor.memory_config())
                                                         : std::nullopt,
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

    const bool conv_is_1d_depthwise = is_1d_depthwise_conv(
        groups, in_channels, out_channels, kernel_size[0], kernel_size[1], input_height, bias_tensor.has_value());

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
        conv_is_1d_depthwise);

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
        orig_stride);

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
    bool input_is_on_device = tt::tt_metal::is_device_tensor(input_tensor_post_tm);
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
        };

        if (parallel_config.shard_scheme != TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor_post_tm.layout() != Layout::ROW_MAJOR || sliding_window_config.get_pad_h() != 0 ||
            sliding_window_config.get_pad_w() != 0) {
            ttnn::Tensor halo_output = ttnn::halo(
                input_tensor_post_tm,
                sliding_window_config,
                0,
                false,
                parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
                input_tensor_post_tm.memory_config(),
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
    using RefTensor = std::reference_wrapper<ttnn::Tensor>;

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

        auto [input_start, input_end] = get_input_slice(output_slice_start, output_slice_end);
        uint32_t input_slice_height = std::get<0>(input_end) - std::get<0>(input_start);
        uint32_t input_slice_width = std::get<1>(input_end) - std::get<1>(input_start);
        uint32_t output_slice_height = std::get<0>(output_slice_end) - std::get<0>(output_slice_start);
        uint32_t output_slice_width = std::get<1>(output_slice_end) - std::get<1>(output_slice_start);

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

    ttnn::Tensor dram_output_tensor = tt::tt_metal::create_device_tensor(
        tt::tt_metal::TensorSpec(
            ttnn::Shape({batch_size, output_height, output_width, out_channels}),
            tt::tt_metal::TensorLayout(
                output_dtype,
                tt::tt_metal::PageConfig(conv_config.output_layout),
                MemoryConfig{
                    TensorMemoryLayout::INTERLEAVED,
                    BufferType::DRAM,
                })),
        device);

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

ResultWithOptions Conv2dOperation::invoke(
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
    // Determine execution path based on configuration and input properties
    Conv2dExecutionPath path = determine_conv2d_execution_path(input_tensor, slice_config_);

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
