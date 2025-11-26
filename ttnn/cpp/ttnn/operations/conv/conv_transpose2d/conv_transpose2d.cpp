// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/conv/conv_transpose2d/conv_transpose2d.hpp"

#include <array>
#include <tt-logger/tt-logger.hpp>
#include <tuple>
#include <utility>

#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/conv/conv_transpose2d/prepare_conv_transpose2d_weights.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"

namespace ttnn::operations::conv::conv_transpose2d {


using namespace tt;
using sliding_window::SlidingWindowConfig;

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

ResultWithOptions conv_transpose2d(
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
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    const std::optional<const DataType>& dtype,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config_,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_,
    bool mirror_kernel,
    bool return_output_dim,
    bool return_weights_and_bias) {
    if (dram_slice_config_.has_value()) {
        if (dram_slice_config_.value().slice_type == Conv2dSliceConfig::SliceType::L1_FULL) {
            log_trace(tt::LogOp, "Conv2d L1 with slice config {}", dram_slice_config_);
            return result_to_result_with_options(
                conv_transpose2d_L1(
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
                    dtype,
                    bias_tensor,
                    conv_config_,
                    compute_config_,
                    memory_config_,
                    mirror_kernel),
                return_output_dim,
                return_weights_and_bias);
        } else {
            log_trace(tt::LogOp, "Conv2d DRAM with slice config {}", dram_slice_config_);
            return result_to_result_with_options(
                conv_transpose2d_DRAM(
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
                    dtype,
                    bias_tensor,
                    conv_config_,
                    compute_config_,
                    memory_config_,
                    dram_slice_config_,
                    mirror_kernel),
                return_output_dim,
                return_weights_and_bias);
        }
    } else {
        // bool input_is_on_device = tt::tt_metal::is_device_tensor(input_tensor);
        // if (input_is_on_device && input_tensor.memory_config().is_l1()) {
        log_trace(tt::LogOp, "Conv2d L1 without slice config");
        return result_to_result_with_options(
            conv_transpose2d_L1(
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
                dtype,
                bias_tensor,
                conv_config_,
                compute_config_,
                memory_config_,
                mirror_kernel),
            return_output_dim,
            return_weights_and_bias);
        // }
        // log_trace(tt::LogOp, "Conv2d DRAM without slice config");
        // return result_to_result_with_options(
        //     conv_transpose2d_DRAM(
        //         input_tensor,
        //         weight_tensor,
        //         device,
        //         in_channels,
        //         out_channels,
        //         batch_size,
        //         input_height,
        //         input_width,
        //         kernel_size,
        //         stride,
        //         padding,
        //         output_padding,
        //         dilation,
        //         groups,
        //         dtype,
        //         bias_tensor,
        //         conv_config_,
        //         compute_config_,
        //         memory_config_,
        //         std::nullopt,
        //         mirror_kernel),
        //     return_output_dim,
        //     return_weights_and_bias);
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
Result conv_transpose2d_DRAM(
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
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    const std::optional<const DataType>& dtype,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config_,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_,
    bool mirror_kernel) {
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    const DataType output_dtype = dtype.value_or(input_tensor.dtype());
    DeviceComputeKernelConfig compute_config = compute_config_.value_or(get_conv_default_compute_kernel_config(device));
    auto padding_n4 = sliding_window::get_pair_n4_padding(padding);

    // Inverse of sliding_window.get_output_shape()
    SlidingWindowConfig sliding_window_config = SlidingWindowConfig{
        .batch_size = batch_size,
        .input_hw = {input_height, input_width},
        .window_hw = {kernel_size[0], kernel_size[1]},
        .stride_hw = {stride[0], stride[1]},
        .padding = padding_n4,
        .output_pad_hw = {output_padding[0], output_padding[1]},
        .dilation_hw = {dilation[0], dilation[1]},
        .is_transpose = true};

    // ConvTranspose2d is implemented via the Conv2d u_op with flipped weights.
    // The input tensor is first passed to the halo op that paddeds the input.
    // In the scenario, where stride > 1, the halo op will add interleaved 0s to the input tensor.
    // The Conv2d u_op is then called with stride = 1, padding = 0.
    // SlidingWindowConfig has a is_transpose flag that is set to true to indicate that the Conv2d u_op & Halo u_op is
    // being called for ConvTranspose2d.
    uint32_t output_height = ((input_height - 1) * stride[0]) - (padding_n4[0] + padding_n4[1]) +
                             (dilation[0] * (kernel_size[0] - 1)) + output_padding[0] + 1;
    uint32_t output_width = ((input_width - 1) * stride[1]) - (padding_n4[2] + padding_n4[3]) +
                            (dilation[1] * (kernel_size[1] - 1)) + output_padding[1] + 1;

    // Dimensions of Input to Conv u_op
    uint32_t full_input_height = output_height + (dilation[0] * (kernel_size[0] - 1));
    uint32_t full_input_width = output_width + (dilation[1] * (kernel_size[1] - 1));

    // Size of input after adding interleaved 0s.
    uint32_t strided_input_height = ((input_height - 1) * stride[0]) + 1;
    uint32_t strided_input_width = ((input_width - 1) * stride[1]) + 1;

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
    Tensor input_tensor_on_device = input_tensor;
    if (!is_device_tensor(input_tensor_on_device)) {
        input_tensor_on_device =
            ttnn::operations::core::to_device(input_tensor_on_device, device, ttnn::DRAM_MEMORY_CONFIG);
    }
    ttnn::Tensor weight_tensor_on_device;
    std::optional<ttnn::Tensor> bias_tensor_on_device;
    if (mm_conv) {
        const uint32_t input_channels_alignment = get_input_channels_alignment(
            tt::tt_metal::TensorMemoryLayout::INTERLEAVED, input_tensor.layout(), true, mm_conv, std::nullopt);
        // Configure weight and bias preparation parameters
        Conv2dWeightsBiasPrepConfig params(
            input_channels_alignment,
            conv_config.weights_dtype,
            1,  // Set to 1 as it doesn't matter in Interleaved MM.
            1,  // Set to 1 as it doesn't matter in Interleaved MM.
            std::nullopt,
            std::nullopt,
            groups,
            1,  // Set to 1 as it doesn't matter in Interleaved MM.
            input_width,
            true,  // DRAM MM Convs are always interleaved
            bias_tensor.has_value(),
            true,  // parameters_on_device
            conv_config.enable_kernel_stride_folding.value(),
            conv_config.full_inner_dim,
            conv_config.enable_activation_reuse,
            kernel_size,
            stride,
            padding_n4);

        // Prepare weights and move to device if necessary
        if (!is_device_tensor(weight_tensor)) {
            log_debug(tt::LogOp, "conv2d: Preprocessing weights for MM DRAM Interleaved.");
            std::tie(weight_tensor_on_device, bias_tensor_on_device) =
                prepare_conv_weights_biases_and_move_to_device(weight_tensor, bias_tensor, params, device);
        } else {
            weight_tensor_on_device = weight_tensor;
            if (bias_tensor.has_value()) {
                bias_tensor_on_device = bias_tensor.value();
            }
        }
        // run conv as matmul
        std::optional<ttnn::operations::matmul::MatmulProgramConfig> program_config = std::nullopt;
        std::optional<MemoryConfig> mm_output_memory_config = std::nullopt;

        // Matmul expects inputs to be in Tile Layout
        tilize_with_optional_deallocation(input_tensor_on_device, conv_config.deallocate_activation);
        Tensor matmul_output = ttnn::linear(
            input_tensor_on_device,
            weight_tensor_on_device,
            bias_tensor_on_device,
            false,
            false,
            mm_output_memory_config,
            output_dtype,
            program_config,
            conv_config.activation,
            compute_config);
        if (conv_config.deallocate_activation) {
            input_tensor_on_device.deallocate(true);
        }
        return {matmul_output, input_height, input_width, weight_tensor_on_device, bias_tensor_on_device};
    }
    if (memory_config_.has_value()) {
        log_warning(
            tt::LogOp,
            "Conv2D DRAM doesn't support specifying memory config, as the output will always be DRAM Interleaved");
    }

    TT_FATAL(
        !(conv_config.output_layout == Layout::ROW_MAJOR && output_dtype == DataType::BFLOAT8_B),
        "Conv output can't be in Row Major if output dtype is BFloat8_B.");

    if (!conv_config.weights_dtype.has_value()) {
        conv_config.weights_dtype = weight_tensor.dtype();
    }
    TT_FATAL(dram_slice_config_.has_value(), "DRAM Auto Slicing not supported for conv_transpose2d.");
    Conv2dSliceConfig dram_slice_config = dram_slice_config_.value();
    // std::tie(dram_slice_config, conv_config) = determine_conv2d_slice_config(
    //     dram_slice_config_,
    //     ConvDRAMParamters{
    //         .in_channels = in_channels,
    //         .out_channels = out_channels,
    //         .batch_size = batch_size,
    //         .input_height = input_height,
    //         .input_width = input_width,
    //         .output_height = output_height,
    //         .output_width = output_width,
    //         .kernel_size = kernel_size,
    //         .stride = stride,
    //         .padding_n4 = padding_n4,
    //         .dilation = dilation,
    //         .groups = groups,
    //         .conv_config = conv_config,
    //         .compute_kernel_config = compute_config,
    //         .compute_grid = compute_grid_size,
    //         .weights_datatype = conv_config.weights_dtype.value_or(weight_tensor.dtype()),
    //         .input_datatype = input_tensor.dtype(),
    //         .output_datatype = output_dtype,
    //         .input_layout = input_tensor.layout(),
    //         .enable_bias = bias_tensor.has_value(),
    //         .mm_conv = mm_conv,
    //     },
    //     device);
    log_debug(tt::LogOp, "Conv2D DRAM with Slice Config {}", dram_slice_config);
    TT_FATAL(dram_slice_config.num_slices > 0, " Number of slices should be greater than 0 for Conv2D DRAM Slicing");

    const uint32_t output_sliced_dim =
        dram_slice_config.slice_type == Conv2dSliceConfig::SliceType::DRAM_HEIGHT ? output_height : output_width;

    if (output_sliced_dim == 1) {
        dram_slice_config.num_slices = 1;
    } else {
        TT_ASSERT(
            dram_slice_config.num_slices < output_sliced_dim,
            " Number of slices {} should be less than the dimension {} being sliced in Conv2D DRAM Slicing",
            dram_slice_config.num_slices,
            output_sliced_dim);
    }
    if (dram_slice_config.num_slices == 1) {
        return conv_transpose2d_L1(
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
            dtype,
            bias_tensor,
            conv_config_,
            compute_config_,
            DRAM_MEMORY_CONFIG,
            mirror_kernel);
    }
    const auto unflattened_input_shape = ttnn::Shape{batch_size, input_height, input_width, in_channels};
    input_tensor_on_device = ttnn::reshape(input_tensor_on_device, unflattened_input_shape, unflattened_input_shape);
    TT_FATAL(input_tensor_on_device.memory_config().is_dram(), "Conv DRAM expects the input tensor to be in DRAM.");
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

    weight_tensor_on_device = weight_tensor;
    bias_tensor_on_device = bias_tensor;
    auto slice_attr = ConvT2DSliceAttr(
        batch_size,
        {input_height, input_width},
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding_n4,
        output_padding,
        dilation,
        groups,
        input_tensor.layout(),
        input_tensor.dtype(),
        output_dtype,
        std::ref(weight_tensor_on_device),
        bias_tensor_on_device.has_value() ? std::make_optional(std::ref(bias_tensor_on_device.value())) : std::nullopt,
        conv_config,
        compute_config,
        device,
        mirror_kernel);

    ttnn::operations::op_slicing::run_sliced_op(
        input_tensor_on_device, dram_output_tensor, &slice_attr, dram_slice_config);

    if (conv_config.deallocate_activation) {
        input_tensor_on_device.deallocate(true);
    }
    const auto flattened_output_shape = flatten_4d_shape(dram_output_tensor.logical_shape());
    const auto flattened_padded_output_shape = flatten_4d_shape(dram_output_tensor.padded_shape());

    dram_output_tensor = ttnn::reshape(dram_output_tensor, flattened_output_shape, flattened_padded_output_shape);

    return {dram_output_tensor, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
}

Result conv_transpose2d_L1(
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
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding_,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    const std::optional<const DataType>& dtype,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config,
    bool mirror_kernel) {
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    const DataType output_dtype = dtype.value_or(input_tensor.dtype());
    DeviceComputeKernelConfig compute_config = compute_config_.value_or(get_conv_default_compute_kernel_config(device));
    auto padding = sliding_window::get_pair_n4_padding(padding_);
    // Inverse of sliding_window.get_output_shape()
    sliding_window::SlidingWindowConfig sliding_window_config = sliding_window::SlidingWindowConfig{
        .batch_size = batch_size,
        .input_hw = {input_height, input_width},
        .window_hw = {kernel_size[0], kernel_size[1]},
        .stride_hw = {stride[0], stride[1]},
        .padding = padding,
        .output_pad_hw = {output_padding[0], output_padding[1]},
        .dilation_hw = {dilation[0], dilation[1]},
        .is_transpose = true};

    // ConvTranspose2d is implemented via the Conv2d u_op with flipped weights.
    // The input tensor is first passed to the halo op that paddeds the input.
    // In the scenario, where stride > 1, the halo op will add interleaved 0s to the input tensor.
    // The Conv2d u_op is then called with stride = 1, padding = 0.
    // SlidingWindowConfig has a is_transpose flag that is set to true to indicate that the Conv2d u_op & Halo u_op is
    // being called for ConvTranspose2d.
    uint32_t output_height = ((input_height - 1) * stride[0]) - (padding[0] + padding[1]) +
                             (dilation[0] * (kernel_size[0] - 1)) + output_padding[0] + 1;
    uint32_t output_width = ((input_width - 1) * stride[1]) - (padding[2] + padding[3]) +
                            (dilation[1] * (kernel_size[1] - 1)) + output_padding[1] + 1;

    // Dimensions of Input to Conv u_op
    uint32_t full_input_height = output_height + (dilation[0] * (kernel_size[0] - 1));
    uint32_t full_input_width = output_width + (dilation[1] * (kernel_size[1] - 1));

    // Size of input after adding interleaved 0s.
    uint32_t strided_input_height = ((input_height - 1) * stride[0]) + 1;
    uint32_t strided_input_width = ((input_width - 1) * stride[1]) + 1;

    uint32_t base_pad_height = dilation[0] * (kernel_size[0] - 1);
    uint32_t base_pad_width = dilation[1] * (kernel_size[1] - 1);
    uint32_t input_pad_top = base_pad_height - padding[0];
    uint32_t input_pad_bottom = base_pad_height - padding[1];

    uint32_t input_pad_left = base_pad_width - padding[2];
    uint32_t input_pad_right = base_pad_width - padding[3];

    log_info(tt::LogOp, "Input : {}x{}", input_height, input_width);
    log_info(tt::LogOp, "Output : {}x{}", output_height, output_width);

    log_info(tt::LogOp, "Conv Op Input : {}x{}", full_input_height, full_input_width);
    log_info(tt::LogOp, "Strided Input : {}x{}", strided_input_height, strided_input_width);

    log_info(tt::LogOp, "Padding : ({},{}) ({},{})", input_pad_top, input_pad_bottom, input_pad_left, input_pad_right);

    TT_FATAL(
        full_input_height - strided_input_height == input_pad_top + input_pad_bottom + output_padding[0],
        "Calculated input padding height does not match the expected value.");
    TT_FATAL(
        full_input_width - strided_input_width == input_pad_left + input_pad_right + output_padding[1],
        "Calculated input padding width does not match the expected value.");

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
            stride,
            dilation,
            padding,
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
            input_tensor_post_tm,
            sliding_window_config,
            0,
            false,
            parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
            input_tensor_post_tm.memory_config(),
            /*is_out_tiled*/ true,
            conv_config.config_tensors_in_dram);

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

    auto opt_conv_op_parallel_config = determine_conv_op_parallel_config_from_conv_output_mem_config(
        conv_out_memory_config,
        get_num_cores_nhw_from_parallel_config(parallel_config),
        get_num_cores_channels_from_parallel_config(parallel_config),
        get_num_cores_channels_from_parallel_config(output_parallel_config));

    const uint32_t input_channels_alignment = get_input_channels_alignment(
        input_tensor_post_tm.memory_config().memory_layout(),
        input_tensor.layout(),
        false,
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
        output_width,
        get_fp32_dest_acc_en(compute_config),
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
            mm_conv && auto_shard,
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
        return {matmul_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
    }
    // call conv micro op
    auto conv_output = ttnn::operations::conv::conv2d::conv2d(
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
        false,  // full_inner_dim
        false,  // enable_activation_reuse
        conv_config.config_tensors_in_dram,
        conv_config.force_split_reader);
    if (memory_config.has_value() && memory_config.value() != conv_output.memory_config()) {
        conv_output = ttnn::to_memory_config(conv_output, memory_config.value(), std::nullopt);
    }
    return {conv_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
}

ConvT2DSliceAttr::ConvT2DSliceAttr(
    uint32_t batch_size,
    IOShape input_shape,
    uint32_t input_channels,
    uint32_t output_channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 4> padding_n4,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    Layout input_layout,
    DataType input_dtype,
    DataType output_dtype,
    Tensor& weight_tensor,
    OptionalRefTensor bias_tensor,
    Conv2dConfig& conv_config,
    DeviceComputeKernelConfig& compute_config,
    MeshDevice* device,
    bool mirror_kernel) :
    batch_size(batch_size),
    input_shape(input_shape),
    input_channels(input_channels),
    output_channels(output_channels),
    kernel_size(kernel_size),
    stride(stride),
    padding_n4(padding_n4),
    output_padding(output_padding),
    dilation(dilation),
    groups(groups),
    input_layout(input_layout),
    input_dtype(input_dtype),
    output_dtype(output_dtype),
    weight_tensor(weight_tensor),
    bias_tensor(bias_tensor),
    conv_config(conv_config),
    compute_config(compute_config),
    device(device),
    mirror_kernel(mirror_kernel),
    input_padding() {
    auto [input_height, input_width] = input_shape;
    uint32_t output_height = ((input_height - 1) * stride[0]) - (padding_n4[0] + padding_n4[1]) +
                             (dilation[0] * (kernel_size[0] - 1)) + output_padding[0] + 1;
    uint32_t output_width = ((input_width - 1) * stride[1]) - (padding_n4[2] + padding_n4[3]) +
                            (dilation[1] * (kernel_size[1] - 1)) + output_padding[1] + 1;

    // Dimensions of Input to Conv u_op
    uint32_t full_input_height = output_height + (dilation[0] * (kernel_size[0] - 1));
    uint32_t full_input_width = output_width + (dilation[1] * (kernel_size[1] - 1));

    // Size of input after adding interleaved 0s.
    uint32_t strided_input_height = ((input_height - 1) * stride[0]) + 1;
    uint32_t strided_input_width = ((input_width - 1) * stride[1]) + 1;

    uint32_t input_pad_top = (full_input_height - strided_input_height) / 2;
    uint32_t input_pad_bottom = full_input_height - strided_input_height - input_pad_top;

    uint32_t input_pad_left = (full_input_width - strided_input_width) / 2;
    uint32_t input_pad_right = full_input_width - strided_input_width - input_pad_left;

    full_input_shape = {full_input_height, full_input_width};
    strided_input_shape = {strided_input_height, strided_input_width};
    this->input_padding = {input_pad_top, input_pad_bottom, input_pad_left, input_pad_right};
}

std::tuple<ConvT2DSliceAttr::IOShape, ConvT2DSliceAttr::IOShape> ConvT2DSliceAttr::get_input_slice(
    IOShape output_slice_start, IOShape output_slice_end) {
    auto [input_height, input_width] = input_shape;
    int output_slice_height_start, output_slice_width_start;
    int output_slice_height_end, output_slice_width_end;
    std::tie(output_slice_height_start, output_slice_width_start) = output_slice_start;
    std::tie(output_slice_height_end, output_slice_width_end) = output_slice_end;
    int input_slice_height_start = div_up((output_slice_height_start - (int)padding_n4[0]), (int)stride[0]);
    int input_slice_width_start = div_up((output_slice_width_start - (int)padding_n4[2]), (int)stride[1]);

    int input_slice_height_end = div_up(
        (output_slice_height_end - input_padding[0] + ((kernel_size[0] - 1) * (dilation[0] - 1)) + kernel_size[0] - 1),
        (int)stride[0]);
    int input_slice_width_end = div_up(
        (output_slice_width_end - input_padding[2] + ((kernel_size[1] - 1) * (dilation[1] - 1)) + kernel_size[1] - 1),
        (int)stride[1]);

    // Clamp input slice to valid input tensor bounds
    input_slice_height_start = std::max<int>(0, input_slice_height_start);
    input_slice_height_end = std::min<int>(input_height, input_slice_height_end);
    input_slice_width_start = std::max<int>(0, input_slice_width_start);
    input_slice_width_end = std::min<int>(input_width, input_slice_width_end);

    log_info(
        tt::LogOp,
        " ConvT2D Output Slice: H({}-{}), W({}-{}) maps to Input Slice H({}-{}), W({}-{})",
        output_slice_height_start,
        output_slice_height_end,
        output_slice_width_start,
        output_slice_width_end,
        input_slice_height_start,
        input_slice_height_end,
        input_slice_width_start,
        input_slice_width_end);
    return {{input_slice_height_start, input_slice_width_start}, {input_slice_height_end, input_slice_width_end}};
}

uint32_t ConvT2DSliceAttr::get_L1_usage() { return 0; }

tt::tt_metal::MemoryConfig ConvT2DSliceAttr::get_input_memory_config(
    IOShape output_slice_start, IOShape output_slice_end) {
    auto compute_grid_size = device->compute_with_storage_grid_size();
    auto [input_start, input_end] = get_input_slice(output_slice_start, output_slice_end);
    uint32_t input_slice_height = std::get<0>(input_end) - std::get<0>(input_start);
    uint32_t input_slice_width = std::get<1>(input_end) - std::get<1>(input_start);
    uint32_t output_slice_height = std::get<0>(output_slice_end) - std::get<0>(output_slice_start);
    uint32_t output_slice_width = std::get<1>(output_slice_end) - std::get<1>(output_slice_start);
    uint32_t width_rounding_value =
        (conv_config.output_layout == tt::tt_metal::Layout::TILE) ? tt::constants::TILE_HEIGHT : 1;

    if (output_slice_width % width_rounding_value != 0) {
        uint32_t additional_padded_width = width_rounding_value - (output_slice_width % width_rounding_value);
        output_slice_width += additional_padded_width;
    }

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
        BufferType::DRAM));
    return sliced_input_tensor_memory_config;
}

ttnn::Tensor ConvT2DSliceAttr::run_L1_op(
    const ttnn::Tensor& sliced_input_tensor, IOShape output_slice_start, IOShape output_slice_end) {
    int output_slice_height_start, output_slice_width_start;
    int output_slice_height_end, output_slice_width_end;
    std::tie(output_slice_height_start, output_slice_width_start) = output_slice_start;
    std::tie(output_slice_height_end, output_slice_width_end) = output_slice_end;
    int input_slice_height_start = div_up((output_slice_height_start - (int)padding_n4[0]), (int)stride[0]);
    int input_slice_width_start = div_up((output_slice_width_start - (int)padding_n4[2]), (int)stride[1]);
    int unpadded_output_height_start = std::max<int>(0, output_slice_height_start - (int)padding_n4[0]);
    int unpadded_output_width_start = std::max<int>(0, output_slice_width_start - (int)padding_n4[2]);
    int pad_top_offset = unpadded_output_height_start % (int)stride[0] == 0
                             ? 0
                             : stride[0] - (unpadded_output_height_start % (int)stride[0]);
    int pad_left_offset = unpadded_output_width_start % (int)stride[1] == 0
                              ? 0
                              : stride[1] - (unpadded_output_width_start % (int)stride[1]);

    int input_slice_height_end = div_up(
        (output_slice_height_end - (int)padding_n4[0] + ((int)kernel_size[0] - 1) * ((int)dilation[0] - 1) +
         (int)kernel_size[0] - 1),
        (int)stride[0]);
    int input_slice_width_end = div_up(
        (output_slice_width_end - (int)padding_n4[2] + ((int)kernel_size[1] - 1) * ((int)dilation[1] - 1) +
         (int)kernel_size[1] - 1),
        (int)stride[1]);

    int pad_top = std::max<int>({0, -input_slice_height_start, pad_top_offset});
    int pad_bottom = std::max<int>(0, input_slice_height_end - std::get<0>(input_shape));
    int pad_left = std::max<int>({0, -input_slice_width_start, pad_left_offset});
    int pad_right = std::max<int>(0, input_slice_width_end - std::get<1>(input_shape));

    input_slice_height_start = std::max<int>(0, input_slice_height_start);
    input_slice_height_end = std::min<int>(std::get<0>(input_shape), input_slice_height_end);
    input_slice_width_start = std::max<int>(0, input_slice_width_start);
    input_slice_width_end = std::min<int>(std::get<1>(input_shape), input_slice_width_end);

    auto [input_height, input_width] = input_shape;
    uint32_t output_height = ((input_height - 1) * stride[0]) - (padding_n4[0] + padding_n4[1]) +
                             (dilation[0] * (kernel_size[0] - 1)) + output_padding[0] + 1;

    uint32_t output_width = ((input_width - 1) * stride[1]) - (padding_n4[2] + padding_n4[3]) +
                            (dilation[1] * (kernel_size[1] - 1)) + output_padding[1] + 1;

    std::array<uint32_t, 2> this_output_pad = {0, 0};
    if (output_slice_height_start == 0) {
        pad_top = padding_n4[0];
        input_slice_height_start = 0;
    }
    if (output_slice_height_end == output_height) {
        pad_bottom = padding_n4[1];
        input_slice_height_end = std::get<0>(input_shape);
        this_output_pad[0] = output_padding[0];
    }
    if (output_slice_width_start == 0) {
        pad_left = padding_n4[2];
        input_slice_width_start = 0;
    }
    if (output_slice_width_end == output_width) {
        pad_right = padding_n4[3];
        input_slice_width_end = std::get<1>(input_shape);
        this_output_pad[1] = output_padding[1];
    }

    int input_slice_height = input_slice_height_end - input_slice_height_start;
    int input_slice_width = input_slice_width_end - input_slice_width_start;

    uint32_t output_slice_width = output_slice_width_end - output_slice_width_start;
    uint32_t width_rounding_value =
        (conv_config.output_layout == tt::tt_metal::Layout::TILE) ? tt::constants::TILE_HEIGHT : 1;

    if (output_slice_width % width_rounding_value != 0) {
        uint32_t additional_padded_width = width_rounding_value - (output_slice_width % width_rounding_value);
        log_info(
            LogOp,
            "Conv2d Transpose DRAM Slicing: Additional padding of {} added to the right side.",
            additional_padded_width);
        pad_right += additional_padded_width * stride[1];
        output_slice_width += additional_padded_width;
    }
    uint32_t base_pad_height = dilation[0] * (kernel_size[0] - 1);
    uint32_t base_pad_width = dilation[1] * (kernel_size[1] - 1);
    auto this_op_padding = std::array<uint32_t, 4>(
        {base_pad_height - pad_top,
         base_pad_height - pad_bottom,
         base_pad_width - pad_left,
         base_pad_width - pad_right});
    log_info(
        tt::LogOp,
        "Conv input {}, padding {}, dilation {}, kernel {}, stride {}, output slice {}x{}",
        sliced_input_tensor.logical_shape(),
        this_op_padding,
        dilation,
        kernel_size,
        stride,
        output_slice_height_end - output_slice_height_start,
        output_slice_width);

    auto conv_config_l1 = conv_config;

    conv_config_l1.deallocate_activation = true;
    conv_config_l1.reallocate_halo_output = true;

    // Force Conv2d_L1 to always output tiled layout to reduce CB Memory usage.
    conv_config_l1.output_layout = Layout::TILE;

    auto conv2d_result = conv_transpose2d_L1(
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
        this_output_pad,
        dilation,
        groups,
        output_dtype,
        bias_tensor,
        conv_config_l1,
        compute_config,
        std::nullopt,
        mirror_kernel);
    weight_tensor = std::get<3>(conv2d_result);
    if (bias_tensor.has_value()) {
        bias_tensor->get() = std::get<4>(conv2d_result).value();
    }
    return std::get<0>(conv2d_result);
}
std::string ConvT2DSliceAttr::name() { return "ConvTranspose2D"; }

ResultWithOptions ConvTranpose2dOperation::invoke(
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
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    const std::optional<const DataType>& dtype,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config_,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_,
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
        dtype,
        bias_tensor,
        conv_config_,
        compute_config_,
        memory_config_,
        dram_slice_config_,
        mirror_kernel,
        return_output_dim,
        return_weights_and_bias);
}

}  // namespace ttnn::operations::conv::conv_transpose2d
