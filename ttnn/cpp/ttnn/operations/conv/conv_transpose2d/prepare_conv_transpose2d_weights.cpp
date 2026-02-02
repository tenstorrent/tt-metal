// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/conv/conv_transpose2d/prepare_conv_transpose2d_weights.hpp"

#include <cstdint>
#include <optional>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/sliding_window/sliding_window.hpp>
#include <ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp>
#include "conv2d/conv2d_utils.hpp"
#include "conv2d/device/conv2d_device_operation_types.hpp"
#include "conv_transpose2d/conv_transpose2d.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace ttnn::operations::sliding_window;

namespace ttnn::operations::conv::conv_transpose2d {

using ttnn::operations::conv::conv2d::prepare_conv_bias;
using ttnn::operations::conv::conv2d::prepare_conv_weights;
using ttnn::prim::Conv2dConfig;
using ttnn::prim::Conv2dSliceConfig;

// Compute all transposed conv2d dimension transformations in one place
// This uses SlidingWindowConfig as the single source of truth for how transposed conv2d
// parameters are transformed into conv2d parameters
ConvTranspose2dDimensions compute_conv_transpose2d_dimensions(
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation) {
    // Create SlidingWindowConfig with transposed conv parameters
    SlidingWindowConfig config;
    config.batch_size = 1;  // Batch size not needed for dimension calculations
    config.input_hw = {input_height, input_width};
    config.window_hw = {kernel_size[0], kernel_size[1]};
    config.stride_hw = {stride[0], stride[1]};
    config.padding = get_pair_n4_padding(padding);
    config.output_pad_hw = {output_padding[0], output_padding[1]};
    config.dilation_hw = {dilation[0], dilation[1]};
    config.is_transpose = true;

    // Use SlidingWindowConfig methods to compute dimensions
    auto output_shape = config.get_output_shape();
    auto full_input_shape = config.get_transposed_full_input_shape();
    auto real_padding = config.get_transposed_real_padding();

    // Calculate strided dimensions (not exposed by SlidingWindowConfig, but simple formula)
    uint32_t strided_input_height = ((input_height - 1) * stride[0]) + 1;
    uint32_t strided_input_width = ((input_width - 1) * stride[1]) + 1;

    // Populate result struct
    ConvTranspose2dDimensions dims{};
    dims.output_height = output_shape[1];
    dims.output_width = output_shape[2];
    dims.full_input_height = full_input_shape[1];
    dims.full_input_width = full_input_shape[2];
    dims.strided_input_height = strided_input_height;
    dims.strided_input_width = strided_input_width;
    dims.input_pad_top = real_padding[0].first;
    dims.input_pad_bottom = real_padding[0].second;
    dims.input_pad_left = real_padding[1].first;
    dims.input_pad_right = real_padding[1].second;

    return dims;
}

template <typename T>
ttnn::Tensor _transform_weights_for_conv_transpose2d(const Tensor& conv_weight_tensor, bool mirror_kernel = true) {
    TT_FATAL(is_cpu_tensor(conv_weight_tensor), "transform_weights_for_conv_transpose2d only supports host tensors");

    // in_w_shape = {in_channels, out_channels, kernel_height, kernel_width}
    // out_w_shape = {out_channels, in_channels, kernel_height, kernel_width}
    // Flip kernel_height and kernel_width
    const auto& in_w_shape = conv_weight_tensor.padded_shape();
    const uint32_t in_channels = in_w_shape[0];
    const uint32_t out_channels = in_w_shape[1];
    const uint32_t kernel_height = in_w_shape[2];
    const uint32_t kernel_width = in_w_shape[3];
    const ttnn::Shape output_shape{out_channels, in_channels, kernel_height, kernel_width};
    auto compute = [&output_shape, in_channels, out_channels, kernel_height, kernel_width, mirror_kernel](
                       const tt::tt_metal::HostBuffer& input_host_buffer) {
        auto input_buffer = tt::tt_metal::host_buffer::get_as<T>(input_host_buffer);
        auto owned_buffer = std::vector<T>(output_shape.volume());

        for (uint32_t out_channels_index = 0; out_channels_index < out_channels; out_channels_index++) {
            uint32_t output_weight_out_channel_base_idx =
                out_channels_index * in_channels * kernel_height * kernel_width;
            uint32_t input_weight_out_channel_base_idx = out_channels_index * kernel_height * kernel_width;
            for (uint32_t in_channels_index = 0; in_channels_index < in_channels; in_channels_index++) {
                uint32_t output_weight_in_channel_base_idx = in_channels_index * kernel_height * kernel_width;
                uint32_t input_weight_in_channel_base_idx =
                    in_channels_index * kernel_height * kernel_width * out_channels;

                for (uint32_t in_kernel_height_index = 0; in_kernel_height_index < kernel_height;
                     in_kernel_height_index++) {
                    uint32_t out_buffer_kh_index =
                        mirror_kernel ? kernel_height - in_kernel_height_index - 1 : in_kernel_height_index;
                    uint32_t in_height_offset = in_kernel_height_index * kernel_width;
                    uint32_t out_height_offset = out_buffer_kh_index * kernel_width;
                    for (uint32_t in_kernel_width_index = 0; in_kernel_width_index < kernel_width;
                         in_kernel_width_index++) {
                        uint32_t out_buffer_kw_index =
                            mirror_kernel ? kernel_width - in_kernel_width_index - 1 : in_kernel_width_index;

                        uint32_t in_idx = input_weight_out_channel_base_idx + input_weight_in_channel_base_idx +
                                          in_height_offset + in_kernel_width_index;
                        uint32_t out_idx = output_weight_out_channel_base_idx + output_weight_in_channel_base_idx +
                                           out_height_offset + out_buffer_kw_index;

                        owned_buffer[out_idx] = input_buffer[in_idx];
                    }
                }
            }
        }
        return tt::tt_metal::HostBuffer(std::move(owned_buffer));
    };

    const TensorSpec output_spec(
        output_shape,
        tt::tt_metal::TensorLayout(
            conv_weight_tensor.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));

    return Tensor(
        conv_weight_tensor.host_storage().transform(compute), output_spec, conv_weight_tensor.tensor_topology());
}

Tensor transform_weights_for_conv_transpose2d(const Tensor& conv_weight_tensor, bool mirror_kernel) {
    Tensor to_mirror_tensor;
    if (tt::tt_metal::is_device_tensor(conv_weight_tensor)) {
        log_warning(
            tt::LogOp,
            "Prepare Weights for ConvTranspose2D needs weights on host, but they are already on device. The op will "
            "move them back to host.");
        to_mirror_tensor = ttnn::operations::core::from_device(conv_weight_tensor);
    } else {
        to_mirror_tensor = conv_weight_tensor;
    }
    switch (conv_weight_tensor.dtype()) {
        case DataType::BFLOAT16:
            return _transform_weights_for_conv_transpose2d<::bfloat16>(to_mirror_tensor, mirror_kernel);
        case DataType::FLOAT32: return _transform_weights_for_conv_transpose2d<float>(to_mirror_tensor, mirror_kernel);
        case DataType::UINT32:
            return _transform_weights_for_conv_transpose2d<uint32_t>(to_mirror_tensor, mirror_kernel);
        default: TT_THROW("Unsupported data type for transform_weights_for_conv_transpose2d", to_mirror_tensor.dtype());
    }
};

ttnn::Tensor prepare_conv_transpose2d_weights(
    const ttnn::Tensor& weight_tensor,
    ttnn::MemoryConfig input_memory_config,
    Layout input_layout,
    const std::string& weights_format,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    const bool has_bias,
    uint32_t groups,
    MeshDevice* device,
    DataType input_dtype,
    const std::optional<const DataType>& output_dtype,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_,
    bool mirror_kernel) {
    auto padding_n4 = sliding_window::get_pair_n4_padding(padding);
    DataType conv_output_dtype = output_dtype.value_or(input_dtype);
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    // Use weights_dtype from config if set, otherwise use weight tensor's dtype
    DataType weight_dtype = conv_config.weights_dtype.value_or(weight_tensor.dtype());
    DeviceComputeKernelConfig compute_config =
        compute_config_.value_or(get_conv_default_compute_kernel_config(device, input_dtype, weight_dtype));
    TT_ASSERT(
        weights_format == "IOHW",
        "PyTorch expects weights for ConvTranspose2D in IOHW format. If you have passed the correct weights, then make "
        "sure that the weights_format string is set to \"IOHW\".");

    // For grouped conv_transpose2d (groups > 1), we need to:
    // 1. Apply grouped layout conversion BEFORE the transpose to expand the weight tensor
    // 2. Then apply the standard transpose transformation
    // 3. Use groups=1 for the rest of the pipeline since grouping is already handled
    Tensor weight_for_transform = weight_tensor;
    uint32_t groups_for_prep = groups;
    if (groups > 1) {
        // Convert [in_channels, out_channels/groups, H, W] -> [in_channels, out_channels, H, W]
        weight_for_transform = conv2d::convert_conv_weight_tensor_to_grouped_layout_for_conv_transpose2d(
            weight_tensor, groups, weight_tensor.dtype());
        // After grouped conversion, we use groups=1 since the grouping is already embedded in the weights
        groups_for_prep = 1;
    }

    // Determine execution path based on configuration and input properties
    ConvT2dExecutionPath path = determine_conv_transpose2d_execution_path(
        tt::tt_metal::StorageType::DEVICE, input_memory_config, dram_slice_config_);
    Tensor mirrored_weight_tensor = transform_weights_for_conv_transpose2d(weight_for_transform, mirror_kernel);
    if (path == ConvT2dExecutionPath::L1) {
        // For transposed conv2d, the conv2d micro-op always uses stride=1x1 and operates on
        // "full_input" dimensions (after halo/padding expansion), not the original input dimensions.
        // Note: prepare_conv_transpose2d_weights is called from Python and doesn't receive output_padding,
        // so we assume output_padding = 0 for weight preparation (the actual conv op handles output_padding)
        auto dims = compute_conv_transpose2d_dimensions(
            input_height, input_width, kernel_size, stride, padding, {0, 0}, dilation);

        return prepare_conv_weights(
            mirrored_weight_tensor,
            input_memory_config,
            input_layout,
            weights_format,
            in_channels,
            out_channels,
            batch_size,
            dims.full_input_height,  // Use full_input dimensions, not original
            dims.full_input_width,   // Use full_input dimensions, not original
            kernel_size,
            ConvTranspose2dDimensions::CONV2D_STRIDE,   // stride is always 1x1 for conv2d micro-op
            ConvTranspose2dDimensions::CONV2D_PADDING,  // padding is 0 (halo already added padding)
            dilation,
            has_bias,
            groups_for_prep,  // Use 1 if groups > 1 since grouped conversion is already done
            device,
            input_dtype,
            output_dtype,
            conv_config_,
            compute_config_,
            op_slicing::Op2DSliceConfig{.slice_type = op_slicing::Op2DSliceConfig::SliceType::L1_FULL});
    }
    Tensor dummy_weight_tensor = tt::tt_metal::create_device_tensor(
        tt::tt_metal::TensorSpec(
            ttnn::Shape({in_channels, out_channels / groups, kernel_size[0], kernel_size[1]}),
            tt::tt_metal::TensorLayout(
                weight_dtype,
                tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
                MemoryConfig{
                    TensorMemoryLayout::INTERLEAVED,
                    BufferType::DRAM,
                })),
        device);
    std::optional<Tensor> dummy_bias_tensor = std::nullopt;
    if (has_bias) {
        dummy_bias_tensor = tt::tt_metal::create_device_tensor(
            tt::tt_metal::TensorSpec(
                ttnn::Shape({1, 1, 1, out_channels}),
                tt::tt_metal::TensorLayout(
                    weight_dtype,
                    tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
                    MemoryConfig{
                        TensorMemoryLayout::INTERLEAVED,
                        BufferType::DRAM,
                    })),
            device);
    }
    auto [output_height, output_width] = calculate_ct2d_output_image_size(
        {input_height, input_width}, kernel_size, stride, padding_n4, {0, 0}, dilation);
    auto convt2d_slice_attr = get_conv_transpose2d_slice_attr(
        batch_size,
        input_height,
        input_width,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding_n4,
        {0, 0},  // output_padding assumed to be 0 for weight preparation
        dilation,
        groups,
        input_layout,
        input_dtype,
        conv_output_dtype,
        std::ref(dummy_weight_tensor),
        has_bias ? std::make_optional(std::ref(dummy_bias_tensor.value())) : std::nullopt,
        conv_config,
        compute_config,
        device,
        mirror_kernel);
    auto dram_slice_config = op_slicing::determine_slice_config(
        convt2d_slice_attr.get(),
        ttnn::Shape{batch_size, input_height, input_width, in_channels},
        ttnn::Shape{batch_size, output_height, output_width, out_channels},
        dram_slice_config_,
        conv_config.output_layout,
        device);
    log_info(
        tt::LogOp,
        "Auto determined DRAM Slice Config in Prepare Conv_Transpose2d Weights as {} for {}",
        dram_slice_config,
        convt2d_slice_attr->name());

    uint32_t slice_rounding_value = 1;
    if (conv_config.output_layout == tt::tt_metal::Layout::TILE &&
        dram_slice_config.slice_type == Conv2dSliceConfig::SliceType::DRAM_WIDTH) {
        // In Conv2d DRAM with Outputs in Tile layout, we need to round the slice size to a multiple of TILE_HEIGHT.
        slice_rounding_value = tt::constants::TILE_HEIGHT;
    }

    const uint32_t output_sliced_dim =
        dram_slice_config.slice_type == Conv2dSliceConfig::SliceType::DRAM_HEIGHT ? output_height : output_width;

    TT_FATAL(
        dram_slice_config.num_slices <= output_sliced_dim,
        " Number of slices {} should be less or equal than the dimension being sliced {} in Conv2D DRAM Slicing",
        dram_slice_config.num_slices,
        output_sliced_dim);

    const uint32_t min_output_slice_size =
        tt::div_up(tt::div_up(output_sliced_dim, slice_rounding_value), dram_slice_config.num_slices) *
        slice_rounding_value;
    if (dram_slice_config.slice_type == Conv2dSliceConfig::SliceType::DRAM_HEIGHT) {
        output_height = min_output_slice_size;
    } else {
        output_width = min_output_slice_size;
    }
    auto [input_slice_start, input_slice_end] =
        convt2d_slice_attr->get_input_slice({0, 0}, {output_height, output_width});
    input_memory_config = convt2d_slice_attr->get_input_memory_config(
        {0, 0},                        // Slice Start
        {output_height, output_width}  // Slice End
    );
    auto [input_height_slice_start, input_width_slice_start] = input_slice_start;
    auto [input_height_slice_end, input_width_slice_end] = input_slice_end;
    auto input_height_sliced = input_height_slice_end - input_height_slice_start;
    auto input_width_sliced = input_width_slice_end - input_width_slice_start;
    auto dims = compute_conv_transpose2d_dimensions(
        input_height_sliced, input_width_sliced, kernel_size, stride, padding, {0, 0}, dilation);

    return prepare_conv_weights(
        mirrored_weight_tensor,
        input_memory_config,
        dram_slice_config.num_slices > 1 ? Layout::ROW_MAJOR : input_layout,
        weights_format,
        in_channels,
        out_channels,
        batch_size,
        dims.full_input_height,  // Use full_input dimensions, not original
        dims.full_input_width,   // Use full_input dimensions, not original
        kernel_size,
        ConvTranspose2dDimensions::CONV2D_STRIDE,   // stride is always 1x1 for conv2d micro-op
        ConvTranspose2dDimensions::CONV2D_PADDING,  // padding is 0 (halo already added padding)
        dilation,
        has_bias,
        groups_for_prep,  // Use 1 if groups > 1 since grouped conversion is already done
        device,
        input_dtype,
        output_dtype,
        conv_config_,
        compute_config_,
        op_slicing::Op2DSliceConfig{.slice_type = op_slicing::Op2DSliceConfig::SliceType::L1_FULL});
}

ttnn::Tensor prepare_conv_transpose2d_bias(
    const ttnn::Tensor& bias_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_layout,
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
    MeshDevice* device,
    DataType input_dtype,
    const std::optional<const DataType>& output_dtype,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_) {
    // For transposed conv2d, the conv2d micro-op always uses stride=1x1 and operates on
    // full_input dimensions. Calculate these dimensions for bias preparation.
    // Note: bias preparation doesn't receive output_padding, so we assume output_padding = 0
    auto dims =
        compute_conv_transpose2d_dimensions(input_height, input_width, kernel_size, stride, padding, {0, 0}, dilation);

    return prepare_conv_bias(
        bias_tensor,
        input_memory_config,
        input_layout,
        in_channels,
        out_channels,
        batch_size,
        dims.full_input_height,  // Use full_input dimensions
        dims.full_input_width,   // Use full_input dimensions
        kernel_size,
        ConvTranspose2dDimensions::CONV2D_STRIDE,   // stride is always 1x1 for conv2d micro-op
        ConvTranspose2dDimensions::CONV2D_PADDING,  // padding is 0 (halo already added padding)
        dilation,
        groups,
        device,
        input_dtype,
        output_dtype,
        conv_config_,
        compute_config_,
        dram_slice_config_);
}

}  // namespace ttnn::operations::conv::conv_transpose2d
