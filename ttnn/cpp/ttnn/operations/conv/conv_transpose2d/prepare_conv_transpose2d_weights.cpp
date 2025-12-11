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

using namespace ttnn::operations::sliding_window;

namespace ttnn::operations::conv::conv_transpose2d {

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
    const ttnn::MemoryConfig& input_memory_config,
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
    const std::optional<const conv2d::Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const conv2d::Conv2dSliceConfig>& dram_slice_config_,
    bool mirror_kernel) {
    TT_ASSERT(
        weights_format == "IOHW",
        "PyTorch expects weights for ConvTranspose2D in IOHW format. If you have passed the correct weights, then make "
        "sure that the weights_format string is set to \"IOHW\".");

    // For transposed conv2d, the conv2d micro-op always uses stride=1x1 and operates on
    // "full_input" dimensions (after halo/padding expansion), not the original input dimensions.
    // Note: prepare_conv_transpose2d_weights is called from Python and doesn't receive output_padding,
    // so we assume output_padding = 0 for weight preparation (the actual conv op handles output_padding)
    auto dims =
        compute_conv_transpose2d_dimensions(input_height, input_width, kernel_size, stride, padding, {0, 0}, dilation);

    Tensor mirrored_weight_tensor = transform_weights_for_conv_transpose2d(weight_tensor, mirror_kernel);
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
        groups,
        device,
        input_dtype,
        output_dtype,
        conv_config_,
        compute_config_,
        dram_slice_config_);
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
    const std::optional<const conv2d::Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const conv2d::Conv2dSliceConfig>& dram_slice_config_) {
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
