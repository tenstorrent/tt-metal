// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include "tt-metalium/assert.hpp"
#include "tt-metalium/buffer.hpp"
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/mesh_device.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/conv/conv_transpose2d/prepare_conv_transpose2d_weights.hpp"
#include "ttnn/operations/core/core.hpp"
namespace ttnn {

namespace operations::conv {
namespace conv_transpose2d {

template <typename T>
Tensor _transform_weights_for_conv_transpose2d(const Tensor& conv_weight_tensor, bool mirror_kernel = true) {
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
        conv_weight_tensor.host_storage().transform(compute),
        output_spec,
        conv_weight_tensor.distributed_tensor_config(),
        conv_weight_tensor.tensor_topology());
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
        case DataType::FLOAT32:
            return _transform_weights_for_conv_transpose2d<float>(to_mirror_tensor, mirror_kernel);
        case DataType::UINT32:
            return _transform_weights_for_conv_transpose2d<uint32_t>(to_mirror_tensor, mirror_kernel);
        default: TT_THROW("Unsupported data type for transform_weights_for_conv_transpose2d", to_mirror_tensor.dtype());
    }
};

template <typename T>
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
    T* device,
    DataType input_dtype,
    const std::optional<const DataType>& output_dtype,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    bool mirror_kernel) {
    TT_ASSERT(
        weights_format == "IOHW",
        "PyTorch expects weights for ConvTranspose2D in IOHW format. If you have passed the correct weights, then make "
        "sure that the weights_format string is set to \"IOHW\".");
    Tensor mirrored_weight_tensor = transform_weights_for_conv_transpose2d(weight_tensor, mirror_kernel);
    return prepare_conv_weights(
        mirrored_weight_tensor,
        input_memory_config,
        input_layout,
        weights_format,
        in_channels,
        out_channels,
        batch_size,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        dilation,
        has_bias,
        groups,
        device,
        input_dtype,
        output_dtype,
        conv_config_,
        compute_config_,
        std::nullopt);
}

template <typename T>
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
    T* device,
    DataType input_dtype,
    const std::optional<const DataType>& output_dtype,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_) {
    return prepare_conv_bias(
        bias_tensor,
        input_memory_config,
        input_layout,
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
        device,
        input_dtype,
        output_dtype,
        conv_config_,
        compute_config_);
}

template ttnn::Tensor prepare_conv_transpose2d_weights(
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
    IDevice* device,
    DataType input_dtype,
    const std::optional<const DataType>& output_dtype,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    bool mirror_kernel);

template ttnn::Tensor prepare_conv_transpose2d_weights(
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
    tt::tt_metal::distributed::MeshDevice* device,
    DataType input_dtype,
    const std::optional<const DataType>& output_dtype,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    bool mirror_kernel);

template ttnn::Tensor prepare_conv_transpose2d_bias(
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
    IDevice* device,
    DataType input_dtype,
    const std::optional<const DataType>& output_dtype,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_);

template ttnn::Tensor prepare_conv_transpose2d_bias(
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
    tt::tt_metal::distributed::MeshDevice* device,
    DataType input_dtype,
    const std::optional<const DataType>& output_dtype,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_);

}  // namespace conv_transpose2d
}  // namespace operations::conv
}  // namespace ttnn
