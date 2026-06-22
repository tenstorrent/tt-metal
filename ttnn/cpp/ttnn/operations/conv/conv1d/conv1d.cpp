// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/conv/conv1d/conv1d.hpp"

#include <array>
#include <variant>

#include <tt_stl/assert.hpp>
#include "ttnn/operations/conv/conv_types.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {

Conv1dResult conv1d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    MeshDevice* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_length,
    uint32_t kernel_size,
    uint32_t stride,
    std::variant<std::array<uint32_t, 2>, uint32_t> padding,
    uint32_t dilation,
    uint32_t groups,
    const std::optional<const DataType>& dtype,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv1dConfig>& conv_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_config,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const Conv1dSliceConfig>& slice_config,
    bool return_output_dim,
    bool return_weights_and_bias) {
    // reshape input tensor to 4D, if it is not already
    const ttnn::Tensor& input_tensor_4d =
        (input_tensor.logical_shape().rank() < 4)
            ? ttnn::reshape(input_tensor, Shape({batch_size, 1, input_length, in_channels}))
            : input_tensor;

    // padding for conv2d based on conv1d padding
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> conv2d_padding;
    if (std::holds_alternative<uint32_t>(padding)) {
        conv2d_padding = std::array<uint32_t, 2>{0, std::get<uint32_t>(padding)};
    } else {
        std::array<uint32_t, 2> padding_lr = std::get<std::array<uint32_t, 2>>(padding);

        conv2d_padding = std::array<uint32_t, 4>{
            0,              // up
            0,              // down
            padding_lr[0],  // left
            padding_lr[1]   // right
        };
    };

    // Conv1d reshapes the input to [N, 1, input_length, C], so the height dimension is always 1.
    // DRAM slicing is therefore only meaningful along the width dimension (DRAM_WIDTH, i.e. input_length);
    // DRAM_HEIGHT would produce a single degenerate slice and silently collapse back to L1.
    // When no slice config is provided, preserve the historical conv1d behaviour of forcing L1_FULL.
    // Forwarding nullopt instead would route by input location (conv2d's default), sending DRAM/host
    // inputs through the DRAM slicing path - which breaks existing auto_shard / shard_layout=None
    // callers (the slicing path requires a shard layout). So conv1d opts in to DRAM slicing explicitly.
    const ttnn::prim::Conv2dSliceConfig effective_slice_config = slice_config.value_or(
        ttnn::prim::Conv2dSliceConfig{.slice_type = ttnn::prim::Conv2dSliceConfig::SliceType::L1_FULL});
    TT_FATAL(
        effective_slice_config.slice_type != ttnn::prim::Conv2dSliceConfig::SliceType::DRAM_HEIGHT,
        "Conv1D does not support DRAM_HEIGHT slicing because the convolution height is always 1. "
        "Use DRAM_WIDTH slicing (slices along input_length) or L1_FULL.");

    auto [output_tensor, output_dimensions, weights_and_bias] =
        std::get<static_cast<int>(ConvResultType::OUTPUT_DIM_WEIGHTS_AND_BIAS)>(ttnn::conv2d(
            input_tensor_4d,
            weight_tensor,
            device,
            in_channels,
            out_channels,
            batch_size,
            1,             // input_height
            input_length,  // input_width
            std::array<uint32_t, 2>{1, kernel_size},
            std::array<uint32_t, 2>{1, stride},
            conv2d_padding,
            std::array<uint32_t, 2>{1, dilation},
            groups,
            dtype,
            bias_tensor,
            conv_config,
            compute_config,
            memory_config,
            effective_slice_config,
            true,
            true));

    if (return_output_dim && return_weights_and_bias) {
        return Conv1dResult(std::tuple(output_tensor, std::get<1>(output_dimensions), weights_and_bias));
    }
    if (return_output_dim) {
        return Conv1dResult(std::tuple(output_tensor, std::get<1>(output_dimensions)));
    }
    if (return_weights_and_bias) {
        return Conv1dResult(std::tuple(output_tensor, weights_and_bias));
    }
    return Conv1dResult(output_tensor);
}

}  // namespace ttnn
