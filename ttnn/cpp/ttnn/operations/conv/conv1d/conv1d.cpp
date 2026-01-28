// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/conv/conv1d/conv1d.hpp"

#include <array>
#include <variant>

#include "ttnn/operations/conv/conv_types.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::conv::conv1d {

using ttnn::prim::Conv2dSliceConfig;

Result Conv1dOperation::invoke(
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

    auto [output_tensor, output_dimensions, weights_and_bias] =
        std::get<static_cast<int>(ResultType::OUTPUT_DIM_WEIGHTS_AND_BIAS)>(ttnn::conv2d(
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
            Conv2dSliceConfig{
                .slice_type = Conv2dSliceConfig::SliceType::L1_FULL},  // Conv1D doesn't support DRAM Slicing. Only L1
            true,
            true));

    if (return_output_dim && return_weights_and_bias) {
        return Result(std::tuple(output_tensor, std::get<1>(output_dimensions), weights_and_bias));
    }
    if (return_output_dim) {
        return Result(std::tuple(output_tensor, std::get<1>(output_dimensions)));
    }
    if (return_weights_and_bias) {
        return Result(std::tuple(output_tensor, weights_and_bias));
    }
    return Result(output_tensor);
}

}  // namespace ttnn::operations::conv::conv1d
