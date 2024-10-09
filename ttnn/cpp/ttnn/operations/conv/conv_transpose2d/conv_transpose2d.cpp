// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_transpose2d.hpp"
#include <sys/types.h>
#include <cstdint>


using namespace tt;
namespace ttnn {
namespace operations::conv {
using sliding_window::SlidingWindowConfig;
using sliding_window::ParallelConfig;

namespace conv_transpose2d {


template<typename T>
std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> conv_transpose2d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    T * device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    std::optional<const conv2d::Conv2dConfig> conv_config_)
    {
        //Inverse of sliding_window.get_output_shape()
        SlidingWindowConfig sliding_window_config = SlidingWindowConfig{
            .batch_size = batch_size,
            .input_hw = {input_height, input_width},
            .window_hw = kernel_size,
            .stride_hw = stride,
            .pad_hw = padding,
            .output_pad_hw = output_padding,
            .dilation_hw = {dilation[0], dilation[1]},
            .is_transpose = true
        };

        uint32_t output_height = (input_height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1;
        uint32_t output_width  = (input_width  - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1;

        //Dimensions of Input to Conv u_op
        uint32_t final_input_height = output_height + dilation[0] * (kernel_size[0] - 1);
        uint32_t final_input_width  =  output_width + dilation[1] * (kernel_size[1] - 1);

        //Size of input after adding interleaved 0s.
        uint32_t strided_input_height = (input_height - 1) * stride[0] + 1;
        uint32_t strided_input_width  = (input_width -  1) * stride[1] + 1;

        uint32_t input_pad_top  = (final_input_height - strided_input_height)/2;
        uint32_t input_pad_bottom = final_input_height - strided_input_height - input_pad_top;

        uint32_t input_pad_left = (final_input_width - strided_input_width)/2;
        uint32_t input_pad_right = final_input_width - strided_input_width - input_pad_left;

        log_debug(LogOp, "Input : {}x{}", input_height, input_width);
        log_debug(LogOp, "Output : {}x{}", output_height, output_width);

        log_debug(LogOp, "Conv Op Input : {}x{}", final_input_height, final_input_width);
        log_debug(LogOp, "Strided Input : {}x{}", strided_input_height, strided_input_width);

        log_debug(LogOp, "Padding : ({},{}) ({},{})", input_pad_top, input_pad_bottom, input_pad_left, input_pad_right);


        //Flip the Weights

        //Call Halo Transpose

        //Call Conv2d u_op with Stride = 1, Padding = 0.
        return {input_tensor, output_height, output_width, weight_tensor, bias_tensor};


    }

template std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> conv_transpose2d<Device>(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    Device * device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    std::optional<const conv2d::Conv2dConfig> conv_config_);

template std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> conv_transpose2d<MeshDevice>(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    MeshDevice * device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    std::optional<const conv2d::Conv2dConfig> conv_config_);

}
}
}
