// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_transpose2d_pybind.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "conv_transpose2d.hpp"
#include "prepare_conv_transpose2d_weights.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::conv::conv_transpose2d {

void py_bind_conv_transpose2d(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::conv_transpose2d,
        R"doc(
        Applies a 2D transposed convolution operator over an input image composed of several input planes.

        This module can be seen as the gradient of Conv2d with respect to its input. It is also known as a
        fractionally-strided convolution or a deconvolution

        The input tensor is expected in the following format (N x H x W x C) **differs from PyTorch** where:

        - N is the batch size
        - H is the height of the input
        - W is the width of the input
        - C is the number of channels in the input

        The weight tensor is expected in the following format (C x O / G x K_H x K_W).
        The bias tensor is optional and expected in the following format (O / G ).
        Where:

        - C is the number of input channels
        - O is the number of output channels
        - G is the number of groups
        - K_H is the height of the kernel
        - K_W is the width of the kernel

        The shape of the output tensor is given by the following equation :

        - H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
        - W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1

        :param ttnn.Tensor input_tensor:  the input tensor.
        :param ttnn.Tensor weight_tensor: the weight tensor.
        :param ttnn.Tensor, None bias_tensor:   optional bias tensor. Default: None
        :param ttnn.IDevice device:  the device to use.
        :param int in_channels:  number of input channels.
        :param int out_channels:  number of output channels.
        :param int batch_size:  batch size.
        :param int input_height:  height of the input tensor.
        :param int input_width:  width of the input tensor.
        :param tuple[int  , int] kernel_size: size of the convolving kernel.
        :param tuple[int, int] stride: stride of the cross-correlation.
        :param tuple[int, int] or tuple[int, int, int, int]) padding: zero-padding added to both sides of the input. [pad_height, pad_width] or [pad_top, pad_bottom, pad_left, pad_right].
        :param tuple[int, int] dilation: spacing between kernel elements.
        :param int groups:  number of blocked connections from input channels to output channels.
        :param DataType, None dtype:  the data type of the output tensor. Default: None (will use the same dtype as input_tensor).
        :param ttnn.Conv2dConfig, None conv_config: configuration for convolution. Default: None
        :param ttnn.DeviceComputeKernelConfig, None compute_config: configuration for compute kernel. Default: None
        :param bool mirror_kernel: Determines if the op should mirror the kernel internally. Should be set to True if the kernel has already been mirrored.
        :param int queue_id: the queue id to use for the operation. Default: `0`.
        :param bool return_output_dim:  If true, the op also returns the height and width of the output tensor in [N, H, W, C] format,
        :param bool return_weights_and_bias:  If true, the op also returns the preprocessed weight and bias on device .

        :return: The output tensor, output height and width, and the preprocessed weights and bias.

        :rtype: [ttnn.Tensor]: the output tensor, when return_output_dim = False and return_weights_and_bias = False
        :rtype: [ttnn.Tensor, Tuple[int, int]]: the output tensor, and it's height and width, if return_output_dim = True
        :rtype: [ttnn.Tensor, Tuple[ttnn.Tensor, ttnn.Tensor]]: the output tensor, and it's height and width, if return_weights_and_bias = True
        :rtype: [ttnn.Tensor, Tuple[int, int], Tuple[ttnn.Tensor, ttnn.Tensor]]: the output tensor, and it's height and width, if return_output_dim = True and return_weights_and_bias = True
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::conv_transpose2d)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               ttnn::IDevice* device,
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
               const std::optional<const DataType>& dtype,
               std::optional<const ttnn::Tensor> bias_tensor,
               const std::optional<const Conv2dConfig>& conv_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_config,
               const std::optional<const MemoryConfig>& memory_config,
               bool mirror_kernel,
               const bool return_output_dim,
               const bool return_weights_and_bias,
               QueueId queue_id) -> Result {
                return self(
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
                    output_padding,
                    dilation,
                    groups,
                    dtype,
                    bias_tensor,
                    conv_config,
                    compute_config,
                    memory_config,
                    mirror_kernel,
                    return_output_dim,
                    return_weights_and_bias);
            },
            py::kw_only(),
            py::arg("input_tensor"),
            py::arg("weight_tensor"),
            py::arg("device"),
            py::arg("in_channels"),
            py::arg("out_channels"),
            py::arg("batch_size"),
            py::arg("input_height"),
            py::arg("input_width"),
            py::arg("kernel_size"),
            py::arg("stride") = std::array<uint32_t, 2>{1, 1},
            py::arg("padding") = std::array<uint32_t, 2>{0, 0},
            py::arg("output_padding") = std::array<uint32_t, 2>{0, 0},
            py::arg("dilation") = std::array<uint32_t, 2>{1, 1},
            py::arg("groups") = 1,
            py::arg("dtype") = std::nullopt,
            py::arg("bias_tensor") = std::nullopt,
            py::arg("conv_config") = std::nullopt,
            py::arg("compute_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("mirror_kernel") = true,
            py::arg("return_output_dim") = false,
            py::arg("return_weights_and_bias") = false,
            py::arg("queue_id") = DefaultQueueId},

        ttnn::pybind_overload_t{
            [](const decltype(ttnn::conv_transpose2d)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               ttnn::MeshDevice* device,
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
               const std::optional<const DataType>& dtype,
               std::optional<const ttnn::Tensor> bias_tensor,
               const std::optional<const Conv2dConfig>& conv_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_config,
               const std::optional<const MemoryConfig>& memory_config,
               bool mirror_kernel,
               const bool return_output_dim,
               const bool return_weights_and_bias,
               QueueId queue_id) -> Result {
                return self(
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
                    output_padding,
                    dilation,
                    groups,
                    dtype,
                    bias_tensor,
                    conv_config,
                    compute_config,
                    memory_config,
                    mirror_kernel,
                    return_output_dim,
                    return_weights_and_bias);
            },
            py::kw_only(),
            py::arg("input_tensor"),
            py::arg("weight_tensor"),
            py::arg("device"),
            py::arg("in_channels"),
            py::arg("out_channels"),
            py::arg("batch_size"),
            py::arg("input_height"),
            py::arg("input_width"),
            py::arg("kernel_size"),
            py::arg("stride") = std::array<uint32_t, 2>{1, 1},
            py::arg("padding") = std::array<uint32_t, 2>{0, 0},
            py::arg("output_padding") = std::array<uint32_t, 2>{0, 0},
            py::arg("dilation") = std::array<uint32_t, 2>{1, 1},
            py::arg("groups") = 1,
            py::arg("dtype") = std::nullopt,
            py::arg("bias_tensor") = std::nullopt,
            py::arg("conv_config") = std::nullopt,
            py::arg("compute_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("mirror_kernel") = true,
            py::arg("return_output_dim") = false,
            py::arg("return_weights_and_bias") = false,
            py::arg("queue_id") = DefaultQueueId});

    module.def(
        "prepare_conv_transpose2d_weights",
        prepare_conv_transpose2d_weights<ttnn::IDevice>,
        py::kw_only(),
        py::arg("weight_tensor"),
        py::arg("input_memory_config"),
        py::arg("input_layout"),
        py::arg("weights_format"),
        py::arg("in_channels"),
        py::arg("out_channels"),
        py::arg("batch_size"),
        py::arg("input_height"),
        py::arg("input_width"),
        py::arg("kernel_size"),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("has_bias"),
        py::arg("groups"),
        py::arg("device"),
        py::arg("input_dtype"),
        py::arg("output_dtype") = std::nullopt,
        py::arg("conv_config") = std::nullopt,
        py::arg("compute_config") = std::nullopt,
        py::arg("mirror_kernel") = true);

    module.def(
        "prepare_conv_transpose2d_weights",
        prepare_conv_transpose2d_weights<ttnn::MeshDevice>,
        py::kw_only(),
        py::arg("weight_tensor"),
        py::arg("input_memory_config"),
        py::arg("input_layout"),
        py::arg("weights_format"),
        py::arg("in_channels"),
        py::arg("out_channels"),
        py::arg("batch_size"),
        py::arg("input_height"),
        py::arg("input_width"),
        py::arg("kernel_size"),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("has_bias"),
        py::arg("groups"),
        py::arg("device"),
        py::arg("input_dtype"),
        py::arg("output_dtype") = std::nullopt,
        py::arg("conv_config") = std::nullopt,
        py::arg("compute_config") = std::nullopt,
        py::arg("mirror_kernel") = true);

    module.def(
        "prepare_conv_transpose2d_bias",
        prepare_conv_transpose2d_bias<ttnn::IDevice>,
        py::kw_only(),
        py::arg("bias_tensor"),
        py::arg("input_memory_config"),
        py::arg("input_layout"),
        py::arg("in_channels"),
        py::arg("out_channels"),
        py::arg("batch_size"),
        py::arg("input_height"),
        py::arg("input_width"),
        py::arg("kernel_size"),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"),
        py::arg("device"),
        py::arg("input_dtype"),
        py::arg("output_dtype") = std::nullopt,
        py::arg("conv_config") = std::nullopt,
        py::arg("compute_config") = std::nullopt);

    module.def(
        "prepare_conv_transpose2d_bias",
        prepare_conv_transpose2d_bias<ttnn::MeshDevice>,
        py::kw_only(),
        py::arg("bias_tensor"),
        py::arg("input_memory_config"),
        py::arg("input_layout"),
        py::arg("in_channels"),
        py::arg("out_channels"),
        py::arg("batch_size"),
        py::arg("input_height"),
        py::arg("input_width"),
        py::arg("kernel_size"),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"),
        py::arg("device"),
        py::arg("input_dtype"),
        py::arg("output_dtype") = std::nullopt,
        py::arg("conv_config") = std::nullopt,
        py::arg("compute_config") = std::nullopt);
}

}  // namespace ttnn::operations::conv::conv_transpose2d
