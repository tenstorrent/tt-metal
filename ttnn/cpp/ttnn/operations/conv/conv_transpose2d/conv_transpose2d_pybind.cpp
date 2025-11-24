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

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            weight_tensor (ttnn.Tensor): the weight tensor.
            device (ttnn.MeshDevice): the device to use.
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            batch_size (int): batch size.
            input_height (int): height of the input tensor.
            input_width (int): width of the input tensor.
            kernel_size (tuple[int, int]): size of the convolving kernel.
            stride (tuple[int, int]): stride of the cross-correlation.
            padding (tuple[int, int] or tuple[int, int, int, int]): zero-padding added to both sides of the input. [pad_height, pad_width] or [pad_top, pad_bottom, pad_left, pad_right].
            dilation (tuple[int, int]): spacing between kernel elements.
            groups (int): number of blocked connections from input channels to output channels.

        Keyword Args:
            bias_tensor (ttnn.Tensor, optional): optional bias tensor. Default: None
            dtype (DataType, optional): the data type of the output tensor. Default: None (will use the same dtype as input_tensor).
            conv_config (ttnn.Conv2dConfig, optional): configuration for convolution. Default: None
            compute_config (ttnn.DeviceComputeKernelConfig, optional): configuration for compute kernel. Default: None
            mirror_kernel (bool, optional): Determines if the op should mirror the kernel internally. Should be set to True if the kernel has already been mirrored. Default: False
            return_output_dim (bool, optional): If true, the op also returns the height and width of the output tensor in [N, H, W, C] format. Default: False
            return_weights_and_bias (bool, optional): If true, the op also returns the preprocessed weight and bias on device. Default: False

        Returns:
            The output tensor, output height and width, and the preprocessed weights and bias.

            - ttnn.Tensor: the output tensor, when return_output_dim = False and return_weights_and_bias = False
            - tuple[ttnn.Tensor, tuple[int, int]]: the output tensor, and its height and width, if return_output_dim = True
            - tuple[ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor]]: the output tensor, and its weights and biases, if return_weights_and_bias = True
            - tuple[ttnn.Tensor, tuple[int, int], tuple[ttnn.Tensor, ttnn.Tensor]]: the output tensor, and its height and width, and its weights and biases, if return_output_dim = True and return_weights_and_bias = True
        )doc",

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
               const bool return_weights_and_bias) -> Result {
                return self(
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
            py::arg("return_weights_and_bias") = false});

    module.def(
        "prepare_conv_transpose2d_weights",
        prepare_conv_transpose2d_weights,
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
        prepare_conv_transpose2d_bias,
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
