// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_transpose2d_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "conv_transpose2d.hpp"
#include "prepare_conv_transpose2d_weights.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::conv::conv_transpose2d {

using ttnn::prim::Conv2dConfig;
using ttnn::prim::Conv2dSliceConfig;

void bind_conv_transpose2d(nb::module_& mod) {
    bind_registered_operation(
        mod,
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
            dram_slice_config (ttnn.Conv2dSliceConfig, optional): Configuration for DRAM slicing. If provided, the operation will use DRAM slicing, else will fall back to L1 version. Default: None
            return_output_dim (bool, optional): If true, the op also returns the height and width of the output tensor in [N, H, W, C] format. Default: False
            return_weights_and_bias (bool, optional): If true, the op also returns the preprocessed weight and bias on device. Default: False

        Returns:
            The output tensor, output height and width, and the preprocessed weights and bias.

            - ttnn.Tensor: the output tensor, when return_output_dim = False and return_weights_and_bias = False
            - tuple[ttnn.Tensor, tuple[int, int]]: the output tensor, and its height and width, if return_output_dim = True
            - tuple[ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor]]: the output tensor, and its weights and biases, if return_weights_and_bias = True
            - tuple[ttnn.Tensor, tuple[int, int], tuple[ttnn.Tensor, ttnn.Tensor]]: the output tensor, and its height and width, and its weights and biases, if return_output_dim = True and return_weights_and_bias = True
        )doc",

        ttnn::nanobind_overload_t{
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
               std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
               std::array<uint32_t, 2> output_padding,
               std::array<uint32_t, 2> dilation,
               uint32_t groups,
               const std::optional<const DataType>& dtype,
               std::optional<const ttnn::Tensor> bias_tensor,
               const std::optional<const Conv2dConfig>& conv_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_config,
               const std::optional<const MemoryConfig>& memory_config,
               const std::optional<const Conv2dSliceConfig>& dram_slice_config,
               bool mirror_kernel,
               const bool return_output_dim,
               const bool return_weights_and_bias) -> ResultWithOptions {
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
                    dram_slice_config,
                    mirror_kernel,
                    return_output_dim,
                    return_weights_and_bias);
            },
            nb::kw_only(),
            nb::arg("input_tensor"),
            nb::arg("weight_tensor"),
            nb::arg("device"),
            nb::arg("in_channels"),
            nb::arg("out_channels"),
            nb::arg("batch_size"),
            nb::arg("input_height"),
            nb::arg("input_width"),
            nb::arg("kernel_size"),
            nb::arg("stride") = nb::cast(std::array<uint32_t, 2>{1, 1}),
            nb::arg("padding") = nb::cast(std::array<uint32_t, 2>{0, 0}),
            nb::arg("output_padding") = nb::cast(std::array<uint32_t, 2>{0, 0}),
            nb::arg("dilation") = nb::cast(std::array<uint32_t, 2>{1, 1}),
            nb::arg("groups") = 1,
            nb::arg("dtype") = nb::none(),
            nb::arg("bias_tensor") = nb::none(),
            nb::arg("conv_config") = nb::none(),
            nb::arg("compute_config") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("dram_slice_config") = nb::none(),
            nb::arg("mirror_kernel") = true,
            nb::arg("return_output_dim") = false,
            nb::arg("return_weights_and_bias") = false});

    mod.def(
        "prepare_conv_transpose2d_weights",
        prepare_conv_transpose2d_weights,
        nb::kw_only(),
        nb::arg("weight_tensor"),
        nb::arg("input_memory_config"),
        nb::arg("input_layout"),
        nb::arg("weights_format"),
        nb::arg("in_channels"),
        nb::arg("out_channels"),
        nb::arg("batch_size"),
        nb::arg("input_height"),
        nb::arg("input_width"),
        nb::arg("kernel_size"),
        nb::arg("stride"),
        nb::arg("padding"),
        nb::arg("dilation"),
        nb::arg("has_bias"),
        nb::arg("groups"),
        nb::arg("device"),
        nb::arg("input_dtype"),
        nb::arg("output_dtype") = nb::none(),
        nb::arg("conv_config") = nb::none(),
        nb::arg("compute_config") = nb::none(),
        nb::arg("dram_slice_config") = nb::none(),
        nb::arg("mirror_kernel") = true);

    mod.def(
        "prepare_conv_transpose2d_bias",
        prepare_conv_transpose2d_bias,
        nb::kw_only(),
        nb::arg("bias_tensor"),
        nb::arg("input_memory_config"),
        nb::arg("input_layout"),
        nb::arg("in_channels"),
        nb::arg("out_channels"),
        nb::arg("batch_size"),
        nb::arg("input_height"),
        nb::arg("input_width"),
        nb::arg("kernel_size"),
        nb::arg("stride"),
        nb::arg("padding"),
        nb::arg("dilation"),
        nb::arg("groups"),
        nb::arg("device"),
        nb::arg("input_dtype"),
        nb::arg("output_dtype") = nb::none(),
        nb::arg("conv_config") = nb::none(),
        nb::arg("compute_config") = nb::none(),
        nb::arg("dram_slice_config") = nb::none());
}

}  // namespace ttnn::operations::conv::conv_transpose2d
