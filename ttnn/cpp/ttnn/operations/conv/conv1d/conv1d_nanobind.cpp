// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv1d_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include <tt-metalium/constants.hpp>
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include "conv1d.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::conv::conv1d {

void bind_conv1d(nb::module_& mod) {
    const auto* doc = R"doc(
        Applies a 1D convolution over an input signal composed of several input planes. Implemented as a 2D Convolution of input height 1 and input width as input_length.

        Args:
            input_tensor (ttnn.Tensor): The input tensor. This must be in the format [N, H, W, C]. It can be on host or device.
            weight_tensor (ttnn.Tensor): The weight tensor. The weights can be passed in the same format as PyTorch, [out_channels, in_channels, kernel_height, kernel_width].
            device (ttnn.MeshDevice): The device to use.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            batch_size (int): Batch size.
            input_length (int): Length of the input signal.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the cross-correlation.
            padding (int or tuple[int, int]): Zero-padding added to both sides of the input. pad_length or [pad_left, pad_right].
            dilation (int): Spacing between kernel elements.
            groups (int): Number of blocked connections from input channels to output channels.

        Keyword Args:
            bias_tensor (ttnn.Tensor, optional): Optional bias tensor. Default: None
            dtype (ttnn.DataType, optional): The data type of the input tensor. Default: None (will use the same dtype as input_tensor).
            conv_config (ttnn.Conv2dConfig, optional): Configuration for convolution. Default: None
            compute_config (ttnn.DeviceComputeKernelConfig, optional): Configuration for compute kernel. Default: None
            memory_config (ttnn.MemoryConfig, optional): Output Tensor's Memory Configuration. Default: None
            return_output_dim (bool, optional): If true, the op also returns the length of the output tensor. Default: False
            return_weights_and_bias (bool, optional): If true, the op also returns the preprocessed weight and bias on device. Default: False

        Returns:
            The output tensor, output length, and the preprocessed weights and bias.

            - ttnn.Tensor: The output tensor, when return_output_dim = False and return_weights_and_bias = False
            - tuple[ttnn.Tensor, int]: The output tensor, and its length, if return_output_dim = True
            - tuple[ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor]]: The output tensor, and its weights and biases, if return_weights_and_bias = True
            - tuple[ttnn.Tensor, int, tuple[ttnn.Tensor, ttnn.Tensor]]: The output tensor, its length, and its weights and biases, if return_output_dim = True and return_weights_and_bias = True
        )doc";

    ttnn::bind_function<"conv1d">(
        mod,
        doc,
        &ttnn::conv1d,
        nb::kw_only(),
        nb::arg("input_tensor"),
        nb::arg("weight_tensor"),
        nb::arg("device"),
        nb::arg("in_channels"),
        nb::arg("out_channels"),
        nb::arg("batch_size"),
        nb::arg("input_length"),
        nb::arg("kernel_size"),
        nb::arg("stride") = 1,
        nb::arg("padding") = 0,
        nb::arg("dilation") = 1,
        nb::arg("groups") = 1,
        nb::arg("dtype") = nb::none(),
        nb::arg("bias_tensor") = nb::none(),
        nb::arg("conv_config") = nb::none(),
        nb::arg("compute_config") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("return_output_dim") = false,
        nb::arg("return_weights_and_bias") = false);
}
}  // namespace ttnn::operations::conv::conv1d
