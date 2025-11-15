// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv1d_pybind.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <variant>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tt-metalium/constants.hpp>
#include "ttnn-pybind/decorators.hpp"

#include "conv1d.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::conv::conv1d {

void py_bind_conv1d(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::conv1d,
        R"doc(
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
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::conv1d)& self,
               ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               ttnn::MeshDevice* device,
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
               std::optional<const ttnn::Tensor> bias_tensor,
               const std::optional<const Conv1dConfig>& conv_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_config,
               const std::optional<const MemoryConfig>& memory_config,
               bool return_output_dim,
               bool return_weights_and_bias) -> Result {
                return self(
                    input_tensor,
                    weight_tensor,
                    device,
                    in_channels,
                    out_channels,
                    batch_size,
                    input_length,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    dtype,
                    bias_tensor,
                    conv_config,
                    compute_config,
                    memory_config,
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
            py::arg("input_length"),
            py::arg("kernel_size"),
            py::arg("stride") = 1,
            py::arg("padding") = 0,
            py::arg("dilation") = 1,
            py::arg("groups") = 1,
            py::arg("dtype") = std::nullopt,
            py::arg("bias_tensor") = std::nullopt,
            py::arg("conv_config") = std::nullopt,
            py::arg("compute_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("return_output_dim") = false,
            py::arg("return_weights_and_bias") = false});
}
}  // namespace ttnn::operations::conv::conv1d
