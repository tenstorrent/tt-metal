// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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

        :param ttnn.Tensor input_tensor:  The input tensor. This must be in the format [N, H, W, C]. It can be on host or device.
        :param ttnn.Tensor weight_tensor: The weight tensor. The weights can be passed in the same format as PyTorch, [out_channels, in_channels, kernel_height, kernel_width]. The op w
        :param ttnn.Tensor, None bias_tensor:   Optional bias tensor. Default: None
        :param ttnn.IDevice device:  The device to use.
        :param int in_channels:  Number of input channels.
        :param int out_channels:  Number of output channels.
        :param int batch_size:  Batch size.
        :param int input_length:  Length of the input signal.
        :param int kernel_size: Size of the convolving kernel.
        :param int stride: Stride of the cross-correlation.
        :param int or tuple[int, int]) padding: Zero-padding added to both sides of the input. pad_length or [pad_left, pad_right].
        :param int dilation: Spacing between kernel elements.
        :param int groups:  Number of blocked connections from input channels to output channels.
        :param ttnn.Conv2dConfig, None conv_config: Configuration for convolution. Default: None
        :param ttnn.DeviceComputeKernelConfig, None compute_config: Configuration for compute kernel. Default: None
        :param ttnn.MemoryConfig, None memory_config: Output Tensor's Memory Configuration. Default: None
        :param bool return_output_dim:  If true, the op also returns the height and width of the output tensor in [N, H, W, C] format,
        :param bool return_weights_and_bias:  If true, the op also returns the preprocessed weight and bias on device .

        :return: The output tensor, output height and width, and the preprocessed weights and bias.

        :rtype: [ttnn.Tensor]: The output tensor, when return_output_dim = False and return_weights_and_bias = False
        :rtype: [ttnn.Tensor, int]: The output tensor, and it's length, if return_output_dim = True
        :rtype: [ttnn.Tensor, Tuple[ttnn.Tensor, ttnn.Tensor]]: The output tensor, it's weights and biases, if return_weights_and_bias = True
        :rtype: [ttnn.Tensor, int,  Tuple[ttnn.Tensor, ttnn.Tensor]]: The output tensor,it's length, it's weights and biases, if return_output_dim = True and return_weights_and_bias = True
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::conv1d)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               ttnn::IDevice* device,
               uint32_t in_channels,
               uint32_t out_channels,
               uint32_t batch_size,
               uint32_t input_length,
               uint32_t kernel_size,
               uint32_t stride,
               std::variant<std::array<uint32_t, 2>, uint32_t> padding,
               uint32_t dilation,
               uint32_t groups,
               std::optional<const ttnn::Tensor> bias_tensor,
               const std::optional<const Conv1dConfig>& conv_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_config,
               const std::optional<const MemoryConfig>& memory_config,
               bool return_output_dim,
               bool return_weights_and_bias,
               QueueId queue_id) -> Result {
                return self(
                    queue_id,
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
            py::arg("bias_tensor") = std::nullopt,
            py::arg("conv_config") = std::nullopt,
            py::arg("compute_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("return_output_dim") = false,
            py::arg("return_weights_and_bias") = false,
            py::arg("queue_id") = DefaultQueueId},

        ttnn::pybind_overload_t{
            [](const decltype(ttnn::conv1d)& self,
               const ttnn::Tensor& input_tensor,
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
               std::optional<const ttnn::Tensor> bias_tensor,
               const std::optional<const Conv1dConfig>& conv_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_config,
               const std::optional<const MemoryConfig>& memory_config,
               bool return_output_dim,
               bool return_weights_and_bias,
               QueueId queue_id) -> Result {
                return self(
                    queue_id,
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
            py::arg("bias_tensor") = std::nullopt,
            py::arg("conv_config") = std::nullopt,
            py::arg("compute_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("return_output_dim") = false,
            py::arg("return_weights_and_bias") = false,
            py::arg("queue_id") = DefaultQueueId});
}
}  // namespace ttnn::operations::conv::conv1d
