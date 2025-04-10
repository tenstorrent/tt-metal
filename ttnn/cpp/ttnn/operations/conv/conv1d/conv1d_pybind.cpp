// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "cpp/pybind11/decorators.hpp"

#include "conv1d_pybind.hpp"
#include "conv1d.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations::conv {
namespace conv1d {

void py_bind_conv1d(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::conv1d,
        R"doc(
Applies a 1D convolution over an input signal composed of several input planes. Implemented as a 2D Convolution of input height 1 and input width as input_length.

:param ttnn.Tensor input_tensor:  the input tensor.
:param ttnn.Tensor weight_tensor: the weight tensor.
:param ttnn.Tensor, None bias_tensor:   optional bias tensor. Default: None
:param ttnn.IDevice device:  the device to use.
:param int: in_channels:  number of input channels.
:param int: out_channels:  number of output channels.
:param int: batch_size:  batch size.
:param int: input_length:  length of the input tensor.
:param int kernel_size: size of the convolving kernel.
:param int stride: stride of the cross-correlation.
:param int padding: zero-padding added to both sides of the input.
:param int dilation: spacing between kernel elements.
:param int groups:  number of blocked connections from input channels to output channels.
:param ttnn.Conv2dConfig, None conv_config: configuration for convolution. Default: None
:param ttnn.DeviceComputeKernelConfig, None compute_config: configuration for compute kernel. Default: None
:param ttnn.MemoryConfig, None memory_config: configuration for memory. Default: None
:param bool: return_output_dim:  If true, the op also returns the height & width of the output tensor in [N, H, W, C] format,
:param bool: return_weights_and_bias:  If true, the op also returns the preprocessed weight and bias on device .

:return: The output tensor, output height & width, and the preprocessed weights & bias.

:rtype: [ttnn.Tensor]: the output tensor, when return_output_dim = False and return_weights_and_bias = False
:rtype: [ttnn.Tensor, Tuple[int, int]]: the output tensor, and it's height & width, if return_output_dim = True
:rtype: [ttnn.Tensor, Tuple[ttnn.Tensor, ttnn.Tensor]]: the output tensor, and it's height & width, if return_weights_and_bias = True
:rtype: [ttnn.Tensor, Tuple[int, int], Tuple[ttnn.Tensor, ttnn.Tensor]]: the output tensor, and it's height & width, if return_output_dim = True and return_weights_and_bias = True
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
}  // namespace conv1d
}  // namespace operations::conv
}  // namespace ttnn
