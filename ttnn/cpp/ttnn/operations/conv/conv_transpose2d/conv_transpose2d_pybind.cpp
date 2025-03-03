// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/pybind11/decorators.hpp"

#include "conv_transpose2d_pybind.hpp"
#include "conv_transpose2d.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations::conv {
namespace conv_transpose2d {

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

        Keyword Args:
            input_tensor   (ttnn.Tensor): the input tensor.
            weight_tensor  (ttnn.Tensor): the weight tensor.
            device         (ttnn.Device): the device on which to run the operation.
            in_channels    (int): the number of input channels.
            out_channels   (int): the number of output channels.
            batch_size     (int): the batch size.
            input_height   (int): the input height.
            input_width    (int): the input width.
            kernel_size    (list[int]): the kernel size.
            stride         (list[int]): the stride of the forward Conv2d. Actually corresponds to the dilation of the input_tensor
            padding        (list[int]): the padding of the forward Conv2d. Increasing padding reduces the output size.
            output_padding (list[int]): the output padding. Additional padding used when stride > 1, to specify the exact output size.
            dilation       (list[int]): kernel dilation.
            groups         (int): the number of groups for grouped convolution.
            bias_tensor    (ttnn.Tensor, optional): the bias tensor. Defaults to `None`.
            conv_config    (ttnn.Conv2dConfig, optional): the configuration for the convolution operation. Defaults to `None`.
            compute_config (ttnn.DeviceComputeKernelConfig, optional): the configuration for the compute kernel. Defaults to `None`.
            memory_config  (ttnn.MemoryConfig, optional): the memory configuration of the output.
            mirror_kernel  (bool): Set to true if the op should mirror the kernels along the height & width axes.
            queue_id       (int): the queue id to use for the operation. Defaults to `0`.

        Returns:
            (ttnn.Tensor, int, int, ttnn.Tensor, ttnn.Tensor): the output tensor, the output height, the output width, & the on-device weight and the bias tensor.

        Example:
            >>> [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv_transpose2d(
                    input_tensor=tt_input_tensor,
                    weight_tensor=tt_weight_tensor,
                    in_channels=input_channels,
                    out_channels=output_channels,
                    device=device,
                    bias_tensor=tt_bias_tensor,
                    kernel_size=(filter_height, filter_width),
                    stride=(stride_h, stride_w),
                    padding=(pad_h, pad_w),
                    output_padding=(out_pad_h, out_pad_w),
                    dilation=(dilation, dilation),
                    batch_size=batch_size,
                    input_height=input_height,
                    input_width=input_width,
                    conv_config=conv_config,
                    compute_config=compute_config,
                    groups=groups,
                )
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
               std::optional<const ttnn::Tensor> bias_tensor,
               const std::optional<const Conv2dConfig>& conv_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_config,
               const std::optional<const MemoryConfig>& memory_config,
               bool mirror_kernel,
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
                    bias_tensor,
                    conv_config,
                    compute_config,
                    memory_config,
                    mirror_kernel);
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
            py::arg("stride"),
            py::arg("padding"),
            py::arg("output_padding"),
            py::arg("dilation"),
            py::arg("groups"),
            py::arg("bias_tensor") = std::nullopt,
            py::arg("conv_config") = std::nullopt,
            py::arg("compute_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("mirror_kernel") = true,
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
               std::optional<const ttnn::Tensor> bias_tensor,
               const std::optional<const Conv2dConfig>& conv_config,
               const std::optional<const DeviceComputeKernelConfig>& compute_config,
               const std::optional<const MemoryConfig>& memory_config,
               bool mirror_kernel,
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
                    bias_tensor,
                    conv_config,
                    compute_config,
                    memory_config,
                    mirror_kernel);
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
            py::arg("stride"),
            py::arg("padding"),
            py::arg("output_padding"),
            py::arg("dilation"),
            py::arg("groups"),
            py::arg("bias_tensor") = std::nullopt,
            py::arg("conv_config") = std::nullopt,
            py::arg("compute_config") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("mirror_kernel") = true,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace conv_transpose2d
}  // namespace operations::conv
}  // namespace ttnn
