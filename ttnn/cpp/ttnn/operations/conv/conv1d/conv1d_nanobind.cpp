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

#include "conv1d.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::conv::conv1d {

void bind_conv1d(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::conv1d,
        R"doc(
        Applies a 1D convolution over an input signal composed of several input planes. Implemented as a 2D Convolution of input height 1 and input width as input_length.

        :param ttnn.Tensor input_tensor:  The input tensor. This must be in the format [N, H, W, C]. It can be on host or device.
        :param ttnn.Tensor weight_tensor: The weight tensor. The weights can be passed in the same format as PyTorch, [out_channels, in_channels, kernel_height, kernel_width]. The op w
        :param ttnn.Tensor, None bias_tensor:   Optional bias tensor. Default: None
        :param ttnn.MeshDevice device:  The device to use.
        :param int in_channels:  Number of input channels.
        :param int out_channels:  Number of output channels.
        :param int batch_size:  Batch size.
        :param int input_length:  Length of the input signal.
        :param int kernel_size: Size of the convolving kernel.
        :param int stride: Stride of the cross-correlation.
        :param int or tuple[int, int]) padding: Zero-padding added to both sides of the input. pad_length or [pad_left, pad_right].
        :param int dilation: Spacing between kernel elements.
        :param int groups:  Number of blocked connections from input channels to output channels.
        :param ttnn.DataType, None dtype: The data type of the input tensor. Default: None (will use the same dtype as input_tensor).
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
        ttnn::nanobind_overload_t{
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
            nb::arg("return_weights_and_bias") = false});
}
}  // namespace ttnn::operations::conv::conv1d
