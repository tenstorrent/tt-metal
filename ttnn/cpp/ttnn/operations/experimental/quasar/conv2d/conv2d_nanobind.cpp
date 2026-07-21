// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv2d_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/quasar/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/sliding_window/sliding_window_nanobind.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_conv2d(nb::module_& mod) {
    const auto* doc = R"doc(
        Applies a 2D convolution over an input signal composed of several input planes.

        Performs a 2D convolution between the input tensor and weight tensor. A 2D kernel (weights tensor) traverses the image (4D input tensor) and a dot product is computed over the overlapping region. For more information, refer to `CNNs on Tenstorrent Architectures <https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/CNNs/ttcnn.md>`_ tech report.

        Args:
            input_tensor (ttnn.Tensor): The input tensor in [N, H, W, C] format. The tensor can be on either the host or the device.
            weight_tensor (ttnn.Tensor): The convolution weights, typically in [out_channels, in_channels // groups, kernel_height, kernel_width] format.
            device (ttnn.MeshDevice): This is a Tenstorrent-specific parameter. The device which will run the operation.
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the convolution.
            batch_size (int): The batch size of the input tensor.
            input_height (int): This is a Tenstorrent-specific parameter. The height of the input tensor.
            input_width (int): This is a Tenstorrent-specific parameter. The width of the input tensor.
            kernel_size (tuple[int, int]): The size of the convolving kernel.
            stride (tuple[int, int]): The stride of the convolution. Default: (1, 1).
            padding (tuple[int, int] or tuple[int, int, int, int]): Zero-padding added to both sides of the input. Default: (0, 0). [pad_height, pad_width] or [pad_top, pad_bottom, pad_left, pad_right].
            dilation (tuple[int, int]): The spacing between kernel elements. Default: (1, 1).
            groups (int): Number of blocked connections from input channels to output channels. Default: 1.

        Keyword Args:
            dtype (ttnn.DataType, optional): The data type of the output tensor. If not provided, it is inferred from the input tensor.
            bias_tensor (ttnn.Tensor, optional): The bias tensor to be added. Default: None.
            conv_config (ttnn.Conv2dConfig, optional): Configuration for convolution. Default: None.
            compute_config (ttnn.DeviceComputeKernelConfig, optional): Configuration for compute kernel. Default: None
            memory_config (ttnn.MemoryConfig, optional): Output Tensor's Memory Configuration. Default: None.
            slice_config (ttnn.Conv2dSliceConfig, optional): Configuration for slicing input & output tensors in DRAM. If set to None and input is in DRAM, DRAM slicing is automatically enabled. Default: None.
            return_output_dim (bool, optional): If true, the op also returns the height and width of the output tensor in [N, H, W, C] format. Default: False
            return_weights_and_bias (bool, optional): If true, the op also returns the preprocessed weight and bias on device. Default: False

        Returns:
            The output tensor, output height and width, and the preprocessed weights and bias.

            - ttnn.Tensor: Default. The output tensor, when return_output_dim = False and return_weights_and_bias = False
            - tuple[ttnn.Tensor, tuple[int, int]]: The output tensor, and its height and width, if return_output_dim = True
            - tuple[ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor]]: The output tensor, and its height and width, if return_weights_and_bias = True
            - tuple[ttnn.Tensor, tuple[int, int], tuple[ttnn.Tensor, ttnn.Tensor]]: The output tensor, and its height and width, if return_output_dim = True and return_weights_and_bias = True

        Note:
            The `input_tensor` supports the following data type and layout:

            .. list-table:: input_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - FLOAT32
                  - ROW_MAJOR, TILE
                * - BFLOAT16
                  - ROW_MAJOR, TILE
                * - BFLOAT8_B
                  - TILE

            The `output_tensor` supports the following data type and layout:

            .. list-table:: output_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - FLOAT32
                  - ROW_MAJOR, TILE
                * - BFLOAT16
                  - ROW_MAJOR, TILE
                * - BFLOAT8_B
                  - TILE

            The `weights_tensor` on the host, supports the following data type and layout:

            .. list-table:: weights_tensor (host)
                :header-rows: 1

                * - dtype
                  - layout
                * - FLOAT32
                  - ROW_MAJOR
                * - BFLOAT16
                  - ROW_MAJOR

            The `weights_tensor` prepared on device, supports the following data type and layout:

            .. list-table:: weights_tensor (prepared on device)
                :header-rows: 1

                * - dtype
                  - layout
                * - FLOAT32
                  - TILE
                * - BFLOAT16
                  - TILE
                * - BFLOAT8_B
                  - TILE

            The `bias_tensor` on the host, supports the following data type and layout:

            .. list-table:: bias_tensor (host)
                :header-rows: 1

                * - dtype
                  - layout
                * - FLOAT32
                  - ROW_MAJOR
                * - BFLOAT16
                  - ROW_MAJOR

            The `bias_tensor` prepared on device, supports the following data type and layout:

            .. list-table:: bias_tensor (prepared on device)
                :header-rows: 1

                * - dtype
                  - layout
                * - FLOAT32
                  - TILE
                * - BFLOAT16
                  - TILE
                * - BFLOAT8_B
                  - TILE
        )doc";

    ttnn::bind_function<"conv2d", "ttnn.experimental.quasar.">(
        mod,
        doc,
        &ttnn::operations::experimental::quasar::conv2d,
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
        nb::arg("dilation") = nb::cast(std::array<uint32_t, 2>{1, 1}),
        nb::arg("groups") = 1,
        nb::arg("dtype") = nb::none(),
        nb::arg("bias_tensor") = nb::none(),
        nb::arg("conv_config") = nb::none(),
        nb::arg("compute_config") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("slice_config") = nb::none(),
        nb::arg("return_output_dim") = false,
        nb::arg("return_weights_and_bias") = false);
}

}  // namespace ttnn::operations::experimental::quasar::detail
