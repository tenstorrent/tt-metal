// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0



#include "ttnn/cpp/pybind11/decorators.hpp"

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
        Conv Tranpose 2D
        +-------------------+-------------------------------+---------------+-------------+----------+
        | Argument          | Description                   | Data type     | Valid range | Required |
        +===================+===============================+===============+=============+==========+
        | input             | Input activations tensor      | Tensor        |             | Yes      |
        | in_n              | Input nbatch                  | Tensor        |             | Yes      |
        | in_h              | Input height                  | Tensor        |             | Yes      |
        | in_w              | Input width                   | Tensor        |             | Yes      |
        | kernel_h          | kernel window height          | uint32_t      |             | Yes      |
        | kernel_w          | kernel window width           | uint32_t      |             | Yes      |
        | stride_h          | stride in height dim          | uint32_t      |             | No       |
        | stride_w          | stride in width dim           | uint32_t      |             | No       |
        | pad_h             | padding in height dim         | uint32_t      |             | No       |
        | pad_w             | padding in width dim          | uint32_t      |             | No       |
        | out_pad_h         | output padding in height dim  | uint32_t      |             | No       |
        | out_pad_w         | output padding in width dim   | uint32_t      |             | No       |
        | dilation_h        | kernel dilation in height dim | uint32_t      |             | No       |
        | dilation_w        | kernel dilation in width dim  | uint32_t      |             | No       |
        | memory_config     | Output memory config          | MemoryConfig  |             | No       |
        +-------------------+-------------------------------+---------------+-------------+----------+
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::conv_transpose2d)& self, const ttnn::Tensor& input_tensor,
                const ttnn::Tensor& weight_tensor,
                ttnn::Device* device,
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
                std::optional<const conv2d::Conv2dConfig> conv_config,
                const uint8_t& queue_id) -> std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> {
                return self(queue_id, input_tensor, weight_tensor, device, in_channels, out_channels, batch_size, input_height, input_width, kernel_size, stride, padding, output_padding, dilation, groups, bias_tensor, conv_config);
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
            py::arg("queue_id") = 0},

        ttnn::pybind_overload_t{
            [](const decltype(ttnn::conv_transpose2d)& self, const ttnn::Tensor& input_tensor,
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
                std::optional<const conv2d::Conv2dConfig> conv_config,
                const uint8_t& queue_id) -> std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> {
                return self(queue_id, input_tensor, weight_tensor, device, in_channels, out_channels, batch_size, input_height, input_width, kernel_size, stride, padding, output_padding, dilation, groups, bias_tensor, conv_config);
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
            py::arg("queue_id") = 0}
    );
}

}  // namespace conv2d
}  // namespace operations
}  // namespace ttnn
