// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"
#include "conv_knit.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_sharded_operation_t>
void bind_conv_knit(pybind11::module& module, const data_movement_sharded_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_sharded_operation_t& self,
               const ttnn::Tensor& input_tensor,
               int kernel_height,
               int num_output_channels,
               int input_width,
               int num_input_channels,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id, input_tensor, kernel_height, num_output_channels, input_width, num_input_channels);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("kernel_height"),
            py::arg("num_output_channels"),
            py::arg("input_width"),
            py::arg("num_input_channels"),
            py::kw_only(),
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace detail

void py_bind_conv_knit(pybind11::module& module) {
    detail::bind_conv_knit(
        module,
        ttnn::conv_knit,
        R"doc(
            conv_knit(input_tensor: ttnn.Tensor, kernel_height: int, kernel_width: int) -> ttnn.Tensor
            Input and output tensors must be height sharded.
            Performs conv knit, using the input tensor, kernel height, and number of output channels.
            Useful after converting transposed conv2d into conv2d, to knit the output tensor.
        )doc");
}

}  // namespace ttnn::operations::data_movement
