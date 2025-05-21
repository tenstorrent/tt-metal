// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fill_pad_pybind.hpp"
#include "fill_pad.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::data_movement {
namespace detail {
namespace py = pybind11;

void bind_fill_pad_op(py::module& module) {
    auto doc = fmt::format(
        R"doc(

            Fills the implicit padding of a tiled input tensor with the specified value.
            Specifically, any nD tensor will have the implicit padding of the last 2 dims that exists from [height:tile_height, width:tile_width] filled with the specified value.

            +----------+-----------------------------------------+-----------------------+------------------------+----------+
            | Argument     | Description                         | Data type             | Valid range            | Required |
            +==========+=========================================+=======================+========================+==========+
            | input_tensor | A tiled input tensor                | tt_lib.tensor.Tensor  |                        | Yes      |
            +----------+-----------------------------------------+-----------------------+------------------------+----------+
            | fill_value   | value to fill into padding          | float                 | [-inf , inf]           | Yes      |
            +----------+-----------------------------------------+-----------------------+------------------------+----------+

            Args:
                input_tensor (ttnn.tensor): Any input tensor with desired device and data types for output tensor.
                value greater than 0 fill_value (float): Value to fill the tensor padding with.

            Keyword args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to
                `None`. queue_id (int, optional): command queue id. Defaults to `0`.

            Returns:
                ttnn.Tensor: the output tensor.

        )doc",
        ttnn::fill_implicit_tile_padding.base_name());

    using OperationType = decltype(ttnn::fill_implicit_tile_padding);
    ttnn::bind_registered_operation(
        module,
        ttnn::fill_implicit_tile_padding,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& input_tensor,
               const float fill_value,
               const std::optional<MemoryConfig>& memory_config,
               QueueId queue_id) { return self(queue_id, input_tensor, fill_value, memory_config); },
            py::arg("input_tensor"),
            py::arg("fill_value"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace detail

void bind_fill_pad(py::module& module) { detail::bind_fill_pad_op(module); }

}  // namespace ttnn::operations::data_movement
