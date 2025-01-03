// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fill_rm_pybind.hpp"
#include "fill_rm.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::data_movement {
namespace detail {
namespace py = pybind11;

void bind_fill_pad_op(py::module& module) {
    auto doc = fmt::format(
        R"doc(

            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | Argument     | Description                                                           | Data type             | Valid range            | Required |
            +==========+=======================================================================+=======================+========================+==========+
            | input_tensor | Any input tensor with desired device and data types for output tensor | tt_lib.tensor.Tensor  |                        | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | fill_value   | value to fill into padding                                            | float                 |                        | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+

            Args:
                input_tensor (ttnn.tensor): Any input tensor with desired device and data types for output tensor. value greater than 0
                fill_value (float): Value to fill the tensor padding with.

            Keyword args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

            Returns:
                ttnn.Tensor: the output tensor.

        )doc",
        ttnn::fill_pad.base_name());

    using OperationType = decltype(ttnn::fill_rm);
    ttnn::bind_registered_operation(
        module,
        ttnn::fill_pad,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& input_tensor,
               const float fill_value,
               const std::optional<MemoryConfig>& memory_config,
               uint8_t queue_id) { return self(queue_id, any, fill_value, memory_config); },
            py::arg("input_tensor"),
            py::arg("fill_value"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0});
}

void bind_fill_pad(py::module& module) { detail::bind_fill_pad_op(module); }

}  // namespace ttnn::operations::data_movement
