// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "slice.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_slice(py::module& module) {
    auto doc =
        R"doc(
            slice(input_tensor: ttnn.Tensor, slice_start: ttnn.Shape, slice_end: ttnn.Shape,  value: Union[int, float], *, Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Returns a sliced tensor.

            Equivalent pytorch code:

            .. code-block:: python

                output_tensor = input_tensor[output_start: output_end]

            Args:
                * :attr:`input_tensor`: Input Tensor.
                * :attr:`slice_start`: Shape describing where to start slice.
                * :attr:`slice_end`: Shape describing where to end slice.

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id
        )doc";

    using OperationType = decltype(ttnn::slice);
    ttnn::bind_registered_operation(
        module,
        ttnn::slice,
        doc,
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const ttnn::Shape & slice_start,
                const ttnn::Shape & slice_end,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, slice_start, slice_end, memory_config);
                },
                py::arg("input_tensor"),
                py::arg("slice_start"),
                py::arg("slice_end"),
                py::kw_only(),
                py::arg("memory_config") = std::nullopt,
                py::arg("queue_id") = 0,
                }
        );
}
}  // namespace ttnn::operations::data_movement::detail
