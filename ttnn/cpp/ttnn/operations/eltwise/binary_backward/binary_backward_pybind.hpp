// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/binary_backward/binary_backward.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace binary_backward {

namespace detail {

void bind_global_backward(py::module& module) {
    auto doc = fmt::format(
R"doc({0}(input_tensor: ttnn.Tensor, dtype: ttnn.DataType, *, memory_config: Optional[ttnn.MemoryConfig] = None, output_tensor : Optional[ttnn.Tensor] = None, queue_id : Optional[int]) -> ttnn.Tensor

Applies {0} to :attr:`input_tensor`.

Args:
    * :attr:`input_tensor` (ttnn.Tensor): input tensors must be on device, in ROW MAJOR or TILE layout
    * :attr:`dtype` (Optional[ttnn.DataType]): data type must be one of the following types BFLOAT16,BFLOAT8_B,BFLOAT4_B,UINT32,INT32 and UINT16.
    *
Keyword Args:
    * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
    * :attr:`output_tensor` (Optional[ttnn.Tensor]): Preallocated tensor to store the output.

Returns:
    ttnn.Tensor: The tensor with the updated data type. Output tensor will be on device, in same layout, and have the given data type.

Example::

    >>> tensor = ttnn.typecast(torch.randn((10, 3, 32, 32), dtype=ttnn.bfloat16), ttnn.uint16)
)doc",
        ttnn::atan2_bw.name());


    using BackwardType = decltype(ttnn::atan2_bw);
    bind_registered_operation(
        module,
        ttnn::atan2_bw,
        doc,
        ttnn::pybind_overload_t{
            [](const BackwardType& self,
               const ttnn::Tensor& grad_tensor,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const ttnn::MemoryConfig& memory_config) -> std::vector<Tensor> {
                return self(grad_tensor, input_tensor_a, input_tensor_b, memory_config);
            },
            py::arg("grad_tensor"),
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("memory_config")});
}

}  // namespace detail

void py_module(py::module& module) { detail::bind_global_backward(module); }

}  // namespace copy
}  // namespace operations
}  // namespace ttnn
