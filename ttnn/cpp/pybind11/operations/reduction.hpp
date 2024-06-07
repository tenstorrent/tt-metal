// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/reduction.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace reduction {

namespace detail {

template <typename reduction_operation_t>
void bind_reduction_operation(py::module& module, const reduction_operation_t& operation) {
    auto doc = fmt::format(
        R"doc({0}(input_tensor: ttnn.Tensor, dim: Optional[Union[int, Tuple[int]]] = None, keepdim: bool = True, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor)doc",
        operation.name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("dim") = std::nullopt,
            py::arg("keepdim") = true,
            py::arg("memory_config") = std::nullopt});
}

using argmax_operation_t = decltype(ttnn::argmax);
void bind_reduction_argmax_operation(py::module& module, const argmax_operation_t& operation) {
    auto doc =
        R"doc(argmax(input_tensor: ttnn.Tensor, *, dim: Optional[int] = None, memory_config: MemoryConfig = std::nullopt, output_tensor : Optional[ttnn.Tensor] = std::nullopt, queue_id : [int] = 0) -> ttnn.Tensor

            Returns the indices of the maximum value of elements in the ``input`` tensor
            If no ``dim`` is provided, it will return the indices of maximum value of all elements in given ``input``

            Input tensor must have BFLOAT16 data type and ROW_MAJOR layout.

            Output tensor will have UINT32 data type.

            Equivalent pytorch code:

            .. code-block:: python

                return torch.argmax(input_tensor, dim=dim)

            Args:
                * :attr:`input_tensor`: Input Tensor for argmax.

            Keyword Args:
                * :attr:`dim`: the dimension to reduce. If None, the argmax of the flattened input is returned
                * :attr:`memory_config`: Memory Config of the output tensor
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id
        )doc";

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [] (const argmax_operation_t& self,
                const ttnn::Tensor& input_tensor,
                const std::optional<int> dim,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                std::optional<ttnn::Tensor> optional_output_tensor,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, dim, memory_config, optional_output_tensor);
                },
                py::arg("input_tensor").noconvert(),
                py::kw_only(),
                py::arg("dim") = std::nullopt,
                py::arg("memory_config") = std::nullopt,
                py::arg("output_tensor") = std::nullopt,
                py::arg("queue_id") = 0});
}

}  // namespace detail

void py_module(py::module& module) {
    // Generic reductions
    detail::bind_reduction_operation(module, ttnn::sum);
    detail::bind_reduction_operation(module, ttnn::mean);
    detail::bind_reduction_operation(module, ttnn::max);
    detail::bind_reduction_operation(module, ttnn::min);
    detail::bind_reduction_operation(module, ttnn::std);
    detail::bind_reduction_operation(module, ttnn::var);

    // Special reductions
    detail::bind_reduction_argmax_operation(module, ttnn::argmax);
}

}  // namespace reduction
}  // namespace operations
}  // namespace ttnn
