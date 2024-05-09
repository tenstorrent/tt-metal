// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../decorators.hpp"
#include "ttnn/operations/unary.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace unary {

namespace detail {

template <typename unary_operation_t>
void bind_unary(py::module& module, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc({0}(input_tensor: ttnn.Tensor, *, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            {0}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor`

        Keyword Args:
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor)
    )doc",
        operation.name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_arguments_t{py::arg("input_tensor"), py::kw_only(), py::arg("memory_config") = std::nullopt});
}

template <typename unary_operation_t>
void bind_unary_with_bool_parameter_set_to_false_by_default(py::module& module, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc({0}(input_tensor: ttnn.Tensor, *, parameter: float, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            {0}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor`

        Keyword Args:
            * :attr:`parameter` (float): Parameter for the operation.
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = {1}(tensor, parameter=true)
    )doc",
        operation.name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("parameter") = false,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace detail

void py_module(py::module& module) {
    detail::bind_unary_with_bool_parameter_set_to_false_by_default(module, ttnn::exp);
    detail::bind_unary(module, ttnn::silu);
}

}  // namespace unary
}  // namespace operations
}  // namespace ttnn
