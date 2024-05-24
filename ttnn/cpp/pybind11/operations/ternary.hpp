// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/ternary.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace ternary {

void py_module(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::where,
        R"doc(where(predicate: ttnn.Tensor, x: ttnn.Tensor, y: ttnn.Tensor, *, memory_config: Optional[MemoryConfig] = None) -> ttnn.Tensor

            Selects elements from x or y based on predicate.

            Args:
                * :attr:`predicate`: Condition Tensor
                * :attr:`x`: Tensor to select elements from when predicate is True
                * :attr:`y`: Tensor to select elements from when predicate is False

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor, if None then it gets set to x.memory_config)doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::where)& self,
               const Tensor& predicate,
               const Tensor& true_value,
               const Tensor& false_value,
               const std::optional<MemoryConfig>& memory_config) {
                return self(predicate, true_value, false_value, memory_config);
            },
            py::arg("predicate"),
            py::arg("true_value"),
            py::arg("false_value"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::where)& self,
               const Tensor& predicate,
               const float true_value,
               const Tensor& false_value,
               const std::optional<MemoryConfig>& memory_config) {
                return self(predicate, true_value, false_value, memory_config);
            },
            py::arg("predicate"),
            py::arg("true_value"),
            py::arg("false_value"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::where)& self,
               const Tensor& predicate,
               const Tensor& true_value,
               const float false_value,
               const std::optional<MemoryConfig>& memory_config) {
                return self(predicate, true_value, false_value, memory_config);
            },
            py::arg("predicate"),
            py::arg("true_value"),
            py::arg("false_value"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::where)& self,
               const Tensor& predicate,
               const float true_value,
               const float false_value,
               const std::optional<MemoryConfig>& memory_config) {
                return self(predicate, true_value, false_value, memory_config);
            },
            py::arg("predicate"),
            py::arg("true_value"),
            py::arg("false_value"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ternary
}  // namespace operations
}  // namespace ttnn
