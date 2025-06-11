// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::examples {
namespace py = pybind11;

void bind_example_operation(py::module& module);

}  // namespace ttnn::operations::examples

/*
void bind_example_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::prim::example,
        R"doc(example(input_tensor: ttnn.Tensor) -> ttnn.Tensor)doc",

        // Add pybind overloads for the C++ APIs that should be exposed to python
        // There should be no logic here, just a call to `self` with the correct arguments
        // The overload with `queue_id` argument will be added automatically for primitive operations
        // This specific function can be called from python as `ttnn.prim.example(input_tensor)` or
        // `ttnn.prim.example(input_tensor, queue_id=queue_id)`
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::prim::example)& self, const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                return self(input_tensor);
            },
            py::arg("input_tensor")});

    bind_registered_operation(
        module,
        ttnn::composite_example,
        R"doc(composite_example(input_tensor: ttnn.Tensor) -> ttnn.Tensor)doc",

        // Add pybind overloads for the C++ APIs that should be exposed to python
        // There should be no logic here, just a call to `self` with the correct arguments
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::composite_example)& self, const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                return self(input_tensor);
            },
            py::arg("input_tensor")});
}
*/
