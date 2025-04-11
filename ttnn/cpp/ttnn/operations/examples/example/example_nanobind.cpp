// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "example_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/examples/example/example.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::examples {

void bind_example_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::prim::example,
        R"doc(example(input_tensor: ttnn.Tensor) -> ttnn.Tensor)doc",

        // Add nanobind overloads for the C++ APIs that should be exposed to python
        // There should be no logic here, just a call to `self` with the correct arguments
        // This specific function can be called from python as `ttnn.prim.example(input_tensor)` or
        // `ttnn.prim.example(input_tensor)`
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::prim::example)& self, const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                return self(input_tensor);
            },
            nb::arg("input_tensor")});

    bind_registered_operation(
        mod,
        ttnn::composite_example,
        R"doc(composite_example(input_tensor: ttnn.Tensor) -> ttnn.Tensor)doc",

        // Add nanobind overloads for the C++ APIs that should be exposed to python
        // There should be no logic here, just a call to `self` with the correct arguments
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::composite_example)& self, const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                return self(input_tensor);
            },
            nb::arg("input_tensor")});
}

}  // namespace ttnn::operations::examples
