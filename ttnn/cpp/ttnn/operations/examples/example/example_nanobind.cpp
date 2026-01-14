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
        ttnn::composite_example,
        R"doc(composite_example(input_tensor: ttnn.Tensor) -> ttnn.Tensor)doc",

        // Add nanobind overloads for the C++ APIs that should be exposed to python
        // There should be no logic here, just a call to `self` with the correct arguments
        ttnn::nanobind_arguments_t{nb::arg("input_tensor")});
}

}  // namespace ttnn::operations::examples
