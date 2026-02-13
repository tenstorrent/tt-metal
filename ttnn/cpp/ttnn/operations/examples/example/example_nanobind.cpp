// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "example_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/examples/example/example.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::examples {

void bind_example_operation(nb::module_& mod) {
    ttnn::bind_function<"composite_example">(
        mod,
        R"doc(composite_example(input_tensor: ttnn.Tensor) -> ttnn.Tensor)doc",
        ttnn::overload_t(&ttnn::composite_example, nb::arg("input_tensor")));
}

}  // namespace ttnn::operations::examples
