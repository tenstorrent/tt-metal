// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "examples_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn/operations/examples/example/example_nanobind.hpp"
#include "ttnn/operations/examples/example_multiple_return/example_multiple_return_nanobind.hpp"

namespace ttnn::operations::examples {

void py_module(nb::module_& mod) {
    bind_example_operation(mod);
    bind_example_multiple_return_operation(mod);
}

}  // namespace ttnn::operations::examples
