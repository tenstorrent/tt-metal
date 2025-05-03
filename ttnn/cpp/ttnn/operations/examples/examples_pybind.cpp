// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "examples_pybind.hpp"

#include "ttnn/operations/examples/example/example_pybind.hpp"
#include "ttnn/operations/examples/example_multiple_return/example_multiple_return_pybind.hpp"

namespace ttnn::operations::examples {

void py_module(py::module& module) {
    bind_example_operation(module);
    bind_example_multiple_return_operation(module);
}

}  // namespace ttnn::operations::examples
