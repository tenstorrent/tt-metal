// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

#include "ttnn/operations/examples/example/example_pybind.hpp"

namespace ttnn::operations::examples {

void py_module(py::module& module) { bind_example_operation(module); }

}  // namespace ttnn::operations::examples
