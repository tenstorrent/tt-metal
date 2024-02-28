// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/matmul.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace matmul {

void py_module(py::module& module) {}

}  // namespace matmul
}  // namespace operations
}  // namespace ttnn
