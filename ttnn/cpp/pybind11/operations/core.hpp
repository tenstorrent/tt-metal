// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace core {

void py_module(py::module& module) {}

}  // namespace core
}  // namespace operations
}  // namespace ttnn
