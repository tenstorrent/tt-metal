// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

namespace ttnn {
namespace operations {
namespace ccl {

namespace py = pybind11;

void py_module(py::module& module);

} // namespace ccl
} // namespace operations
} // namespace ttnn
