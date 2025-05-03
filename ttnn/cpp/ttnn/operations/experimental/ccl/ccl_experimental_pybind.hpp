// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ttnn::operations::experimental::ccl {

void py_module(py::module& module);

}  // namespace ttnn::operations::experimental::ccl
