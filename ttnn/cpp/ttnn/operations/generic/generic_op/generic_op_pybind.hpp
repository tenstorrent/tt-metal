// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ttnn::operations::generic {

void py_module_types(py::module& module);
void py_module(py::module& module);

}  // namespace ttnn::operations::generic
