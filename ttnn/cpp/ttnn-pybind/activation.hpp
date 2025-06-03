// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::activation {

namespace py = pybind11;

void py_module_types(py::module& m);

void py_module(py::module& m);
}  // namespace ttnn::activation
