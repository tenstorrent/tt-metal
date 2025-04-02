// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"
#include <string>

namespace ttnn::operations::binary_ng {

namespace py = pybind11;

void py_module(py::module& module);

}  // namespace ttnn::operations::binary_ng
