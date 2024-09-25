// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace ttnn::operations::examples {

void py_module(py::module& module);

}  // namespace ttnn::operations::examples
