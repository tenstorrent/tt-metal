// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ttnn::operations::moreh::moreh_adamw {

void bind_moreh_adamw_operation(py::module& module);

void py_module(py::module& module);

}  // namespace ttnn::operations::moreh::moreh_adamw
