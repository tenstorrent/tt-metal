// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace py = pybind11;

namespace ttnn::operations::pool {

void bind_max_pool2d_operation(py::module& module);
void bind_avg_pool2d_operation(py::module& module);
void py_module(py::module& module);

}  // namespace ttnn::operations::pool
