// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::global_semaphore {

namespace py = pybind11;
void py_module_types(py::module& module);
void py_module(py::module& module);

}  // namespace ttnn::global_semaphore
