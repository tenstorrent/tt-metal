// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"
#include "ttnn/global_semaphore.hpp"

namespace py = pybind11;

namespace ttnn::global_semaphore {

void py_module_types(py::module& module);
void py_module(py::module& module);

}  // namespace ttnn::global_semaphore
