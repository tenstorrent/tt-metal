// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/debug/debug_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/debug/apply_device_delay_pybind.hpp"

namespace ttnn::operations::debug {

void py_module(py::module& module) { py_bind_apply_device_delay(module); }

}  // namespace ttnn::operations::debug
