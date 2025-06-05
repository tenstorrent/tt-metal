// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::device {
namespace py = pybind11;

void py_device_module_types(py::module& module);
void py_device_module(py::module& module);

}  // namespace ttnn::device
