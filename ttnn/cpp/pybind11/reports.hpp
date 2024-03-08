// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/reports.hpp"

namespace py = pybind11;

namespace ttnn {
namespace reports {
void py_module(py::module& module) {
    module.def("print_l1_buffers", &print_l1_buffers, py::arg("file_name") = std::nullopt);
}

}  // namespace reports
}  // namespace ttnn
