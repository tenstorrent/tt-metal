// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/device.hpp"

namespace py = pybind11;

namespace ttnn {
namespace device {
void py_module(py::module& module) {
    module.def(
        "open_device", &ttnn::open_device, py::kw_only(), py::arg("device_id"), py::return_value_policy::reference);

    module.def("close_device", &ttnn::close_device, py::arg("device"), py::kw_only());

    module.def("enable_program_cache", &ttnn::enable_program_cache, py::arg("device"), py::kw_only());

    module.def("disable_and_clear_program_cache", &ttnn::disable_and_clear_program_cache, py::arg("device"), py::kw_only());
}

}  // namespace device
}  // namespace ttnn
