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

    py::class_<ttnn::reports::BufferPage>(module, "BufferPage")
        .def_property_readonly("address", [](const ttnn::reports::BufferPage& self) { return self.address; })
        .def_property_readonly("device_id", [](const ttnn::reports::BufferPage& self) { return self.device_id; })
        .def_property_readonly("core_y", [](const ttnn::reports::BufferPage& self) { return self.core_y; })
        .def_property_readonly("core_x", [](const ttnn::reports::BufferPage& self) { return self.core_x; })
        .def_property_readonly("page_index", [](const ttnn::reports::BufferPage& self) { return self.page_index; })
        .def_property_readonly("page_address", [](const ttnn::reports::BufferPage& self) { return self.page_address; })
        .def_property_readonly("page_size", [](const ttnn::reports::BufferPage& self) { return self.page_size; })
        .def_property_readonly("buffer_type", [](const ttnn::reports::BufferPage& self) { return self.buffer_type; });

    module.def("get_buffer_pages", &get_buffer_pages);
}

}  // namespace reports
}  // namespace ttnn
