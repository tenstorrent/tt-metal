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

    py::class_<ttnn::reports::DeviceInfo>(module, "DeviceInfo")
        .def_property_readonly("l1_num_banks", [](const ttnn::reports::DeviceInfo& self) { return self.l1_num_banks; })
        .def_property_readonly("l1_bank_size", [](const ttnn::reports::DeviceInfo& self) { return self.l1_bank_size; })
        .def_property_readonly("address_at_first_l1_bank", [](const ttnn::reports::DeviceInfo& self) { return self.address_at_first_l1_bank; })
        .def_property_readonly("address_at_first_l1_cb_buffer", [](const ttnn::reports::DeviceInfo& self) { return self.address_at_first_l1_cb_buffer; })
        .def_property_readonly("num_banks_per_storage_core", [](const ttnn::reports::DeviceInfo& self) { return self.num_banks_per_storage_core; })
        .def_property_readonly("num_compute_cores", [](const ttnn::reports::DeviceInfo& self) { return self.num_compute_cores; })
        .def_property_readonly("num_storage_cores", [](const ttnn::reports::DeviceInfo& self) { return self.num_storage_cores; })
        .def_property_readonly("total_l1_memory", [](const ttnn::reports::DeviceInfo& self) { return self.total_l1_memory; })
        .def_property_readonly("total_l1_for_interleaved_buffers", [](const ttnn::reports::DeviceInfo& self) { return self.total_l1_for_interleaved_buffers; })
        .def_property_readonly("total_l1_for_sharded_buffers", [](const ttnn::reports::DeviceInfo& self) { return self.total_l1_for_sharded_buffers; })
        .def_property_readonly("cb_limit", [](const ttnn::reports::DeviceInfo& self) { return self.cb_limit; });

    module.def("get_device_info", [](const Device& device) -> ttnn::reports::DeviceInfo {
            return ttnn::reports::get_device_info(device);
        },
        py::arg("device")
        );


}

}  // namespace reports
}  // namespace ttnn
