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
void py_module_types(py::module& module) {
    py::class_<ttnn::reports::BufferInfo>(module, "BufferInfo");
    py::class_<ttnn::reports::BufferPageInfo>(module, "BufferPageInfo");
    py::class_<ttnn::reports::DeviceInfo>(module, "DeviceInfo");
}

void py_module(py::module& module) {
    auto py_buffer_info = static_cast<py::class_<ttnn::reports::BufferInfo>>(module.attr("BufferInfo"));
    py_buffer_info
        .def_property_readonly("device_id", [](const ttnn::reports::BufferInfo& self) { return self.device_id; })
        .def_property_readonly("address", [](const ttnn::reports::BufferInfo& self) { return self.address; })
        .def_property_readonly(
            "max_size_per_bank", [](const ttnn::reports::BufferInfo& self) { return self.max_size_per_bank; })
        .def_property_readonly("buffer_type", [](const ttnn::reports::BufferInfo& self) { return self.buffer_type; });

    module.def("get_buffers", &get_buffers);

    auto py_buffer_page_info = static_cast<py::class_<ttnn::reports::BufferPageInfo>>(module.attr("BufferPageInfo"));
    py_buffer_page_info
        .def_property_readonly("device_id", [](const ttnn::reports::BufferPageInfo& self) { return self.device_id; })
        .def_property_readonly("address", [](const ttnn::reports::BufferPageInfo& self) { return self.address; })
        .def_property_readonly("core_y", [](const ttnn::reports::BufferPageInfo& self) { return self.core_y; })
        .def_property_readonly("core_x", [](const ttnn::reports::BufferPageInfo& self) { return self.core_x; })
        .def_property_readonly("bank_id", [](const ttnn::reports::BufferPageInfo& self) { return self.bank_id; })
        .def_property_readonly("page_index", [](const ttnn::reports::BufferPageInfo& self) { return self.page_index; })
        .def_property_readonly(
            "page_address", [](const ttnn::reports::BufferPageInfo& self) { return self.page_address; })
        .def_property_readonly("page_size", [](const ttnn::reports::BufferPageInfo& self) { return self.page_size; })
        .def_property_readonly(
            "buffer_type", [](const ttnn::reports::BufferPageInfo& self) { return self.buffer_type; });

    module.def("get_buffer_pages", &get_buffer_pages);

    auto py_device_info = static_cast<py::class_<ttnn::reports::DeviceInfo>>(module.attr("DeviceInfo"));
    py_device_info
        .def_property_readonly("num_y_cores", [](const ttnn::reports::DeviceInfo& self) { return self.num_y_cores; })
        .def_property_readonly("num_x_cores", [](const ttnn::reports::DeviceInfo& self) { return self.num_x_cores; })
        .def_property_readonly(
            "num_y_compute_cores", [](const ttnn::reports::DeviceInfo& self) { return self.num_y_compute_cores; })
        .def_property_readonly(
            "num_x_compute_cores", [](const ttnn::reports::DeviceInfo& self) { return self.num_x_compute_cores; })
        .def_property_readonly(
            "worker_l1_size", [](const ttnn::reports::DeviceInfo& self) { return self.worker_l1_size; })
        .def_property_readonly("l1_num_banks", [](const ttnn::reports::DeviceInfo& self) { return self.l1_num_banks; })
        .def_property_readonly("l1_bank_size", [](const ttnn::reports::DeviceInfo& self) { return self.l1_bank_size; })
        .def_property_readonly(
            "address_at_first_l1_bank",
            [](const ttnn::reports::DeviceInfo& self) { return self.address_at_first_l1_bank; })
        .def_property_readonly(
            "address_at_first_l1_cb_buffer",
            [](const ttnn::reports::DeviceInfo& self) { return self.address_at_first_l1_cb_buffer; })
        .def_property_readonly(
            "num_banks_per_storage_core",
            [](const ttnn::reports::DeviceInfo& self) { return self.num_banks_per_storage_core; })
        .def_property_readonly(
            "num_compute_cores", [](const ttnn::reports::DeviceInfo& self) { return self.num_compute_cores; })
        .def_property_readonly(
            "num_storage_cores", [](const ttnn::reports::DeviceInfo& self) { return self.num_storage_cores; })
        .def_property_readonly(
            "total_l1_memory", [](const ttnn::reports::DeviceInfo& self) { return self.total_l1_memory; })
        .def_property_readonly(
            "total_l1_for_tensors", [](const ttnn::reports::DeviceInfo& self) { return self.total_l1_for_tensors; })
        .def_property_readonly(
            "total_l1_for_interleaved_buffers",
            [](const ttnn::reports::DeviceInfo& self) { return self.total_l1_for_interleaved_buffers; })
        .def_property_readonly(
            "total_l1_for_sharded_buffers",
            [](const ttnn::reports::DeviceInfo& self) { return self.total_l1_for_sharded_buffers; })
        .def_property_readonly("cb_limit", [](const ttnn::reports::DeviceInfo& self) { return self.cb_limit; });

    module.def("get_device_info", &get_device_info, py::arg("device"));
}

}  // namespace reports
}  // namespace ttnn
