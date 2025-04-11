// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reports.hpp"

#include <nanobind/nanobind.h>

#include "ttnn/reports.hpp"
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/mesh_device.hpp>

using tt::tt_metal::distributed::MeshDevice;

namespace ttnn::reports {

void py_module_types(nb::module_& mod) {
    nb::class_<ttnn::reports::BufferInfo>(mod, "BufferInfo");
    nb::class_<ttnn::reports::BufferPageInfo>(mod, "BufferPageInfo");
    nb::class_<ttnn::reports::DeviceInfo>(mod, "DeviceInfo");
}

void py_module(nb::module_& mod) {
    auto py_buffer_info = static_cast<nb::class_<ttnn::reports::BufferInfo>>(mod.attr("BufferInfo"));
    py_buffer_info.def_prop_ro("device_id", [](const ttnn::reports::BufferInfo& self) { return self.device_id; })
        .def_prop_ro("address", [](const ttnn::reports::BufferInfo& self) { return self.address; })
        .def_prop_ro("max_size_per_bank", [](const ttnn::reports::BufferInfo& self) { return self.max_size_per_bank; })
        .def_prop_ro("buffer_type", [](const ttnn::reports::BufferInfo& self) { return self.buffer_type; })
        .def_prop_ro("buffer_layout", [](const ttnn::reports::BufferInfo& self) { return self.buffer_layout; });
    mod.def("get_buffers", &get_buffers, nb::arg("devices"));
    mod.def("get_buffers", [](MeshDevice* device) { return get_buffers({device}); }, nb::arg("device"));

    auto py_buffer_page_info = static_cast<nb::class_<ttnn::reports::BufferPageInfo>>(mod.attr("BufferPageInfo"));
    py_buffer_page_info
        .def_prop_ro("device_id", [](const ttnn::reports::BufferPageInfo& self) { return self.device_id; })
        .def_prop_ro("address", [](const ttnn::reports::BufferPageInfo& self) { return self.address; })
        .def_prop_ro("core_y", [](const ttnn::reports::BufferPageInfo& self) { return self.core_y; })
        .def_prop_ro("core_x", [](const ttnn::reports::BufferPageInfo& self) { return self.core_x; })
        .def_prop_ro("bank_id", [](const ttnn::reports::BufferPageInfo& self) { return self.bank_id; })
        .def_prop_ro("page_index", [](const ttnn::reports::BufferPageInfo& self) { return self.page_index; })
        .def_prop_ro(
            "page_address", [](const ttnn::reports::BufferPageInfo& self) { return self.page_address; })
        .def_prop_ro("page_size", [](const ttnn::reports::BufferPageInfo& self) { return self.page_size; })
        .def_prop_ro(
            "buffer_type", [](const ttnn::reports::BufferPageInfo& self) { return self.buffer_type; });

    mod.def("get_buffer_pages", &get_buffer_pages, nb::arg("devices"));
    mod.def("get_buffer_pages", [](MeshDevice* device) { return get_buffer_pages({device}); }, nb::arg("device"));

    auto py_device_info = static_cast<nb::class_<ttnn::reports::DeviceInfo>>(mod.attr("DeviceInfo"));
    py_device_info
        .def_prop_ro("num_y_cores", [](const ttnn::reports::DeviceInfo& self) { return self.num_y_cores; })
        .def_prop_ro("num_x_cores", [](const ttnn::reports::DeviceInfo& self) { return self.num_x_cores; })
        .def_prop_ro(
            "num_y_compute_cores", [](const ttnn::reports::DeviceInfo& self) { return self.num_y_compute_cores; })
        .def_prop_ro(
            "num_x_compute_cores", [](const ttnn::reports::DeviceInfo& self) { return self.num_x_compute_cores; })
        .def_prop_ro(
            "worker_l1_size", [](const ttnn::reports::DeviceInfo& self) { return self.worker_l1_size; })
        .def_prop_ro("l1_num_banks", [](const ttnn::reports::DeviceInfo& self) { return self.l1_num_banks; })
        .def_prop_ro("l1_bank_size", [](const ttnn::reports::DeviceInfo& self) { return self.l1_bank_size; })
        .def_prop_ro(
            "address_at_first_l1_bank",
            [](const ttnn::reports::DeviceInfo& self) { return self.address_at_first_l1_bank; })
        .def_prop_ro(
            "address_at_first_l1_cb_buffer",
            [](const ttnn::reports::DeviceInfo& self) { return self.address_at_first_l1_cb_buffer; })
        .def_prop_ro(
            "num_banks_per_storage_core",
            [](const ttnn::reports::DeviceInfo& self) { return self.num_banks_per_storage_core; })
        .def_prop_ro(
            "num_compute_cores", [](const ttnn::reports::DeviceInfo& self) { return self.num_compute_cores; })
        .def_prop_ro(
            "num_storage_cores", [](const ttnn::reports::DeviceInfo& self) { return self.num_storage_cores; })
        .def_prop_ro(
            "total_l1_memory", [](const ttnn::reports::DeviceInfo& self) { return self.total_l1_memory; })
        .def_prop_ro(
            "total_l1_for_tensors", [](const ttnn::reports::DeviceInfo& self) { return self.total_l1_for_tensors; })
        .def_prop_ro(
            "total_l1_for_interleaved_buffers",
            [](const ttnn::reports::DeviceInfo& self) { return self.total_l1_for_interleaved_buffers; })
        .def_prop_ro(
            "total_l1_for_sharded_buffers",
            [](const ttnn::reports::DeviceInfo& self) { return self.total_l1_for_sharded_buffers; })
        .def_prop_ro("cb_limit", [](const ttnn::reports::DeviceInfo& self) { return self.cb_limit; });

    mod.def("get_device_info", &get_device_info, nb::arg("device"));
}

}  // namespace ttnn::reports
