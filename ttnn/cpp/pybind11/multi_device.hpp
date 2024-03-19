// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/multi_device.hpp"

namespace py = pybind11;

namespace ttnn {

namespace multi_device {

void py_module(py::module& module) {

    py::class_<DeviceMesh>(module, "DeviceMesh")
        .def(py::init<DeviceGrid, std::vector<int>>(), py::kw_only(), py::arg("device_grid"), py::arg("device_ids"))
        .def("get_device", &ttnn::multi_device::DeviceMesh::get_device, py::return_value_policy::reference)
        .def("get_num_devices", &ttnn::multi_device::DeviceMesh::num_devices)
        .def("get_device_ids", &ttnn::multi_device::DeviceMesh::get_device_ids);

    module.def(
        "open_device_mesh", &open_device_mesh, py::kw_only(), py::arg("device_grid"), py::arg("device_ids"));

    module.def("close_device_mesh", &close_device_mesh, py::arg("device_mesh"), py::kw_only());
    module.def("get_device_tensors", &get_device_tensors, py::arg("tensor"), py::kw_only());
    module.def("aggregate_as_tensor", &aggregate_as_tensor, py::arg("tensors"), py::kw_only());
}

}  // namespace multi_device

}  // namespace ttnn
