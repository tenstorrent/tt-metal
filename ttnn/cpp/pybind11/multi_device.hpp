// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/multi_device.hpp"

namespace py = pybind11;

namespace ttnn {

namespace multi_device {

void py_module(py::module& module) {
    py::class_<DeviceMesh>(module, "DeviceMesh")
        .def(
            py::init<DeviceGrid, std::vector<int>, size_t, size_t, size_t>(),
            py::kw_only(),
            py::arg("device_grid"),
            py::arg("device_ids"),
            py::arg("l1_small_size"),
            py::arg("trace_region_size"),
            py::arg("num_command_queues"))
        .def("get_num_devices", &ttnn::multi_device::DeviceMesh::num_devices)
        .def("get_device_ids", &ttnn::multi_device::DeviceMesh::get_device_ids)
        .def(
            "get_device",
            py::overload_cast<int>(&ttnn::multi_device::DeviceMesh::get_device, py::const_),
            py::return_value_policy::reference)
        .def(
            "get_device",
            py::overload_cast<int, int>(&ttnn::multi_device::DeviceMesh::get_device, py::const_),
            py::return_value_policy::reference)
        .def("get_devices", &ttnn::multi_device::DeviceMesh::get_devices, py::return_value_policy::reference, R"doc(
            Get the devices in the device mesh.

            Returns:
                List[Device]: The devices in the device mesh.
        )doc")
        .def(
            "get_devices_on_row",
            &ttnn::multi_device::DeviceMesh::get_devices_on_row,
            py::return_value_policy::reference,
            R"doc(
            Get the devices in a row of the device mesh.

            Returns:
                List[Device]: The devices on a row in the device mesh.
        )doc")
        .def(
            "get_devices_on_column",
            &ttnn::multi_device::DeviceMesh::get_devices_on_column,
            py::return_value_policy::reference,
            R"doc(
            Get the devices in a row of the device mesh.

            Returns:
                List[Device]: The devices on a row in the device mesh.
        )doc");

    module.def(
        "open_device_mesh",
        &open_device_mesh,
        py::kw_only(),
        py::arg("device_grid"),
        py::arg("device_ids"),
        py::arg("l1_small_size"),
        py::arg("trace_region_size"),
        py::arg("num_command_queues"));

    module.def("close_device_mesh", &close_device_mesh, py::arg("device_mesh"), py::kw_only());
    module.def(
        "get_device_tensor",
        py::overload_cast<const Tensor&, int>(&tt::tt_metal::get_device_tensor),
        py::arg("tensor"),
        py::arg("device_id"),
        py::kw_only(),
        R"doc(
        Get the tensor shard corresponding to the device_id.

        Args:
            tensor (Tensor): The tensor to get the shard from.
            device_id (int): The device id to get the shard for.

        Returns:
            Tensor: The shard of the tensor corresponding to the device_id.
    )doc");
    module.def(
        "get_device_tensor",
        py::overload_cast<const Tensor&, const Device*>(&tt::tt_metal::get_device_tensor),
        py::arg("tensor"),
        py::arg("device"),
        py::kw_only(),
        R"doc(
        Get the tensor shard corresponding to the device.

        Args:
            tensor (Tensor): The tensor to get the shard from.
            device (Device): The device to get the shard for.

        Returns:
            Tensor: The shard of the tensor corresponding to the device.
    )doc");
    module.def("get_device_tensors", &get_device_tensors, py::arg("tensor"), py::kw_only());
    module.def("aggregate_as_tensor", &aggregate_as_tensor, py::arg("tensors"), py::kw_only());
}

}  // namespace multi_device

}  // namespace ttnn
