// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/multi_device.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace py = pybind11;

namespace ttnn {

namespace multi_device {

void py_module_types(py::module& module) { py::class_<DeviceMesh>(module, "DeviceMesh"); }

void py_module(py::module& module) {
    auto py_device_mesh = static_cast<py::class_<DeviceMesh>>(module.attr("DeviceMesh"));
    py_device_mesh
        .def(
            py::init<DeviceGrid, std::vector<int>, size_t, size_t, size_t, DispatchCoreType>(),
            py::kw_only(),
            py::arg("device_grid"),
            py::arg("device_ids"),
            py::arg("l1_small_size"),
            py::arg("trace_region_size"),
            py::arg("num_command_queues"),
            py::arg("dispatch_core_type"))
        .def("get_num_devices", &DeviceMesh::num_devices)
        .def("get_device_ids", &DeviceMesh::get_device_ids)
        .def(
            "get_device",
            py::overload_cast<int>(&DeviceMesh::get_device, py::const_),
            py::return_value_policy::reference)
        .def(
            "get_device",
            py::overload_cast<int, int>(&DeviceMesh::get_device, py::const_),
            py::return_value_policy::reference)
        .def("get_devices", &DeviceMesh::get_devices, py::return_value_policy::reference, R"doc(
            Get the devices in the device mesh.

            Returns:
                List[Device]: The devices in the device mesh.
        )doc")
        .def(
            "get_devices_on_row",
            &DeviceMesh::get_devices_on_row,
            py::return_value_policy::reference,
            R"doc(
            Get the devices in a row of the device mesh.

            Returns:
                List[Device]: The devices on a row in the device mesh.
        )doc")
        .def(
            "get_devices_on_column",
            &DeviceMesh::get_devices_on_column,
            py::return_value_policy::reference,
            R"doc(
            Get the devices in a row of the device mesh.

            Returns:
                List[Device]: The devices on a row in the device mesh.
        )doc")
        .def(
            "compute_with_storage_grid_size",
            &DeviceMesh::compute_with_storage_grid_size,
            R"doc(
            Get the compute grid size (x, y) of the first device in the device mesh denoting region that can be targeted by ops.

            Returns:
                CoreCoord: The compute grid size of the first device in the device mesh.
        )doc")
        .def(
            "dram_grid_size",
            &DeviceMesh::dram_grid_size,
            R"doc(
            Get the dram grid size (x, y) of the first device in the device mesh.

            Returns:
                CoreCoord: The dram grid size of the first device in the device mesh.
        )doc")
        .def(
            "arch",
            &DeviceMesh::arch,
            R"doc(
            Get the arch of the first device in the device mesh.

            Returns:
                Arch: The arch of the first device in the device mesh.
        )doc")
        .def_property_readonly("shape", &DeviceMesh::shape, R"doc(
            Get the shape of the device mesh.

            Returns:
                Tuple[int, int]: The shape of the device mesh as (num_rows, num_cols).
        )doc");

    module.def(
        "open_device_mesh",
        &open_device_mesh,
        py::kw_only(),
        py::arg("device_grid"),
        py::arg("device_ids"),
        py::arg("l1_small_size"),
        py::arg("trace_region_size"),
        py::arg("num_command_queues"),
        py::arg("dispatch_core_type"));

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
