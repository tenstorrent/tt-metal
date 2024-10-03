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

void py_module_types(py::module& module) {
    py::class_<MeshDevice, std::shared_ptr<MeshDevice>>(module, "MeshDevice");
}

void py_module(py::module& module) {
    py::enum_<MeshType>(module, "MeshType")
        .value("RowMajor", MeshType::RowMajor)
        .value("Ring", MeshType::Ring)
        .value("Line", MeshType::Line)
        .export_values();
    auto py_mesh_device = static_cast<py::class_<MeshDevice, std::shared_ptr<MeshDevice>>>(module.attr("MeshDevice"));
    py_mesh_device
        .def(
            py::init([](const MeshShape& mesh_device_shape,
                        size_t l1_small_size,
                        size_t trace_region_size,
                        size_t num_command_queues,
                        DispatchCoreType dispatch_core_type,
                        const std::pair<size_t, size_t>& offset,
                        const std::vector<chip_id_t>& physical_device_ids,
                        MeshType mesh_type) {
                return MeshDevice::create(
                    MeshDeviceConfig(mesh_device_shape, offset, physical_device_ids, mesh_type),
                    l1_small_size,
                    trace_region_size,
                    num_command_queues,
                    dispatch_core_type);
            }),
            py::kw_only(),
            py::arg("mesh_shape"),
            py::arg("l1_small_size"),
            py::arg("trace_region_size"),
            py::arg("num_command_queues"),
            py::arg("dispatch_core_type"),
            py::arg("offset"),
            py::arg("physical_device_ids"),
            py::arg("mesh_type"))
        .def("get_num_devices", &MeshDevice::num_devices)
        .def("get_mesh_id", &MeshDevice::get_mesh_id)
        .def("get_device_ids", &MeshDevice::get_device_ids)
        .def(
            "get_device",
            py::overload_cast<chip_id_t>(&MeshDevice::get_device, py::const_),
            py::return_value_policy::reference)
        .def(
            "get_device",
            py::overload_cast<size_t, size_t>(&MeshDevice::get_device, py::const_),
            py::return_value_policy::reference)
        .def("get_devices", &MeshDevice::get_devices, py::return_value_policy::reference, R"doc(
            Get the devices in the device mesh.

            Returns:
                List[Device]: The devices in the device mesh.
        )doc")
        .def("create_submesh", &MeshDevice::create_submesh,
            py::arg("submesh_shape"), py::arg("offset"), py::arg("mesh_type"),
            py::keep_alive<1, 0>())  // Keep MeshDevice alive as long as SubmeshDevice is alive
        .def("create_submeshes", &MeshDevice::create_submeshes,
            py::arg("submesh_shape"), py::arg("mesh_type"),
            py::keep_alive<1, 0>())  // Keep MeshDevice alive as long as SubmeshDevices are alive
        .def(
            "compute_with_storage_grid_size",
            &MeshDevice::compute_with_storage_grid_size,
            R"doc(
            Get the compute grid size (x, y) of the first device in the device mesh denoting region that can be targeted by ops.

            Returns:
                CoreCoord: The compute grid size of the first device in the device mesh.
        )doc")
        .def(
            "dram_grid_size",
            &MeshDevice::dram_grid_size,
            R"doc(
            Get the dram grid size (x, y) of the first device in the device mesh.

            Returns:
                CoreCoord: The dram grid size of the first device in the device mesh.
        )doc")
        .def(
            "arch",
            &MeshDevice::arch,
            R"doc(
            Get the arch of the first device in the device mesh.

            Returns:
                Arch: The arch of the first device in the device mesh.
        )doc")
        .def_property_readonly("shape", &MeshDevice::shape, R"doc(
            Get the shape of the device mesh.

            Returns:
                Tuple[int, int]: The shape of the device mesh as (num_rows, num_cols).
        )doc")
        .def("__repr__", &MeshDevice::to_string);

    module.def(
        "open_mesh_device",
        &open_mesh_device,
        py::kw_only(),
        py::arg("mesh_shape"),
        py::arg("l1_small_size"),
        py::arg("trace_region_size"),
        py::arg("num_command_queues"),
        py::arg("dispatch_core_type"),
        py::arg("offset"),
        py::arg("physical_device_ids"),
        py::arg("mesh_type"));

    module.def("close_mesh_device", &close_mesh_device, py::arg("mesh_device"), py::kw_only());
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
    module.def("get_t3k_physical_device_ids_ring", &tt::tt_metal::get_t3k_physical_device_ids_ring);
}

}  // namespace multi_device

}  // namespace ttnn
