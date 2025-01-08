// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/distributed_pybind.hpp"
#include <utility>

#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "pybind11/stl.h"

using namespace tt::tt_metal;

namespace ttnn::distributed {

namespace py = pybind11;

void py_module_types(py::module& module) {
    py::class_<MeshDevice, std::shared_ptr<MeshDevice>>(module, "MeshDevice");
    py::class_<MeshSubDeviceManagerId>(module, "MeshSubDeviceManagerId");
    py::class_<MeshShape>(module, "MeshShape", "Struct representing the shape of a mesh device.");
    py::class_<MeshOffset>(module, "MeshOffset", "Struct representing the offset of a mesh device.");
}

void py_module(py::module& module) {
    py::enum_<MeshType>(module, "MeshType")
        .value("RowMajor", MeshType::RowMajor)
        .value("Ring", MeshType::Ring)
        .value("Line", MeshType::Line)
        .export_values();

    static_cast<py::class_<MeshShape>>(module.attr("MeshShape"))
        .def(
            py::init([](size_t num_rows, size_t num_cols) { return MeshShape(num_rows, num_cols); }),
            "Constructor with specified number of rows and columns.",
            py::arg("num_rows"),
            py::arg("num_cols"))
        .def_readwrite("num_rows", &MeshShape::num_rows, "Number of rows in the mesh.")
        .def_readwrite("num_cols", &MeshShape::num_cols, "Number of columns in the mesh.")
        .def(
            "__repr__",
            [](const MeshShape& ms) {
                return "<MeshShape num_rows=" + std::to_string(ms.num_rows) +
                       " num_cols=" + std::to_string(ms.num_cols) + ">";
            })
        .def("__iter__", [](const MeshShape& ms) { return py::iter(py::make_tuple(ms.num_rows, ms.num_cols)); });
    static_cast<py::class_<MeshOffset>>(module.attr("MeshOffset"))
        .def(
            py::init([](size_t row, size_t col) { return MeshOffset(row, col); }),
            "Constructor with specified row and column offsets.",
            py::arg("row"),
            py::arg("col"))
        .def_readwrite("row", &MeshOffset::row, "Row offset in the mesh.")
        .def_readwrite("col", &MeshOffset::col, "Column offset in the mesh.")
        .def(
            "__repr__",
            [](const MeshOffset& mo) {
                return "<MeshOffset row=" + std::to_string(mo.row) + " col=" + std::to_string(mo.col) + ">";
            })
        .def("__iter__", [](const MeshOffset& mo) { return py::iter(py::make_tuple(mo.row, mo.col)); });

    auto py_mesh_device = static_cast<py::class_<MeshDevice, std::shared_ptr<MeshDevice>>>(module.attr("MeshDevice"));
    py_mesh_device
        .def(
            py::init([](const MeshShape& mesh_device_shape,
                        size_t l1_small_size,
                        size_t trace_region_size,
                        size_t num_command_queues,
                        const DispatchCoreConfig& dispatch_core_config,
                        const MeshOffset& offset,
                        const std::vector<chip_id_t>& physical_device_ids,
                        MeshType mesh_type) {
                return MeshDevice::create(
                    MeshDeviceConfig(mesh_device_shape, offset, physical_device_ids, mesh_type),
                    l1_small_size,
                    trace_region_size,
                    num_command_queues,
                    dispatch_core_config);
            }),
            py::kw_only(),
            py::arg("mesh_shape"),
            py::arg("l1_small_size"),
            py::arg("trace_region_size"),
            py::arg("num_command_queues"),
            py::arg("dispatch_core_config"),
            py::arg("offset"),
            py::arg("physical_device_ids"),
            py::arg("mesh_type"))
        .def("get_num_devices", &MeshDevice::num_devices)
        .def("id", &MeshDevice::id)
        .def("get_device_ids", &MeshDevice::get_device_ids)
        .def(
            "get_device",
            py::overload_cast<chip_id_t>(&MeshDevice::get_device, py::const_),
            py::return_value_policy::reference)
        .def(
            "get_device",
            py::overload_cast<size_t, size_t>(&MeshDevice::get_device, py::const_),
            py::return_value_policy::reference)
        .def(
            "get_devices",
            &MeshDevice::get_devices,
            py::return_value_policy::reference,
            py::arg("type") = py::none(),
            R"doc(
            Get the devices in the device mesh.

            Returns:
                List[Device]: The devices in the device mesh.
        )doc")
        .def(
            "create_submesh",
            &MeshDevice::create_submesh,
            py::arg("submesh_shape"),
            py::arg("offset"),
            py::arg("mesh_type"),
            py::keep_alive<1, 0>())  // Keep MeshDevice alive as long as SubmeshDevice is alive
        .def(
            "create_submeshes",
            &MeshDevice::create_submeshes,
            py::arg("submesh_shape"),
            py::arg("mesh_type"),
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
        .def(
            "enable_async",
            &MeshDevice::enable_async,
            py::arg("enable"),
            R"doc(
                Enable or disable async mode across all devices in the mesh.

                Args:
                    enable (bool): True to enable async mode, False to disable it.
            )doc")
        .def(
            "enable_program_cache",
            &MeshDevice::enable_program_cache,
            R"doc(
                Enable program cache across all devices in the mesh.
            )doc")
        .def(
            "disable_and_clear_program_cache",
            &MeshDevice::disable_and_clear_program_cache,
            R"doc(
                Disable program cache across all devices in the mesh.
            )doc")
        .def_property_readonly(
            "shape",
            &MeshDevice::shape,
            R"doc(
            Get the shape of the device mesh.

            Returns:
                Tuple[int, int]: The shape of the device mesh as (num_rows, num_cols).
        )doc")
        .def("__repr__", &MeshDevice::to_string)
        .def(
            "create_sub_device_manager",
            [](MeshDevice& self, const std::vector<SubDevice>& sub_devices, DeviceAddr local_l1_size) {
                return self.create_sub_device_manager(sub_devices, local_l1_size);
            },
            py::arg("sub_devices"),
            py::arg("local_l1_size"),
            R"doc(
                Creates a sub-device manager for the given mesh device.

                Args:
                    sub_devices (List[ttnn.SubDevice]): The sub-devices to include in the sub-device manager.
                    This configuration will be used for each device in the MeshDevice.
                    local_l1_size (int): The size of the local allocators of each sub-device. The global allocator will be shrunk by this amount.

                Returns:
                    MeshSubDeviceManagerId: The ID of the created sub-device manager.
            )doc")
        .def(
            "create_sub_device_manager_with_fabric",
            [](MeshDevice& self, const std::vector<SubDevice>& sub_devices, DeviceAddr local_l1_size) {
                return self.create_sub_device_manager_with_fabric(sub_devices, local_l1_size);
            },
            py::arg("sub_devices"),
            py::arg("local_l1_size"),
            R"doc(
                Creates a sub-device manager for the given mesh device. This will automatically create a sub-device of ethernet cores for use with fabric.
                Note that this is a temporary API until migration to actual fabric is complete.

                Args:
                    sub_devices (List[ttnn.SubDevice]): The sub-devices to include in the sub-device manager. No ethernet cores should be included in this list.
                    This configuration will be used for each device in the MeshDevice.
                    local_l1_size (int): The size of the local allocators of each sub-device. The global allocator will be shrunk by this amount.

                Returns:
                    MeshSubDeviceManagerId: The ID of the created sub-device manager.
                    SubDeviceId: The ID of the sub-device that will be used for fabric.
            )doc")
        .def(
            "load_sub_device_manager",
            &MeshDevice::load_sub_device_manager,
            py::arg("mesh_sub_device_manager_id"),
            R"doc(
                Loads the sub-device manager with the given ID.

                Args:
                    mesh_sub_device_manager_id (MeshSubDeviceManagerId): The ID of the sub-device manager to load.
            )doc")
        .def(
            "clear_loaded_sub_device_manager",
            &MeshDevice::clear_loaded_sub_device_manager,
            R"doc(
                Clears the loaded sub-device manager for the given mesh device.
            )doc")
        .def(
            "remove_sub_device_manager",
            &MeshDevice::remove_sub_device_manager,
            py::arg("mesh_sub_device_manager_id"),
            R"doc(
                Removes the sub-device manager with the given ID.

                Args:
                    mesh_sub_device_manager_id (MeshSubDeviceManagerId): The ID of the sub-device manager to remove.
            )doc")
        .def(
            "set_sub_device_stall_group",
            [](MeshDevice& self, const std::vector<SubDeviceId>& sub_device_ids) {
                self.set_sub_device_stall_group(sub_device_ids);
            },
            py::arg("sub_device_ids"),
            R"doc(
                Set the SubDevice IDs that will be stalled on by default for Fast Dispatch commands such as reading, writing, synchronizing.
                Stalling here refers to the Fast Dispatch cores waiting for programs to complete execution on the specified SubDevices before proceeding with the specified instruction.
                The default SubDevice IDs to stall on are set to all SubDevice IDs, and whenever a new SubDevice Manager is loaded.

                Args:
                    sub_device_ids (List[SubDeviceId]): The IDs of the SubDevices to stall on.
            )doc")
        .def(
            "reset_sub_device_stall_group",
            &MeshDevice::reset_sub_device_stall_group,
            R"doc(
                Resets the sub_device_ids that will be stalled on by default for Fast Dispatch commands such as reading, writing, synchronizing
                back to all SubDevice IDs.
            )doc");

    module.def(
        "open_mesh_device",
        &open_mesh_device,
        py::kw_only(),
        py::arg("mesh_shape"),
        py::arg("l1_small_size"),
        py::arg("trace_region_size"),
        py::arg("num_command_queues"),
        py::arg("offset"),
        py::arg("physical_device_ids"),
        py::arg("mesh_type"),
        py::arg("dispatch_core_config"));

    module.def("close_mesh_device", &close_mesh_device, py::arg("mesh_device"), py::kw_only());
    module.def(
        "get_device_tensor",
        py::overload_cast<const Tensor&, int>(&ttnn::distributed::get_device_tensor),
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
        py::overload_cast<const Tensor&, const IDevice*>(&ttnn::distributed::get_device_tensor),
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
    module.def(
        "aggregate_as_tensor",
        [](const std::vector<Tensor>& tensors) -> Tensor { return aggregate_as_tensor(tensors, AllGatherTensor{}); },
        py::arg("tensors"),
        py::kw_only());
    module.def("get_t3k_physical_device_ids_ring", &get_t3k_physical_device_ids_ring);
}

}  // namespace ttnn::distributed
