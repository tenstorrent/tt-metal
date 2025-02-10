// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/distributed_pybind.hpp"
<<<<<<< HEAD
<<<<<<< HEAD
#include <pybind11/cast.h>

<<<<<<< HEAD
#include <ostream>
=======
=======
#include <pybind11/cast.h>
>>>>>>> one type error left
#include <cstddef>
#include <memory>
#include <pybind11/pytypes.h>
>>>>>>> expose classes to python

=======
#include <pybind11/pytypes.h>
>>>>>>> fix rebase
#include <tt-metalium/command_queue.hpp>
<<<<<<< HEAD
#include "tt-metalium/mesh_coord.hpp"
#include "distributed_tensor.hpp"
#include "tt-metalium/assert.hpp"
=======
#include "distributed_tensor.hpp"
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> expose classes to python
#include "ttnn/distributed/api.hpp"
<<<<<<< HEAD
#include "ttnn/distributed/types.hpp"
=======
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/core/core.hpp"
<<<<<<< HEAD
#include "ttnn/tensor/tensor_utils.hpp"
>>>>>>> one type error left
=======
>>>>>>> clean up imports, fix test cases and change them to use mapper/composer functions, fix storage error by using from_device instead of cpu
=======
#include "distributed_tensor.cpp"
=======
>>>>>>> move class definitions from from distributed_tensor.cpp to.hpp so they can be exposed to the pybind.cpp; add dummy void methods in .cpp to satisfy linker; add new constructors and factory methods to fix type errors
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
>>>>>>> one type error left
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/command_queue.hpp>

// This is required for automatic conversions, as in the creation of mesh devices
// https://github.com/tenstorrent/tt-metal/issues/18082
#include "pybind11/stl.h"

using namespace tt::tt_metal;

namespace ttnn::distributed {

namespace py = pybind11;

// Trampoline class to clear virtual method errors
struct ConcreteTensorToMesh : TensorToMesh {
    using TensorToMesh::TensorToMesh;  // Inherit constructors

    std::vector<Tensor> map(const Tensor& tensor) const override {
        PYBIND11_OVERRIDE(std::vector<Tensor>, TensorToMesh, map, tensor);
    }

    tt::tt_metal::DistributedTensorConfig config() const override {
        PYBIND11_OVERRIDE(tt::tt_metal::DistributedTensorConfig, TensorToMesh, config);
    }
};

// Trampoline class to clear virtual method errors
struct ConcreteMeshToTensor : MeshToTensor {
    Tensor compose(const std::vector<Tensor>& tensors) const override {
        PYBIND11_OVERRIDE(Tensor, MeshToTensor, compose, tensors);
    }
};

void py_module_types(py::module& module) {
<<<<<<< HEAD
<<<<<<< HEAD
    py::class_<MeshToTensor, ConcreteMeshToTensor, std::unique_ptr<MeshToTensor>>(module, "CppMeshToTensor");
    py::class_<TensorToMesh, ConcreteTensorToMesh, std::unique_ptr<TensorToMesh>>(module, "CppTensorToMesh");

    py::class_<ReplicateTensorToMesh, TensorToMesh, std::unique_ptr<ReplicateTensorToMesh>>(
        module, "CppReplicateTensorToMesh");
    py::class_<ShardTensorToMesh, TensorToMesh, std::unique_ptr<ShardTensorToMesh>>(module, "CppShardTensorToMesh");
    py::class_<ShardTensorTo2dMesh, TensorToMesh, std::unique_ptr<ShardTensorTo2dMesh>>(
        module, "CppShardTensorTo2dMesh");
    py::class_<ConcatMeshToTensor, MeshToTensor, std::unique_ptr<ConcatMeshToTensor>>(module, "CppConcatMeshToTensor");
    py::class_<Concat2dMeshToTensor, MeshToTensor, std::unique_ptr<Concat2dMeshToTensor>>(
        module, "CppConcat2dMeshToTensor");

    py::class_<ReplicateTensor>(module, "ReplicateTensor");
    py::class_<ShardTensor>(module, "ShardTensor");
    py::class_<ShardTensor2D>(module, "ShardTensor2d");
    py::class_<ShardMesh>(module, "ShardMesh");
    py::class_<AllGatherTensor>(module, "AllGatherTensor");
    py::class_<DistributedTensorConfig>(module, "DistributedTensorConfig");

    py::class_<Shard2dConfig>(module, "Shard2dConfig");
    py::class_<Concat2dConfig>(module, "Concat2dConfig");
=======
    py::class_<MeshToTensor, std::unique_ptr<MeshToTensor>>(module, "MeshToTensor");
    py::class_<TensorToMesh, std::unique_ptr<TensorToMesh>>(module, "TensorToMesh");
    py::class_<TensorToMesh, TensorToMesh>(module, "TensorToMesh");
    py::class_<ShardTensorToMesh, TensorToMesh>(module, "ShardTensorToMesh");
    py::class_<ShardTensorTo2dMesh, TensorToMesh>(module, "ShardTensorTo2dMesh");
    py::class_<ConcatMeshToTensor, MeshToTensor>(module, "ConcatMeshToTensor");
    py::class_<Concat2dMeshToTensor, MeshToTensor>(module, "Concat2dMeshToTensor");
>>>>>>> expose classes to python
=======
    py::class_<MeshToTensor, ConcreteMeshToTensor, std::unique_ptr<MeshToTensor>>(module, "MeshToTensor");
    py::class_<TensorToMesh, ConcreteTensorToMesh, std::unique_ptr<TensorToMesh>>(module, "TensorToMesh");
    py::class_<ReplicateTensorToMesh, TensorToMesh, std::unique_ptr<ReplicateTensorToMesh>>(
        module, "ReplicateTensorToMesh");
    py::class_<ShardTensorToMesh, TensorToMesh, std::unique_ptr<ShardTensorToMesh>>(module, "ShardTensorToMesh");
    py::class_<ShardTensor2dMesh, TensorToMesh, std::unique_ptr<ShardTensor2dMesh>>(module, "ShardTensor2dMesh");
    py::class_<ConcatMeshToTensor, MeshToTensor, std::unique_ptr<ConcatMeshToTensor>>(module, "ConcatMeshToTensor");
<<<<<<< HEAD
    py::class_<Concat2dMeshToTensor, MeshToTensor, std::unique_ptr<Concat2dMeshToTensor>>(
        module, "Concat2dMeshToTensor");
>>>>>>> one type error left
=======
    py::class_<ConcatMesh2dToTensor, MeshToTensor, std::unique_ptr<ConcatMesh2dToTensor>>(
        module, "ConcatMesh2dToTensor");
>>>>>>> fix naming errors, add tests, add imports - TODO, fix weird aliasing error with meshdevice vs ttnn.multidevice.meshdevice

    py::class_<MeshDevice, std::shared_ptr<MeshDevice>>(module, "MeshDevice");
    py::class_<MeshSubDeviceManagerId>(module, "MeshSubDeviceManagerId");
    py::class_<MeshShape>(module, "MeshShape", "Struct representing the shape of a mesh device.");
    py::class_<MeshCoordinate>(module, "MeshCoordinate", "Struct representing the coordinate of a mesh device.");
}

void py_module(py::module& module) {
    // TODO: #17477 - Remove overloads that accept 'row' and 'col'. Instead, use generic ND terms.
    static_cast<py::class_<MeshShape>>(module.attr("MeshShape"))
        .def(
            py::init([](size_t num_rows, size_t num_cols) { return MeshShape(num_rows, num_cols); }),
            "Constructor with the specified number of rows and columns.",
            py::arg("num_rows"),
            py::arg("num_cols"))
        .def(
<<<<<<< HEAD
            py::init([](size_t x, size_t y, size_t z) { return MeshShape(x, y, z); }),
            "Constructor with the specified 3D shape.",
            py::arg("x"),
            py::arg("y"),
            py::arg("z"))
        .def(
            py::init([](const std::vector<uint32_t>& shape) { return MeshShape(shape); }),
            "Constructor with the specified ND shape.",
            py::arg("shape"))
=======
            py::init([](const std::tuple<int, int>& dims) { return MeshShape(std::get<0>(dims), std::get<1>(dims)); }),
            "Constructor with specified number of rows and columns as a tuple (rows, columns).",
            py::arg("dims"))
        .def_readwrite("num_rows", &MeshShape::num_rows, "Number of rows in the mesh.")
        .def_readwrite("num_cols", &MeshShape::num_cols, "Number of columns in the mesh.")
>>>>>>> Replace none types, expose configs, fix tuple errors
        .def(
            "__repr__",
            [](const MeshShape& ms) {
                std::ostringstream str;
                str << ms;
                return str.str();
            })
        .def(
            "__iter__",
            [](const MeshShape& ms) { return py::make_iterator(ms.view().begin(), ms.view().end()); },
            py::keep_alive<0, 1>());
    static_cast<py::class_<MeshCoordinate>>(module.attr("MeshCoordinate"))
        .def(
            py::init([](size_t row, size_t col) { return MeshCoordinate(row, col); }),
            "Constructor with specified row and column offsets.",
            py::arg("row"),
            py::arg("col"))
        .def(
            py::init([](size_t x, size_t y, size_t z) { return MeshCoordinate(x, y, z); }),
            "Constructor with the specified 3D coordinate.",
            py::arg("x"),
            py::arg("y"),
            py::arg("z"))
        .def(
            py::init([](const std::vector<uint32_t>& coords) { return MeshCoordinate(coords); }),
            "Constructor with the specified ND coordinate.",
            py::arg("coords"))
        .def(
            "__repr__",
            [](const MeshCoordinate& mc) {
                std::ostringstream str;
                str << mc;
                return str.str();
            })
        .def(
            "__iter__",
            [](const MeshCoordinate& mc) { return py::make_iterator(mc.coords().begin(), mc.coords().end()); },
            py::keep_alive<0, 1>());

    auto py_mesh_device = static_cast<py::class_<MeshDevice, std::shared_ptr<MeshDevice>>>(module.attr("MeshDevice"));
    py_mesh_device
        .def(
            py::init([](const MeshShape& mesh_shape,
                        size_t l1_small_size,
                        size_t trace_region_size,
                        size_t num_command_queues,
                        const DispatchCoreConfig& dispatch_core_config,
                        const std::optional<MeshCoordinate>& offset,
                        const std::vector<chip_id_t>& physical_device_ids) {
                return MeshDevice::create(
                    MeshDeviceConfig{
                        .mesh_shape = mesh_shape,
                        .offset = offset,
                        .physical_device_ids = physical_device_ids,
                    },
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
            py::arg("physical_device_ids"))
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
            py::keep_alive<1, 0>())  // Keep MeshDevice alive as long as SubmeshDevice is alive
        .def(
            "create_submeshes",
            &MeshDevice::create_submeshes,
            py::arg("submesh_shape"),
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
        .def(
            "reshape",
            &MeshDevice::reshape,
            py::arg("new_shape"),
            R"doc(
               Reshapes the logical mesh and re-maps the physical devices to the new logical coordinates.


               Reshaping Rules:
               1. The old_shape volume must equal the new_shape volume (i.e. number of devices must remain constant)
               2. Line-to-Line Reshaping (when either dimension is 1):
                  - Always possible between 1xN and Nx1 shapes (e.g.: 1x8 <-> 8x1)
               3. Grid-to-Grid Reshaping:
                  - Only possible if the devices can form a connected physical mesh in the new shape
                  - Must maintain physical connectivity between adjacent devices
               4. Line-to-Grid Reshaping:
                  - Only possible if the physical devices can form a connected physical mesh in the new shape
                  - Example: 1x8 -> 2x4 is possible only if physical mesh permits a 2x4 configuration


               Args:
                   new_shape (MeshShape): The new shape of the mesh.


               Raises:
                   RuntimeError: If the reshaping constraints are not met:
                   1. The old_shape volume must equal the new_shape volume (i.e. number of devices must remain constant)
                   2. For Grid-to-Grid or Line-to-Grid reshaping: physical connectivity must be possible with current devices
           )doc")
        .def("__repr__", &MeshDevice::to_string)
        .def(
            "create_sub_device_manager",
            [](MeshDevice& self, const std::vector<SubDevice>& sub_devices, DeviceAddr local_l1_size) {
                return self.mesh_create_sub_device_manager(sub_devices, local_l1_size);
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
                return self.mesh_create_sub_device_manager_with_fabric(sub_devices, local_l1_size);
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
            &MeshDevice::mesh_load_sub_device_manager,
            py::arg("mesh_sub_device_manager_id"),
            R"doc(
               Loads the sub-device manager with the given ID.


               Args:
                   mesh_sub_device_manager_id (MeshSubDeviceManagerId): The ID of the sub-device manager to load.
           )doc")
        .def(
            "clear_loaded_sub_device_manager",
            &MeshDevice::mesh_clear_loaded_sub_device_manager,
            R"doc(
               Clears the loaded sub-device manager for the given mesh device.
           )doc")
        .def(
            "remove_sub_device_manager",
            &MeshDevice::mesh_remove_sub_device_manager,
            py::arg("mesh_sub_device_manager_id"),
            R"doc(
               Removes the sub-device manager with the given ID.


               Args:
                   mesh_sub_device_manager_id (MeshSubDeviceManagerId): The ID of the sub-device manager to remove.
           )doc")
        .def(
            "set_sub_device_stall_group",
            [](MeshDevice& self, const std::vector<SubDeviceId>& sub_device_ids) {
                self.mesh_set_sub_device_stall_group(sub_device_ids);
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
            &MeshDevice::mesh_reset_sub_device_stall_group,
            R"doc(
               Resets the sub_device_ids that will be stalled on by default for Fast Dispatch commands such as reading, writing, synchronizing
               back to all SubDevice IDs.
           )doc");

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    auto py_tensor_to_mesh = static_cast<py::class_<TensorToMesh, ConcreteTensorToMesh, std::unique_ptr<TensorToMesh>>>(
        module.attr("CppTensorToMesh"));
    py_tensor_to_mesh
        .def(py::init([]() -> std::unique_ptr<TensorToMesh> { return std::make_unique<ConcreteTensorToMesh>(); }))
        .def("map", &TensorToMesh::map)
        .def("config", &TensorToMesh::config);
    auto py_replicate_tensor_to_mesh =
        static_cast<py::class_<ReplicateTensorToMesh, std::unique_ptr<ReplicateTensorToMesh>>>(
            module.attr("CppReplicateTensorToMesh"));
    py_replicate_tensor_to_mesh
        .def(
            py::init([](MeshDevice& mesh_device) -> std::unique_ptr<ReplicateTensorToMesh> {
                return std::make_unique<ReplicateTensorToMesh>(ReplicateTensorToMesh(mesh_device.num_devices()));
            }),
            py::arg("mesh_device"))
        .def(
            "map",
            [](const ReplicateTensorToMesh& self, const Tensor& tensor) { return self.map(tensor); },
            py::arg("tensor"))
        .def("config", &ReplicateTensorToMesh::config);
    auto py_shard_tensor_to_mesh = static_cast<py::class_<ShardTensorToMesh, std::unique_ptr<ShardTensorToMesh>>>(
        module.attr("CppShardTensorToMesh"));
    py_shard_tensor_to_mesh
        .def(
            py::init([](MeshDevice& mesh_device, int dim) -> std::unique_ptr<ShardTensorToMesh> {
                return std::make_unique<ShardTensorToMesh>(ShardTensorToMesh(mesh_device.num_devices(), dim));
            }),
            py::arg("mesh_device"),
            py::arg("dim"))
        .def(
            "map",
            [](const ShardTensorToMesh& self, const Tensor& tensor) { return self.map(tensor); },
            py::arg("tensor"))
        .def("config", &ShardTensorToMesh::config);
    auto py_shard_tensor_to_2d_mesh =
        static_cast<py::class_<ShardTensorTo2dMesh, std::unique_ptr<ShardTensorTo2dMesh>>>(
            module.attr("CppShardTensorTo2dMesh"));
    py_shard_tensor_to_2d_mesh
        .def(
            py::init(
                [](MeshDevice& mesh_device,
                   const std::tuple<int, int> mesh_shape,
                   const std::tuple<int, int> dims) -> std::unique_ptr<ShardTensorTo2dMesh> {
                    int shape_rows = std::get<0>(mesh_shape);
                    int shape_cols = std::get<1>(mesh_shape);

                    int config_rows = std::get<0>(dims);
                    int config_cols = std::get<1>(dims);
                    TT_FATAL(
                        config_rows || config_cols,
                        "Sharding a tensor to 2D mesh requires at least one dimension to shard");
                    TT_FATAL(
                        shape_rows <= mesh_device.shape().num_rows &&  //
                            shape_cols <= mesh_device.shape().num_cols,
                        "Device mesh shape does not match the provided mesh shape.");

                    return std::make_unique<ShardTensorTo2dMesh>(
                        MeshShape{.num_rows = shape_rows, .num_cols = shape_cols},
                        Shard2dConfig{.row_dim = std::get<0>(dims), .col_dim = std::get<1>(dims)});
                }),
            py::arg("mesh_device"),
            py::arg("mesh_shape"),
            py::arg("dims"))
        .def(
            "map",
            [](const ShardTensorTo2dMesh& self, const Tensor& tensor) { return self.map(tensor); },
            py::arg("tensor"))
        .def("config", &ShardTensorTo2dMesh::config);
    auto py_mesh_to_tensor = static_cast<py::class_<MeshToTensor, ConcreteMeshToTensor, std::unique_ptr<MeshToTensor>>>(
        module.attr("CppMeshToTensor"));
    py_mesh_to_tensor
        .def(py::init([]() -> std::unique_ptr<MeshToTensor> { return std::make_unique<ConcreteMeshToTensor>(); }))
        .def("compose", &MeshToTensor::compose);
    auto py_concat_mesh_to_tensor = static_cast<py::class_<ConcatMeshToTensor, std::unique_ptr<ConcatMeshToTensor>>>(
        module.attr("CppConcatMeshToTensor"));
    py_concat_mesh_to_tensor
        .def(
            py::init([](MeshDevice& mesh_device, int dim) -> std::unique_ptr<ConcatMeshToTensor> {
                return std::make_unique<ConcatMeshToTensor>(dim);
            }),
            py::arg("mesh_device"),
            py::arg("dim"))
        .def(
            "compose",
            [](const ConcatMeshToTensor& self, const std::vector<Tensor>& tensors) { return self.compose(tensors); },
            py::arg("tensors"));

    auto py_concat_2d_mesh_to_tensor =
        static_cast<py::class_<Concat2dMeshToTensor, std::unique_ptr<Concat2dMeshToTensor>>>(
            module.attr("CppConcat2dMeshToTensor"));
    py_concat_2d_mesh_to_tensor
        .def(
            py::init(
                [](MeshDevice& mesh_device,
                   const std::tuple<int, int> mesh_shape,
                   const std::tuple<int, int> dims) -> std::unique_ptr<Concat2dMeshToTensor> {
                    int row_dim = std::get<0>(dims);
                    int col_dim = std::get<1>(dims);
                    TT_FATAL(
                        std::get<0>(mesh_shape) <= mesh_device.shape().num_rows &&  //
                            std::get<1>(mesh_shape) <= mesh_device.shape().num_cols,
                        "Device mesh shape does not match the provided mesh shape.");

                    TT_FATAL(
                        row_dim != col_dim,
                        "Dimensions in 'dims' must be different; got row_dim: {}, col_dim: {}",
                        row_dim,
                        col_dim);
                    return std::make_unique<Concat2dMeshToTensor>(
                        mesh_device,
                        Concat2dConfig{
                            .row_dim = row_dim,
                            .col_dim = col_dim,
                        });
                }),
            py::arg("mesh_device"),
            py::arg("Mesh_shape"),
            py::arg("dims"))
        .def(
            "compose",
            [](const Concat2dMeshToTensor& self, const std::vector<Tensor>& tensors) -> Tensor {
                return self.compose(tensors);
            },
            py::arg("tensors"));
=======
    auto py_tensor_to_mesh = static_cast<TensorToMesh, std::unique_ptr<TensorToMesh>>>(module.attr("TensorToMesh"));
=======
    auto py_tensor_to_mesh =
        static_cast<py::class_<TensorToMesh, std::unique_ptr<TensorToMesh>>>(module.attr("TensorToMesh"));
>>>>>>> one type error left
=======
    auto py_tensor_to_mesh = static_cast<py::class_<TensorToMesh, ConcreteTensorToMesh, std::unique_ptr<TensorToMesh>>>(
        module.attr("TensorToMesh"));
>>>>>>> move class definitions from from distributed_tensor.cpp to.hpp so they can be exposed to the pybind.cpp; add dummy void methods in .cpp to satisfy linker; add new constructors and factory methods to fix type errors
    py_tensor_to_mesh
        .def(py::init([]() -> std::unique_ptr<TensorToMesh> { return std::make_unique<ConcreteTensorToMesh>(); }))
        .def("map", &TensorToMesh::map)
        .def("config", &TensorToMesh::config);

    auto py_replicate_tensor_to_mesh =
        static_cast<py::class_<ReplicateTensorToMesh, std::unique_ptr<ReplicateTensorToMesh>>>(
            module.attr("ReplicateTensorToMesh"));

    py_replicate_tensor_to_mesh
        .def(
            py::init([](MeshDevice& mesh_device) -> std::unique_ptr<ReplicateTensorToMesh> {
                return std::make_unique<ReplicateTensorToMesh>(ReplicateTensorToMesh(mesh_device.num_devices()));
            }),
            py::kw_only(),
            py::arg("mesh_device"))
        .def(
            py::init([](size_t num_devices) -> std::unique_ptr<ReplicateTensorToMesh> {
                return std::make_unique<ReplicateTensorToMesh>(ReplicateTensorToMesh(num_devices));
            }),
            py::kw_only(),
            py::arg("num_devices"))
        .def(
            "map",
            [](const ReplicateTensorToMesh& self, const Tensor& tensor) { return self.map(tensor); },
            py::arg("tensor"))
        .def("config", &ReplicateTensorToMesh::config);

    auto py_shard_tensor_to_mesh = static_cast<py::class_<ShardTensorToMesh, std::unique_ptr<ShardTensorToMesh>>>(
        module.attr("ShardTensorToMesh"));
    py_shard_tensor_to_mesh
        .def(
            py::init([](MeshDevice& mesh_device, int dim) -> std::unique_ptr<ShardTensorToMesh> {
                return std::make_unique<ShardTensorToMesh>(ShardTensorToMesh(mesh_device, dim));
            }),
            py::kw_only(),
            py::arg("mesh_device"),
            py::arg("dim"))
        .def(
            py::init([](size_t num_devices, int dim) -> std::unique_ptr<ShardTensorToMesh> {
                return std::make_unique<ShardTensorToMesh>(ShardTensorToMesh(num_devices, dim));
            }),
            py::kw_only(),
            py::arg("num_devices"),
            py::arg("dim"))
        .def(
            "map",
            [](const ShardTensorToMesh& self, const Tensor& tensor) { return self.map(tensor); },
            py::arg("tensor"))
        .def("config", &ShardTensorToMesh::config);

    auto py_shard_tensor_to_2d_mesh = static_cast<py::class_<ShardTensor2dMesh, std::unique_ptr<ShardTensor2dMesh>>>(
        module.attr("ShardTensor2dMesh"));
    py_shard_tensor_to_2d_mesh
        .def(
            py::init(
                [](MeshDevice& mesh_device,
                   const MeshShape& mesh_shape,
                   const Shard2dConfig& config) -> std::unique_ptr<ShardTensor2dMesh> {
                    return std::make_unique<ShardTensor2dMesh>(ShardTensor2dMesh(mesh_device, mesh_shape, config));
                }),
            py::kw_only(),
            py::arg("mesh_device"),
            py::arg("mesh_shape"),
            py::arg("config"))
        .def(
            py::init(
                [](const MeshShape& mesh_shape, const Shard2dConfig& config) -> std::unique_ptr<ShardTensor2dMesh> {
                    return std::make_unique<ShardTensor2dMesh>(ShardTensor2dMesh(mesh_shape, config));
                }),
            py::kw_only(),
            py::arg("mesh_shape"),
            py::arg("config"))
        .def(
            "map",
            [](const ShardTensor2dMesh& self, const Tensor& tensor) { return self.map(tensor); },
            py::arg("tensor"))
        .def("config", &ShardTensor2dMesh::config);

    auto py_mesh_to_tensor = static_cast<py::class_<MeshToTensor, ConcreteMeshToTensor, std::unique_ptr<MeshToTensor>>>(
        module.attr("MeshToTensor"));
    py_mesh_to_tensor
        .def(py::init([]() -> std::unique_ptr<MeshToTensor> { return std::make_unique<ConcreteMeshToTensor>(); }))
        .def("compose", &MeshToTensor::compose);

    auto py_concat_mesh_to_tensor = static_cast<py::class_<ConcatMeshToTensor, std::unique_ptr<ConcatMeshToTensor>>>(
        module.attr("ConcatMeshToTensor"));
    py_concat_mesh_to_tensor
        .def(
            py::init([](int dim) -> std::unique_ptr<ConcatMeshToTensor> {
                return std::make_unique<ConcatMeshToTensor>(dim);
            }),
            py::kw_only(),
            py::arg("dim"))
        .def(
            "compose",
            [](const ConcatMeshToTensor& self, const std::vector<Tensor>& tensors) { return self.compose(tensors); },
            py::arg("tensors"));

    auto py_concat_2d_mesh_to_tensor =
        static_cast<py::class_<ConcatMesh2dToTensor, std::unique_ptr<ConcatMesh2dToTensor>>>(
            module.attr("ConcatMesh2dToTensor"));
    py_concat_2d_mesh_to_tensor
<<<<<<< HEAD
    .def(py::init<>(MeshDevice & mesh_device, const Concat2dConfig& config) {
        return concat_2d_mesh_to_tensor_composer(mesh_device, config);
    },
         py::kw_only(),
         py::arg("mesh_device"),
         py::arg("config"))
         .def("compose",[](self, const std::vector<Tensor>& tensors) {
            return self.compose(tensors);
         },
         .py::arg("tensors"));
>>>>>>> expose classes to python
=======
        .def(
            py::init(
                [](MeshDevice& mesh_device, const Concat2dConfig& config) -> std::unique_ptr<ConcatMesh2dToTensor> {
                    TT_FATAL(
                        config.row_dim != config.col_dim,
                        "Dimensions in 'dims' must be different; got row_dim: {}, col_dim: {}",
                        config.row_dim,
                        config.col_dim);
                    return std::make_unique<ConcatMesh2dToTensor>(mesh_device, config);
                }),
            py::kw_only(),
            py::arg("mesh_device"),
            py::arg("config"))
        .def(
            "compose",
            [](ConcatMesh2dToTensor self, const std::vector<Tensor>& tensors) -> Tensor {
                return self.compose(tensors);
            },
            py::arg("tensors"));
>>>>>>> one type error left

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

    auto py_replicate_tensor_config = static_cast<py::class_<ReplicateTensor>>(module.attr("ShardTensor"));
    py_replicate_tensor_config.def(py::init<>())
        .def(py::init<int>(), py::arg("replication_factor") = 1)
        .def_readwrite("shard_dimension", &ReplicateTensor::replication_factor)
        .def("__eq__", [](const ReplicateTensor& a, const ReplicateTensor& b) {
            return a.replication_factor == b.replication_factor;
        });

    auto py_shard_tensor_config = static_cast<py::class_<ShardTensor>>(module.attr("ShardTensor"));
    py_shard_tensor_config.def(py::init<int>(), py::arg("shard_dimension"))
        .def_readwrite("shard_dimension", &ShardTensor::shard_dimension)
        .def("__eq__", [](const ShardTensor& a, const ShardTensor& b) { return a == b; });
    auto py_shard_mesh = static_cast<py::class_<ShardMesh>>(module.attr("ShardMesh"));
    py_shard_mesh.def(py::init<>()).def_readwrite("y", &ShardMesh::y).def_readwrite("x", &ShardMesh::x);
    auto py_shard_tensor2d = static_cast<py::class_<ShardTensor2D>>(module.attr("ShardTensor2d"));
    py_shard_tensor2d.def(py::init<ShardMesh>(), py::arg("mesh"))
        .def_readonly("shard_mesh", &ShardTensor2D::shard_mesh)
        .def("__eq__", [](const ShardTensor2D& a, const ShardTensor2D& b) { return a == b; });
    auto py_allgather_config =
        static_cast<py::class_<AllGatherTensor>>(module.attr("AllGatherTensor"))
            .def(py::init<>())
            .def("__eq__", [](const AllGatherTensor& a, const AllGatherTensor& b) { return a == b; });

    auto py_shard2d_config = static_cast<py::class_<Shard2dConfig>>(module.attr("Shard2dConfig"));
    py_shard2d_config.def(py::init<int, int>(), py::arg("row_dim"), py::arg("col_dim"))
        .def_readwrite("row_dim", &Shard2dConfig::row_dim)
        .def_readwrite("col_dim", &Shard2dConfig::col_dim);
    auto py_concat2d_config = static_cast<py::class_<Concat2dConfig>>(module.attr("Concat2dConfig"));
    py_concat2d_config.def(py::init<int, int>(), py::arg("row_dim"), py::arg("col_dim"))
        .def_readwrite("row_dim", &Concat2dConfig::row_dim)
        .def_readwrite("col_dim", &Concat2dConfig::col_dim);

    module.def(
        "get_distributed_tensor_config",
        &get_distributed_tensor_config,
        py::arg("metadata"),
        R"doc(
            Returns a distributed_tensor_config object given a valid metadata object of the type

            {
                "item": "field",
                "item": "field",
            }
        )doc");
    module.def(
        "get_shard2d_config",
        &get_shard2d_config,
        py::arg("metadata"),
        R"doc(
            Returns a Shard2dConfig object given a valid metadata object of the type
            {
                "row_dim": "field",
                "col_dim": "field",
            }
        )doc");
    module.def(
        "get_concat2d_config",
        &get_concat2d_config,
        py::arg("metadata"),
        R"doc(
            Returns a Concat2dConfig object given a valid metadata object of the type
            {
                "row_dim": "field",
                "col_dim": "field",
            }
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    // TODO: Add rdocs
=======
>>>>>>> move class definitions from from distributed_tensor.cpp to.hpp so they can be exposed to the pybind.cpp; add dummy void methods in .cpp to satisfy linker; add new constructors and factory methods to fix type errors
    module.def(
        "replicate_tensor_to_mesh_mapper",
        [](MeshDevice& mesh_device) -> std::unique_ptr<TensorToMesh> {
            return replicate_tensor_to_mesh_mapper(mesh_device);
        },
        py::arg("mesh_device"));
    module.def(
        "shard_tensor_to_mesh_mapper",
        [](MeshDevice& mesh_device, int dim) -> std::unique_ptr<TensorToMesh> {
            return shard_tensor_to_mesh_mapper(mesh_device, dim);
        },
        py::arg("mesh_device"),
        py::arg("dim"));
    module.def(
        "shard_tensor_to_2d_mesh_mapper",
        [](MeshDevice& mesh_device,
           const MeshShape& mesh_shape,
           const Shard2dConfig& config) -> std::unique_ptr<TensorToMesh> {
            return shard_tensor_to_2d_mesh_mapper(mesh_device, mesh_shape, config);
        },
        py::arg("mesh_device"),
        py::arg("mesh_shape"),
        py::arg("config"));
    module.def(
<<<<<<< HEAD
        "shard_tensor_to_2d_mesh_mapper",
        [](MeshDevice& mesh_device,
           const std::tuple<int, int> mesh_shape,
           const std::tuple<int, int> dims) -> std::unique_ptr<TensorToMesh> {
            return shard_tensor_to_2d_mesh_mapper(
                mesh_device,
                MeshShape(std::get<0>(mesh_shape), std::get<1>(mesh_shape)),
                Shard2dConfig{.row_dim = std::get<0>(dims), .col_dim = std::get<1>(dims)});
        },
        py::arg("mesh_device"),
        py::arg("mesh_shape"),
        py::arg("dims"),
        R"doc(
            Create a ShardTensor2dMesh mapper with the given mesh device, mesh shape, and dimensions.

            Args:
                mesh_device (MeshDevice): The mesh device to create the mapper for.
                mesh_shape (MeshShape): The shape of the 2D mesh as (num_rows, num_cols).
                dims (Tuple[int, int]): The dimensions to create the mapper for in (row, column) format.

            Returns:
                TensorToMesh: The created ShardTensor2dMesh mapper.
   )doc");
    module.def(
=======
>>>>>>> move class definitions from from distributed_tensor.cpp to.hpp so they can be exposed to the pybind.cpp; add dummy void methods in .cpp to satisfy linker; add new constructors and factory methods to fix type errors
        "concat_mesh_to_tensor_composer",
        [](int dim) -> std::unique_ptr<MeshToTensor> { return concat_mesh_to_tensor_composer(dim); },
        py::arg("dim"));
    module.def(
        "concat_2d_mesh_to_tensor_composer",
        [](MeshDevice& mesh_device, const Concat2dConfig& config) -> std::unique_ptr<MeshToTensor> {
            return concat_2d_mesh_to_tensor_composer(mesh_device, config);
        },
        py::arg("mesh_device"),
        py::arg("config"));
<<<<<<< HEAD
    module.def(
        "concat_2d_mesh_to_tensor_composer",
        [](MeshDevice& mesh_device,
           const std::tuple<int, int> mesh_shape,
           const std::tuple<int, int> dims) -> std::unique_ptr<MeshToTensor> {
            TT_FATAL(
                std::get<0>(mesh_shape) <= mesh_device.shape().num_rows &&  //
                    std::get<1>(mesh_shape) <= mesh_device.shape().num_cols,
                "Device mesh shape does not match the provided mesh shape.");
            return concat_2d_mesh_to_tensor_composer(
                mesh_device, Concat2dConfig{.row_dim = std::get<0>(dims), .col_dim = std::get<1>(dims)});
        },
        py::arg("mesh_device"),
        py::arg("dims"),
        py::arg("mesh_shape"),
        R"doc(
            Create a ConcatMesh2dToTensor composer with the given mesh device and dimensions.

            Args:
                mesh_device (MeshDevice): The mesh device to create the composer for.
                dims (Tuple[int, int]): The dimensions to create the composer for in (row, column) format.
                mesh_shape (Tuple[int, int]): The shape of the 2D mesh as (num_rows, num_cols).

            Returns:
                TensorToMesh: The created ConcatMesh2dToTensor composer.
   )doc");
    module.def(
        "distribute_tensor",
        [](const Tensor& tensor,
           const TensorToMesh& mapper,
           std::optional<std::reference_wrapper<MeshDevice>> mesh_device) -> Tensor {
            return distribute_tensor(from_device(tensor), mapper, mesh_device);
        },
        py::arg("tensor"),
        py::arg("mapper"),
        py::arg("mesh_device"));
    module.def(
        "aggregate_tensor",
        [](const Tensor& tensor, const MeshToTensor& composer) -> Tensor {
            return aggregate_tensor(from_device(tensor), composer);
        },
        py::arg("tensor"),
        py::arg("composer"));
    module.def(
        "aggregate_tensor",
        [](const std::vector<Tensor>& tensors, const MeshToTensor& composer) -> Tensor {
            return aggregate_tensor(from_device(aggregate_as_tensor(tensors, AllGatherTensor{})), composer);
        },
        py::arg("tensor"),
        py::arg("composer"));
=======
    //TODO: overload this method to enable selection of a subset of shards with a config or something before passing to aggregate 
>>>>>>> expose classes to python
=======
=======
>>>>>>> move class definitions from from distributed_tensor.cpp to.hpp so they can be exposed to the pybind.cpp; add dummy void methods in .cpp to satisfy linker; add new constructors and factory methods to fix type errors
    // TODO: overload this method to enable selection of a subset of shards with a config or something before passing to
    // aggregate
>>>>>>> one type error left
    module.def(
        "aggregate_as_tensor",
        [](const std::vector<Tensor>& tensors) -> Tensor { return aggregate_as_tensor(tensors, AllGatherTensor{}); },
        py::arg("tensors"),
        py::kw_only());
    module.def("get_t3k_physical_device_ids_ring", &get_t3k_physical_device_ids_ring);
    module.attr("DefaultMeshCommandQueueId") = ttnn::DefaultMeshCommandQueueId;
}

}  // namespace ttnn::distributed
