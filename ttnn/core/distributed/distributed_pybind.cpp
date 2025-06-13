// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/distributed_pybind.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <ostream>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/pytypes.h>

#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/types.hpp"

// This is required for automatic conversions, as in the creation of mesh devices
// https://github.com/tenstorrent/tt-metal/issues/18082
#include "pybind11/stl.h"
#include "ttnn/tensor/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::distributed {

namespace py = pybind11;

void py_module_types(py::module& module) {
    py::class_<MeshToTensor, std::unique_ptr<MeshToTensor>>(module, "CppMeshToTensor");
    py::class_<TensorToMesh, std::unique_ptr<TensorToMesh>>(module, "CppTensorToMesh");

    py::class_<MeshMapperConfig>(module, "MeshMapperConfig");
    py::class_<MeshComposerConfig>(module, "MeshComposerConfig");

    py::class_<MeshDevice, std::shared_ptr<MeshDevice>>(module, "MeshDevice");
    py::class_<MeshShape>(module, "MeshShape", "Shape of a mesh device.");
    py::class_<MeshCoordinate>(module, "MeshCoordinate", "Coordinate within a mesh device.");
    py::class_<MeshCoordinateRange>(module, "MeshCoordinateRange", "Range of coordinates within a mesh device.");
    py::class_<MeshCoordinateRangeSet>(
        module, "MeshCoordinateRangeSet", "Set of coordinate ranges within a mesh device.");
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
            py::init([](size_t x, size_t y, size_t z) { return MeshShape(x, y, z); }),
            "Constructor with the specified 3D shape.",
            py::arg("x"),
            py::arg("y"),
            py::arg("z"))
        .def(
            py::init([](const std::vector<uint32_t>& shape) { return MeshShape(shape); }),
            "Constructor with the specified ND shape.",
            py::arg("shape"))
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

    static_cast<py::class_<MeshCoordinateRange>>(module.attr("MeshCoordinateRange"))
        .def(
            py::init(
                [](const MeshCoordinate& start, const MeshCoordinate& end) { return MeshCoordinateRange(start, end); }),
            "Constructor with specified start and end coordinates.",
            py::arg("start"),
            py::arg("end"))
        .def(
            py::init([](const MeshShape& shape) { return MeshCoordinateRange(shape); }),
            "Constructor that spans the entire mesh.",
            py::arg("shape"))
        .def(
            "__repr__",
            [](const MeshCoordinateRange& mcr) {
                std::ostringstream str;
                str << mcr;
                return str.str();
            })
        .def(
            "__iter__",
            [](const MeshCoordinateRange& mcr) { return py::make_iterator(mcr.begin(), mcr.end()); },
            py::keep_alive<0, 1>());

    static_cast<py::class_<MeshCoordinateRangeSet>>(module.attr("MeshCoordinateRangeSet"))
        .def(
            py::init([]() { return MeshCoordinateRangeSet(); }),
            "Default constructor for an empty MeshCoordinateRangeSet.")
        .def(
            py::init([](const MeshCoordinateRange& range) { return MeshCoordinateRangeSet(range); }),
            "Constructor with specified range.",
            py::arg("range"))
        .def("merge", &MeshCoordinateRangeSet::merge, py::arg("range"))
        .def("__repr__", [](const MeshCoordinateRangeSet& mcrs) {
            std::ostringstream str;
            str << mcrs;
            return str.str();
        });

    auto py_mesh_device = static_cast<py::class_<MeshDevice, std::shared_ptr<MeshDevice>>>(module.attr("MeshDevice"));
    py_mesh_device
        .def(
            py::init([](const MeshShape& mesh_shape,
                        size_t l1_small_size,
                        size_t trace_region_size,
                        size_t num_command_queues,
                        const DispatchCoreConfig& dispatch_core_config,
                        const std::optional<MeshCoordinate>& offset,
                        const std::vector<chip_id_t>& physical_device_ids,
                        size_t worker_l1_size) {
                return MeshDevice::create(
                    MeshDeviceConfig(mesh_shape, offset, physical_device_ids),
                    l1_small_size,
                    trace_region_size,
                    num_command_queues,
                    dispatch_core_config,
                    {},
                    worker_l1_size);
            }),
            py::kw_only(),
            py::arg("mesh_shape"),
            py::arg("l1_small_size"),
            py::arg("trace_region_size"),
            py::arg("num_command_queues"),
            py::arg("dispatch_core_config"),
            py::arg("offset"),
            py::arg("physical_device_ids"),
            py::arg("worker_l1_size") = DEFAULT_WORKER_L1_SIZE)
        .def("get_num_devices", &MeshDevice::num_devices)
        .def("id", &MeshDevice::id)
        .def("get_device_ids", &MeshDevice::get_device_ids)
        .def(
            "get_device_id",
            [](MeshDevice& self, const MeshCoordinate& coord) {
                auto device = self.get_device(coord);
                TT_FATAL(device, "Device ID requested for MeshCoord {} not found.", coord);
                return device->id();
            })
        .def(
            "create_submesh",
            &MeshDevice::create_submesh,
            py::arg("submesh_shape"),
            py::arg("offset") = std::nullopt,
            py::keep_alive<1, 0>())  // Keep MeshDevice alive as long as SubmeshDevice is alive
        .def(
            "create_submeshes",
            &MeshDevice::create_submeshes,
            py::arg("submesh_shape"),
            py::keep_alive<1, 0>())  // Keep MeshDevice alive as long as SubmeshDevices are alive
        .def(
            "get_submeshes",
            &MeshDevice::get_submeshes,
            R"doc(
              Get the submeshes created on this MeshDevice.

                Returns:
                    List[MeshDevice]: The submeshes created on this MeshDevice.
        )doc")
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
        .def(
            "set_program_cache_misses_allowed",
            &MeshDevice::set_program_cache_misses_allowed,
            R"doc(
               Set whether program cache misses are allowed across all devices in the mesh.
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
                   SubDeviceManagerId: The ID of the created sub-device manager.
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
                   SubDeviceManagerId: The ID of the created sub-device manager.
                   SubDeviceId: The ID of the sub-device that will be used for fabric.
           )doc")
        .def(
            "load_sub_device_manager",
            &MeshDevice::load_sub_device_manager,
            py::arg("sub_device_manager_id"),
            R"doc(
               Loads the sub-device manager with the given ID.


               Args:
                   sub_device_manager_id (SubDeviceManagerId): The ID of the sub-device manager to load.
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
            py::arg("sub_device_manager_id"),
            R"doc(
               Removes the sub-device manager with the given ID.


               Args:
                   sub_device_manager_id (SubDeviceManagerId): The ID of the sub-device manager to remove.
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
           )doc")
        .def(
            "num_program_cache_entries",
            &MeshDevice::num_program_cache_entries,
            "Number of entries in the program cache for this device")
        .def(
            "sfpu_eps",
            [](MeshDevice* device) { return tt::tt_metal::hal::get_eps(); },
            R"doc(Returns machine epsilon value for current architecture.)doc")
        .def(
            "sfpu_nan",
            [](MeshDevice* device) { return tt::tt_metal::hal::get_nan(); },
            R"doc(Returns NaN value for current architecture.)doc")
        .def(
            "sfpu_inf",
            [](MeshDevice* device) { return tt::tt_metal::hal::get_inf(); },
            R"doc(Returns Infinity value for current architecture.)doc");

    auto py_tensor_to_mesh =
        static_cast<py::class_<TensorToMesh, std::unique_ptr<TensorToMesh>>>(module.attr("CppTensorToMesh"));

    auto py_mesh_to_tensor =
        static_cast<py::class_<MeshToTensor, std::unique_ptr<MeshToTensor>>>(module.attr("CppMeshToTensor"));

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
        py::arg("dispatch_core_config"),
        py::arg("worker_l1_size") = DEFAULT_WORKER_L1_SIZE);
    module.def("close_mesh_device", &close_mesh_device, py::arg("mesh_device"), py::kw_only());

    // TODO: #22258 - make this more flexible and useful.
    auto py_mesh_mapper_config = static_cast<py::class_<MeshMapperConfig>>(module.attr("MeshMapperConfig"));
    py_mesh_mapper_config.def(
        py::init([](size_t row_dim, size_t col_dim) {
            MeshMapperConfig config;
            config.placements.push_back(MeshMapperConfig::Shard{row_dim});
            config.placements.push_back(MeshMapperConfig::Shard{col_dim});
            return config;
        }),
        py::arg("row_dim"),
        py::arg("col_dim"));
    auto py_mesh_composer_config = static_cast<py::class_<MeshComposerConfig>>(module.attr("MeshComposerConfig"));
    py_mesh_composer_config.def(py::init([](size_t row_dim, size_t col_dim) {
        MeshComposerConfig config;
        config.dims.push_back(row_dim);
        config.dims.push_back(col_dim);
        return config;
    }));

    module.def(
        "get_device_tensors",
        &get_device_tensors,
        py::arg("tensor"),
        py::kw_only(),
        R"doc(
       Get a list of tensor shards from a multidevice tensor.

       Args:
           tensor (Tensor): The tensor to get the shards from.

       Returns:
           List[Tensor]: The shards of the tensor corresponding to the devices.
           )doc");
    module.def(
        "replicate_tensor_to_mesh_mapper",
        [](MeshDevice& mesh_device) -> std::unique_ptr<TensorToMesh> {
            return replicate_tensor_to_mesh_mapper(mesh_device);
        },
        py::arg("mesh_device"),
        R"doc(
       Returns a mapper replicating a tensor over the given mesh.

       Args:
           mesh_device (MeshDevice): The mesh to replicate over.


       Returns:
           TensorToMesh: A mapper providing the desired sharding.
   )doc");
    module.def(
        "shard_tensor_to_mesh_mapper",
        [](MeshDevice& mesh_device, int dim) -> std::unique_ptr<TensorToMesh> {
            return shard_tensor_to_mesh_mapper(mesh_device, dim);
        },
        py::arg("mesh_device"),
        py::arg("dim"),
        R"doc(
       Returns a mapper sharding a tensor over the given dimension for the given mesh.

       Args:
           mesh_device (MeshDevice): The mesh to replicate over.
           dim (int): The dimension to shard over.

       Returns:
           TensorToMesh: A mapper providing the desired sharding.
   )doc");
    module.def(
        "create_mesh_mapper",
        [](MeshDevice& mesh_device,
           const MeshMapperConfig& config,
           const std::optional<MeshShape>& mesh_shape = std::nullopt) -> std::unique_ptr<TensorToMesh> {
            return create_mesh_mapper(mesh_device, config, mesh_shape);
        },
        py::arg("mesh_device"),
        py::arg("config"),
        py::arg("mesh_shape") = std::nullopt,
        R"doc(
       Returns an ND mapper for sharding and replicating a tensor over the given dimensions for the given mesh.

       Args:
           mesh_device (MeshDevice): The mesh to create the mapper for.
           config (MeshMapperConfig): A config object representing a set of placements.
           mesh_shape (MeshShape): If provided, provides overrides the logical shape for the mapper.
            Useful for distributing a tensor over ND shape that exceeds the number of physical dimensions
            in the mesh device.

       Returns:
           TensorToMesh: A mapper providing the desired sharding.
   )doc");
    module.def(
        "concat_mesh_to_tensor_composer",
        [](MeshDevice& mesh_device, int dim) -> std::unique_ptr<MeshToTensor> {
            return concat_mesh_to_tensor_composer(mesh_device, dim);
        },
        py::arg("mesh_device"),
        py::arg("dim"));
    module.def(
        "create_mesh_composer",
        [](MeshDevice& mesh_device, const MeshComposerConfig& config, const std::optional<MeshShape>& shape)
            -> std::unique_ptr<MeshToTensor> { return create_mesh_composer(mesh_device, config, shape); },
        py::arg("mesh_device"),
        py::arg("config"),
        py::arg("shape") = std::nullopt,
        R"doc(
            Returns an ND composer that concatenates a tensor across the given dimensions.

            Args:
                mesh_device (MeshDevice): The mesh device to create the composer for.
                config (MeshComposerConfig): A config object representing the dimensions to concat over.
                shape (MeshShape): If provided, overrides the logical shape of the mesh.

            Returns:
                TensorToMesh: A composer providing the desired concatenation.
   )doc");
    module.def(
        "distribute_tensor",
        [](const Tensor& tensor,
           const TensorToMesh& mapper,
           std::optional<std::reference_wrapper<MeshDevice>> mesh_device = std::nullopt) -> Tensor {
            return distribute_tensor(tensor, mapper, mesh_device);
        },
        py::arg("tensor"),
        py::arg("mapper"),
        py::arg("mesh_device") = py::none(),
        R"doc(
            Distributes a tensor across a mesh device according to provided mapper.

            Args:
                tensor (Tensor): The tensor to distribute.
                mapper (TensorToMesh): The mapper to use for distribution.
                mesh_device (MeshDevice): The mesh device to distribute the tensor over.

            Returns:
                Tensor: The distributed tensor.
        )doc");
    module.def(
        "aggregate_tensor",
        [](const Tensor& tensor, const MeshToTensor& composer) -> Tensor { return aggregate_tensor(tensor, composer); },
        py::arg("tensor"),
        py::arg("composer"),
        R"doc(
            Aggregates a set of shard tensors into a single host tensor using the provided composer.

            Args:
                tensor (Tensor): The tensor to aggregate.
                composer (MeshToTensor): The composer to use for aggregation.

            Returns:
                Tensor: The aggregated tensor.
            )doc");
    module.def(
        "aggregate_as_tensor",
        [](const std::vector<Tensor>& tensors) -> Tensor { return aggregate_as_tensor(tensors, AllGatherTensor{}); },
        py::arg("tensors"),
        py::kw_only(),
        R"doc(
            Aggregates a set of shards into one tensor. Device shards will remain on device and be packed into a multidevice storage object.

            Args:
                tensor (Tensor): The tensor to aggregate.

            Returns:
                Tensor: The aggregated tensor.
            )doc");
    module.def(
        "combine_device_tensors",
        [](const std::vector<Tensor>& tensors) -> Tensor { return combine_device_tensors(tensors); },
        py::arg("tensors"),
        py::kw_only(),
        R"doc(
            Combines tensor shards allocated on individual devices into a single multi-device tensor. All tensors shards must be allocated on the same mesh buffer.

            Args:
                tensors (List[Tensor]): The tensor shards to combine.

            Returns:
                Tensor: The combined tensor.
            )doc");
    module.def("get_t3k_physical_device_ids_ring", &get_t3k_physical_device_ids_ring);
}

}  // namespace ttnn::distributed
