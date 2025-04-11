// SPDX-FileConbrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/distributed_nanobind.hpp"

#include <cstddef>
#include <memory>
#include <ostream>

#include <nanobind/nanobind.h>
#include <nanobind/make_iterator.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "distributed_tensor.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/types.hpp"

// This is required for automatic conversions, as in the creation of mesh devices
// https://github.com/tenstorrent/tt-metal/issues/18082
#include "ttnn/tensor/types.hpp"

// note from nanobind docs:
// We strongly recommend that you replace all use of std::unique_ptr<T> by
// std::unique_ptr<T, nb::deleter<T>> in your code. Without the latter type
// declaration, which references a custom nanobind-provided deleter nb::deleter<T>,
// nanobind cannot transfer ownership of objects constructed using nb::init<...>
// to C++ and will refuse to do so with an error message. Further detail on this
// special case can be found in the advanced section on object ownership.

using namespace tt::tt_metal;

namespace ttnn::distributed {

namespace nb = nanobind;

void nb_module_types(nb::module_& mod) {
    nb::class_<MeshToTensor>(mod, "CppMeshToTensor");
    nb::class_<TensorToMesh>(mod, "CppTensorToMesh");

    nb::class_<Shard2dConfig>(mod, "Shard2dConfig");
    nb::class_<Concat2dConfig>(mod, "Concat2dConfig");

    nb::class_<MeshDevice, IDevice>(mod, "MeshDevice"); //shared_ptr
    nb::class_<MeshSubDeviceManagerId>(mod, "MeshSubDeviceManagerId");
    nb::class_<MeshShape, ShapeBase>(mod, "MeshShape", "Shape of a mesh device.");
    nb::class_<MeshCoordinate>(mod, "MeshCoordinate", "Coordinate within a mesh device.");
    nb::class_<MeshCoordinateRange>(mod, "MeshCoordinateRange", "Range of coordinates within a mesh device.");
    nb::class_<MeshCoordinateRangeSet>(
        mod, "MeshCoordinateRangeSet", "Set of coordinate ranges within a mesh device.");
}

void nb_module(nb::module_& mod) {
    // TODO: #17477 - Remove overloads that accept 'row' and 'col'. Instead, use generic ND terms.

    static_cast<nb::class_<MeshShape>>(mod.attr("MeshShape"))
        .def(
            nb::init<MeshShape::coord_type, MeshShape::coord_type>(),
            "Constructor with the specified number of rows and columns.",
            nb::arg("num_rows"),
            nb::arg("num_cols"))
        .def(
            nb::init<MeshShape::coord_type, MeshShape::coord_type, MeshShape::coord_type>(),
            "Constructor with the specified 3D shape.",
            nb::arg("x"),
            nb::arg("y"),
            nb::arg("z"))
        .def(
            nb::init<const std::vector<uint32_t>&>(), // tt::stl::Span<const uint32_t>
            "Constructor with the specified ND shape.",
            nb::arg("shape"))
        .def(
            "__repr__",
            [](const MeshShape& ms) {
                std::ostringstream str;
                str << ms;
                return str.str();
            })
        .def(
            "__iter__",
            [](const MeshShape& ms) {
                return nb::make_iterator(
                    nb::type<decltype(ms.view().begin())>(),
                    "iterator",
                    ms.view().begin(),
                    ms.view().end());
            },
            nb::keep_alive<0, 1>());

    static_cast<nb::class_<MeshCoordinate>>(mod.attr("MeshCoordinate"))
        .def(
            nb::init<MeshCoordinate::coord_type, MeshCoordinate::coord_type>(),
            "Constructor with specified row and column offsets.",
            nb::arg("row"),
            nb::arg("col"))
        .def(
            nb::init<MeshCoordinate::coord_type, MeshCoordinate::coord_type, MeshCoordinate::coord_type>(),
            "Constructor with the specified 3D coordinate.",
            nb::arg("x"),
            nb::arg("y"),
            nb::arg("z"))
        .def(
            nb::init<const std::vector<uint32_t>&>(), // tt::stl::Span<const uint32_t>
            "Constructor with the specified ND coordinate.",
            nb::arg("coords"))
        .def(
            "__repr__",
            [](const MeshCoordinate& mc) {
                std::ostringstream str;
                str << mc;
                return str.str();
            })
        .def(
            "__iter__",
            [](const MeshCoordinate& mc) {
                return nb::make_iterator(
                        nb::type<decltype(mc.coords().begin())>(),
                        "iterator",
                        mc.coords().begin(),
                        mc.coords().end());
            },
            nb::keep_alive<0, 1>());

    static_cast<nb::class_<MeshCoordinateRange>>(mod.attr("MeshCoordinateRange"))
        .def(
            nb::init<const MeshCoordinate&, const MeshCoordinate&>(),
            "Constructor with specified start and end coordinates.",
            nb::arg("start"),
            nb::arg("end"))
        .def(
            nb::init<const MeshShape&>(),
            "Constructor that spans the entire mesh.",
            nb::arg("shape"))
        .def(
            "__repr__",
            [](const MeshCoordinateRange& mcr) {
                std::ostringstream str;
                str << mcr;
                return str.str();
            })
        .def(
            "__iter__",
            [](const MeshCoordinateRange& mcr) {
                return nb::make_iterator(
                        nb::type<decltype(mcr.begin())>(),
                        "iterator",
                        mcr.begin(), mcr.end());
            },
            nb::keep_alive<0, 1>());

    static_cast<nb::class_<MeshCoordinateRangeSet>>(mod.attr("MeshCoordinateRangeSet"))
        .def(
            nb::init<>(),
            "Default constructor for an empty MeshCoordinateRangeSet.")
        .def(
            nb::init<const MeshCoordinateRange&>(),
            "Constructor with specified range.",
            nb::arg("range"))
        .def("merge", &MeshCoordinateRangeSet::merge, nb::arg("range"))
        .def("__repr__", [](const MeshCoordinateRangeSet& mcrs) {
            std::ostringstream str;
            str << mcrs;
            return str.str();
        });

    auto nb_mesh_device = static_cast<nb::class_<MeshDevice>>(mod.attr("MeshDevice"));
    nb_mesh_device
        .def(
            nb::new_([](const MeshShape& mesh_shape,
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
            nb::kw_only(),
            nb::arg("mesh_shape"),
            nb::arg("l1_small_size"),
            nb::arg("trace_region_size"),
            nb::arg("num_command_queues"),
            nb::arg("dispatch_core_config"),
            nb::arg("offset"),
            nb::arg("physical_device_ids"),
            nb::arg("worker_l1_size") = DEFAULT_WORKER_L1_SIZE)
        .def("get_num_devices", &MeshDevice::num_devices)
        .def("id", &MeshDevice::id)
        .def("get_device_ids", &MeshDevice::get_device_ids)
        .def(
            "get_device",
            nb::overload_cast<chip_id_t>(&MeshDevice::get_device, nb::const_),
            nb::rv_policy::reference)
        .def(
            "get_device",
            nb::overload_cast<size_t, size_t>(&MeshDevice::get_device, nb::const_),
            nb::rv_policy::reference)
        .def(
            "get_devices",
            &MeshDevice::get_devices,
            nb::rv_policy::reference,
            R"doc(
           Get the devices in the device mesh.


           Returns:
               List[Device]: The devices in the device mesh.
       )doc")
        .def(
            "create_submesh",
            &MeshDevice::create_submesh,
            nb::arg("submesh_shape"),
            nb::arg("offset") = std::nullopt,
            nb::keep_alive<1, 0>())  // Keep MeshDevice alive as long as SubmeshDevice is alive
        .def(
            "create_submeshes",
            &MeshDevice::create_submeshes,
            nb::arg("submesh_shape"),
            nb::keep_alive<1, 0>())  // Keep MeshDevice alive as long as SubmeshDevices are alive
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
        .def_prop_ro(
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
            nb::arg("new_shape"),
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
            nb::arg("sub_devices"),
            nb::arg("local_l1_size"),
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
            nb::arg("sub_devices"),
            nb::arg("local_l1_size"),
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
            nb::arg("sub_device_manager_id"),
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
            nb::arg("sub_device_manager_id"),
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
            nb::arg("sub_device_ids"),
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
        static_cast<nb::class_<TensorToMesh>>(mod.attr("CppTensorToMesh"));

    auto py_mesh_to_tensor =
        static_cast<nb::class_<MeshToTensor>>(mod.attr("CppMeshToTensor"));

    mod.def(
        "open_mesh_device",
        &open_mesh_device,
        nb::kw_only(),
        nb::arg("mesh_shape"),
        nb::arg("l1_small_size"),
        nb::arg("trace_region_size"),
        nb::arg("num_command_queues"),
        nb::arg("offset"),
        nb::arg("physical_device_ids"),
        nb::arg("dispatch_core_config"),
        nb::arg("worker_l1_size") = DEFAULT_WORKER_L1_SIZE);
    mod.def(
        "close_mesh_device",
        &close_mesh_device,
        nb::kw_only(),
        nb::arg("mesh_device"));

    auto py_shard2d_config = static_cast<nb::class_<Shard2dConfig>>(mod.attr("Shard2dConfig"));
    py_shard2d_config.def(nb::init<int, int>(), nb::arg("row_dim"), nb::arg("col_dim"))
        .def_rw("row_dim", &Shard2dConfig::row_dim)
        .def_rw("col_dim", &Shard2dConfig::col_dim);
    auto py_concat2d_config = static_cast<nb::class_<Concat2dConfig>>(mod.attr("Concat2dConfig"));
    py_concat2d_config.def(nb::init<int, int>(), nb::arg("row_dim"), nb::arg("col_dim"))
        .def_rw("row_dim", &Concat2dConfig::row_dim)
        .def_rw("col_dim", &Concat2dConfig::col_dim);


    mod.def(
        "get_device_tensors",
        &get_device_tensors,
        nb::kw_only(),
        nb::arg("tensor"),
        R"doc(
       Get a list of tensor shards from a multidevice tensor.

       Args:
           tensor (Tensor): The tensor to get the shards from.

       Returns:
           List[Tensor]: The shards of the tensor corresponding to the devices.
           )doc");
    mod.def(
        "replicate_tensor_to_mesh_mapper",
        [](MeshDevice& mesh_device) -> std::unique_ptr<TensorToMesh> {
            return replicate_tensor_to_mesh_mapper(mesh_device);
        },
        nb::arg("mesh_device"),
        R"doc(
       Returns a mapper replicating a tensor over the given mesh.

       Args:
           mesh_device (MeshDevice): The mesh to replicate over.


       Returns:
           TensorToMesh: A mapper providing the desired sharding.
   )doc");
    mod.def(
        "shard_tensor_to_mesh_mapper",
        [](MeshDevice& mesh_device, int dim) -> std::unique_ptr<TensorToMesh> {
            return shard_tensor_to_mesh_mapper(mesh_device, dim);
        },
        nb::arg("mesh_device"),
        nb::arg("dim"),
        R"doc(
       Returns a mapper sharding a tensor over the given dimension for the given mesh.

       Args:
           mesh_device (MeshDevice): The mesh to replicate over.
           dim (int): The dimension to shard over.

       Returns:
           TensorToMesh: A mapper providing the desired sharding.
   )doc");
    mod.def(
        "shard_tensor_to_2d_mesh_mapper",
        [](MeshDevice& mesh_device,
           const MeshShape& mesh_shape,
           const Shard2dConfig& config) -> std::unique_ptr<TensorToMesh> {
            return shard_tensor_to_2d_mesh_mapper(mesh_device, mesh_shape, config);
        },
        nb::arg("mesh_device"),
        nb::arg("mesh_shape"),
        nb::arg("config"),
        R"doc(
       Returns a mapper sharding a tensor over the given dimension for the given mesh.

       Args:
           mesh_device (MeshDevice): The mesh to create the mapper for.
           mesh_shape (MeshShape): The shape of the 2D mesh as (num_rows, num_cols).
           config (Shard2dConfig): A config object representing the row and column to shard over.

       Returns:
           TensorToMesh: A mapper providing the desired sharding.
   )doc");
    mod.def(
        "concat_mesh_to_tensor_composer",
        [](int dim) -> std::unique_ptr<MeshToTensor> { return concat_mesh_to_tensor_composer(dim); },
        nb::arg("dim"));
    mod.def(
        "concat_2d_mesh_to_tensor_composer",
        [](MeshDevice& mesh_device, const Concat2dConfig& config) -> std::unique_ptr<MeshToTensor> {
            return concat_2d_mesh_to_tensor_composer(mesh_device, config);
        },
        nb::arg("mesh_device"),
        nb::arg("config"),
        R"doc(
            Returns a Concat2dMeshToTensor composer with the given mesh device and dimensions.

            Args:
                mesh_device (MeshDevice): The mesh device to create the composer for.
                config (Concat2dConfig): A config object representing the row and column to concat over.

            Returns:
                TensorToMesh: A composer providing the desired concatenation.
   )doc");
    mod.def(
        "distribute_tensor",
        [](const Tensor& tensor,
           const TensorToMesh& mapper,
           std::optional<std::reference_wrapper<MeshDevice>> mesh_device = std::nullopt) -> Tensor {
            return distribute_tensor(tensor, mapper, mesh_device);
        },
        nb::arg("tensor"),
        nb::arg("mapper"),
        nb::arg("mesh_device") = nb::none(),
        R"doc(
            Distributes a tensor across a mesh device according to provided mapper.

            Args:
                tensor (Tensor): The tensor to distribute.
                mapper (TensorToMesh): The mapper to use for distribution.
                mesh_device (MeshDevice): The mesh device to distribute the tensor over.

            Returns:
                Tensor: The distributed tensor.
        )doc");
    mod.def(
        "aggregate_tensor",
        [](const Tensor& tensor, const MeshToTensor& composer) -> Tensor { return aggregate_tensor(tensor, composer); },
        nb::arg("tensor"),
        nb::arg("composer"),
        R"doc(
            Aggregates a set of shard tensors into a single host tensor using the provided composer.

            Args:
                tensor (Tensor): The tensor to aggregate.
                composer (MeshToTensor): The composer to use for aggregation.

            Returns:
                Tensor: The aggregated tensor.
            )doc");
    mod.def(
        "aggregate_tensor",
        [](const std::vector<Tensor>& tensors, const MeshToTensor& composer) -> Tensor {
            Tensor aggregated_tensor = aggregate_as_tensor(tensors, AllGatherTensor{});

            return aggregate_tensor(aggregated_tensor, composer);
        },
        nb::arg("tensor"),
        nb::arg("composer"),
        R"doc(
            Aggregates a set of shard tensors into a single host tensor using the provided composer.

            Args:
                tensor (Tensor): The tensor to aggregate.
                composer (MeshToTensor): The composer to use for aggregation.

            Returns:
                Tensor: The aggregated tensor.
            )doc");
    mod.def(
        "aggregate_as_tensor",
        [](const std::vector<Tensor>& tensors) -> Tensor { return aggregate_as_tensor(tensors, AllGatherTensor{}); },
        nb::kw_only(),
        nb::arg("tensors"),
        R"doc(
            Aggregates a set of shards into one tensor. Device shards will remain on device and be packed into a multidevice storage object.

            Args:
                tensor (Tensor): The tensor to aggregate.

            Returns:
                Tensor: The aggregated tensor.
            )doc");
    mod.def("get_t3k_physical_device_ids_ring", &get_t3k_physical_device_ids_ring);
}

}  // namespace ttnn::distributed
