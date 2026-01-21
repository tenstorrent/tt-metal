// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/distributed_nanobind.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/make_iterator.h>
#include <nanobind/operators.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/nanobind_helpers.hpp"
#include "ttnn-nanobind/small_vector_caster.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/maybe_remote.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/distributed/tensor_topology.hpp"
#include "distribution_mode.hpp"

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

class SystemMeshDescriptor {
private:
    MeshShape global_shape_;
    MeshShape local_shape_;
    tt::tt_metal::distributed::MeshContainer<tt::tt_metal::distributed::MaybeRemote<int>> device_ids_;

public:
    SystemMeshDescriptor() :
        global_shape_(tt::tt_metal::distributed::SystemMesh::instance().shape()),
        local_shape_(tt::tt_metal::distributed::SystemMesh::instance().local_shape()),
        device_ids_(
            global_shape_,
            tt::tt_metal::distributed::SystemMesh::instance().get_mapped_devices(global_shape_).device_ids) {}

    const MeshShape& shape() const { return global_shape_; }
    const MeshShape& local_shape() const { return local_shape_; }

    int get_device_id(const MeshCoordinate& coord) const {
        TT_FATAL(device_ids_.at(coord).is_local(), "Device at {} is remote.", coord);
        return device_ids_.at(coord).value();
    }

    bool is_local(const MeshCoordinate& coord) const { return device_ids_.at(coord).is_local(); }

    bool all_local() const { return global_shape_ == local_shape_; }
};

// NOLINTBEGIN(bugprone-unused-raii)
// NOLINTBEGIN(misc-redundant-expression)
void py_module_types(nb::module_& mod) {
    using namespace tt::tt_metal::distributed::multihost;

    // Bind strong types for Rank and Size
    nb::class_<Rank>(mod, "Rank", "Rank of a process in the distributed context")
        .def(nb::init<int>())
        .def("__int__", [](const Rank& r) { return *r; })
        .def("__repr__", [](const Rank& r) { return nb::str("Rank({})").format(*r); })
        .def("__str__", [](const Rank& r) { return nb::str("{}").format(*r); })
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def(nb::self < nb::self)
        .def(nb::self <= nb::self)
        .def(nb::self > nb::self)
        .def(nb::self >= nb::self);

    nb::class_<Size>(mod, "Size", "Size (number of processes) in the distributed context")
        .def(nb::init<int>())
        .def("__int__", [](const Size& s) { return *s; })
        .def("__repr__", [](const Size& s) { return nb::str("Size({})").format(*s); })
        .def("__str__", [](const Size& s) { return nb::str("{}").format(*s); })
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def(nb::self < nb::self)
        .def(nb::self <= nb::self)
        .def(nb::self > nb::self)
        .def(nb::self >= nb::self);

    nb::class_<MeshToTensor>(mod, "CppMeshToTensor");
    nb::class_<TensorToMesh>(mod, "CppTensorToMesh");

    nb::class_<MeshMapperConfig>(mod, "MeshMapperConfig");
    nb::class_<MeshComposerConfig>(mod, "MeshComposerConfig");
    nb::class_<MeshMapperConfig::Replicate>(mod, "PlacementReplicate");
    nb::class_<MeshMapperConfig::Shard>(mod, "PlacementShard");

    nb::class_<MeshDevice>(mod, "MeshDevice");
    nb::class_<MeshDeviceView>(mod, "MeshDeviceView");
    nb::class_<MeshShape>(mod, "MeshShape", "Shape of a mesh device.");
    nb::class_<MeshCoordinate>(mod, "MeshCoordinate", "Coordinate within a mesh device.");
    nb::class_<MeshCoordinateRange>(mod, "MeshCoordinateRange", "Range of coordinates within a mesh device.");
    nb::class_<MeshCoordinateRangeSet>(mod, "MeshCoordinateRangeSet", "Set of coordinate ranges within a mesh device.");
    nb::class_<SystemMeshDescriptor>(mod, "SystemMeshDescriptor");
    nb::class_<DistributedHostBuffer>(mod, "DistributedHostBuffer");
    nb::class_<TensorTopology>(mod, "TensorTopology");
}
// NOLINTEND(misc-redundant-expression)
// NOLINTEND(bugprone-unused-raii)

void py_module(nb::module_& mod) {
    static_cast<nb::class_<MeshShape>>(mod.attr("MeshShape"))
        .def(
            nb::init<uint32_t, uint32_t>(),  // uint32, not size_t!
            "Constructor with the specified 2D shape. The value s0 is assumed to be the outer dimension.",
            nb::arg("s0"),
            nb::arg("s1"))
        .def(
            nb::init<uint32_t, uint32_t, uint32_t>(),
            "Constructor with the specified 3D shape. The values s0...s2 are assumed to be supplied in row-major order "
            "(from outer dim to inner dim).",
            nb::arg("s0"),
            nb::arg("s1"),
            nb::arg("s2"))
        .def(
            nb::init<const std::vector<uint32_t>&>(),
            "Constructor with the specified ND shape. The values s0...sn are assumed to be supplied in row-major order "
            "(from outer dim to inner dim).",
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
                return nb::make_iterator<nb::rv_policy::reference_internal>(
                    nb::type<MeshShape>(), "iterator", ms.view().begin(), ms.view().end());
            },
            nb::keep_alive<0, 1>())
        .def(
            "__getitem__", [](const MeshShape& ms, int index) { return ms[index]; }, nb::arg("index"))
        .def("dims", &MeshShape::dims)
        .def("mesh_size", &MeshShape::mesh_size)
        .def("__eq__", [](const MeshShape& lhs, const MeshShape& rhs) { return lhs == rhs; })
        .def("__ne__", [](const MeshShape& lhs, const MeshShape& rhs) { return lhs != rhs; });

    static_cast<nb::class_<MeshCoordinate>>(mod.attr("MeshCoordinate"))
        .def(
            nb::init<uint32_t, uint32_t>(),
            "Constructor with the specified 2D coordinate. The value c0 is assumed to be the outer dimension.",
            nb::arg("c0"),
            nb::arg("c1"))
        .def(
            nb::init<uint32_t, uint32_t, uint32_t>(),
            "Constructor with the specified 3D coordinate. The values c0...c2 are assumed to be supplied in row-major "
            "order (from outer dim to inner dim).",
            nb::arg("c0"),
            nb::arg("c1"),
            nb::arg("c2"))
        .def(
            nb::init<const std::vector<uint32_t>&>(),
            "Constructor with the specified ND coordinate. The values c0...cn are assumed to be supplied in row-major "
            "order (from outer dim to inner dim).",
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
                return nb::make_iterator<nb::rv_policy::reference_internal>(
                    nb::type<MeshCoordinate>(), "iterator", mc.coords().begin(), mc.coords().end());
            },
            nb::keep_alive<0, 1>())
        .def(
            "__getitem__", [](const MeshCoordinate& mc, int index) { return mc[index]; }, nb::arg("index"))
        .def("dims", &MeshCoordinate::dims)
        .def("__hash__", [](const MeshCoordinate& mc) { return std::hash<MeshCoordinate>{}(mc); })
        .def("__eq__", [](const MeshCoordinate& a, const MeshCoordinate& b) { return a == b; });

    static_cast<nb::class_<MeshCoordinateRange>>(mod.attr("MeshCoordinateRange"))
        .def(
            nb::init<const MeshCoordinate&, const MeshCoordinate&>(),
            "Constructor with specified start and end coordinates.",
            nb::arg("start"),
            nb::arg("end"))
        .def(nb::init<const MeshShape&>(), "Constructor that spans the entire mesh.", nb::arg("shape"))
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
                return nb::make_iterator<nb::rv_policy::reference_internal>(
                    nb::type<MeshCoordinateRange>(), "iterator", mcr.begin(), mcr.end());
            },
            nb::keep_alive<0, 1>());

    static_cast<nb::class_<MeshCoordinateRangeSet>>(mod.attr("MeshCoordinateRangeSet"))
        .def(nb::init<>(), "Default constructor for an empty MeshCoordinateRangeSet.")
        .def(nb::init<const MeshCoordinateRange&>(), "Constructor with specified range.", nb::arg("range"))
        .def("merge", &MeshCoordinateRangeSet::merge, nb::arg("range"))
        .def("__repr__", [](const MeshCoordinateRangeSet& mcrs) {
            std::ostringstream str;
            str << mcrs;
            return str.str();
        });

    static_cast<nb::class_<SystemMeshDescriptor>>(mod.attr("SystemMeshDescriptor"))
        .def(nb::init<>())
        .def("shape", &SystemMeshDescriptor::shape)
        .def("local_shape", &SystemMeshDescriptor::local_shape)
        .def("get_device_id", &SystemMeshDescriptor::get_device_id)
        .def("is_local", &SystemMeshDescriptor::is_local)
        .def("all_local", &SystemMeshDescriptor::all_local);

    auto nb_mesh_device = static_cast<nb::class_<MeshDevice>>(mod.attr("MeshDevice"));
    nb_mesh_device.def("get_num_devices", &MeshDevice::num_devices)
        .def("id", &MeshDevice::id)
        .def("get_device_ids", &MeshDevice::get_device_ids)
        .def(
            "get_device_id",
            [](MeshDevice& self, const MeshCoordinate& coord) {
                auto* device = self.get_device(coord);
                TT_FATAL(device, "Device ID requested for MeshCoord {} not found.", coord);
                return device->id();
            })
        .def("get_fabric_node_id", &MeshDevice::get_fabric_node_id, nb::arg("coord"))
        .def(
            "create_submesh",
            &MeshDevice::create_submesh,
            nb::arg("submesh_shape"),
            nb::arg("offset") = nb::none(),
            nb::keep_alive<1, 0>())  // Keep MeshDevice alive as long as SubmeshDevice is alive
        .def(
            "create_submeshes",
            &MeshDevice::create_submeshes,
            nb::arg("submesh_shape"),
            nb::keep_alive<1, 0>())  // Keep MeshDevice alive as long as SubmeshDevices are alive
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
            "clear_program_cache",
            &MeshDevice::clear_program_cache,
            R"doc(
               Clear program cache across all devices in the mesh.
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
        .def("get_view", &MeshDevice::get_view, nb::rv_policy::reference_internal)
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
            [](MeshDevice* /*device*/) { return tt::tt_metal::hal::get_eps(); },
            R"doc(Returns machine epsilon value for current architecture.)doc")
        .def(
            "sfpu_nan",
            [](MeshDevice* /*device*/) { return tt::tt_metal::hal::get_nan(); },
            R"doc(Returns NaN value for current architecture.)doc")
        .def(
            "sfpu_inf",
            [](MeshDevice* /*device*/) { return tt::tt_metal::hal::get_inf(); },
            R"doc(Returns Infinity value for current architecture.)doc")
        .def(
            "worker_core_from_logical_core",
            &MeshDevice::worker_core_from_logical_core,
            nb::arg("logical_core"),
            R"doc(
                Convert a logical coordinate to a virtual coordinate for a worker core.

                Args:
                    logical_core (CoreCoord): The logical coordinate to convert.

                Returns:
                    CoreCoord: The virtual coordinate of the worker core.

                Example:
                    >>> device = ttnn.open_device(device_id=0)
                    >>> logical_core = ttnn.CoreCoord(0, 0)
                    >>> worker_core = device.worker_core_from_logical_core(logical_core)
                    >>> print(f"Worker core: x={worker_core.x}, y={worker_core.y}")
            )doc")
        .def(
            "get_worker_noc_hop_distance",
            [](MeshDevice& self, const CoreCoord& logical_src, const CoreCoord& logical_dst, NOC noc) {
                return tt::tt_metal::experimental::Device::get_worker_noc_hop_distance(
                    &self, logical_src, logical_dst, noc);
            },
            nb::arg("logical_src"),
            nb::arg("logical_dst"),
            nb::arg("noc"),
            R"doc(
                Returns the hop distance between two logical worker coordinates on a given NOC.

                This API is experimental and may evolve into a stable Device API in the future.

                Args:
                    logical_src (CoreCoord): The source logical coordinate.
                    logical_dst (CoreCoord): The destination logical coordinate.
                    noc (NOC): The NOC to use (ttnn.NOC.NOC_0 or ttnn.NOC.NOC_1).

                Returns:
                    int: The hop distance between the two coordinates on the given NOC.

                Example:
                    >>> device = ttnn.open_device(device_id=0)
                    >>> src = ttnn.CoreCoord(0, 0)
                    >>> dst = ttnn.CoreCoord(2, 3)
                    >>> noc0_distance = device.get_worker_noc_hop_distance(src, dst, ttnn.NOC.NOC_0)
                    >>> noc1_distance = device.get_worker_noc_hop_distance(src, dst, ttnn.NOC.NOC_1)
            )doc");

    auto py_mesh_device_view = static_cast<nb::class_<MeshDeviceView>>(mod.attr("MeshDeviceView"));
    py_mesh_device_view.def("shape", &MeshDeviceView::shape, nb::rv_policy::reference_internal)
        .def("num_devices", &MeshDeviceView::num_devices)
        .def("is_local", &MeshDeviceView::is_local, nb::arg("coord"));

    auto py_tensor_to_mesh = static_cast<nb::class_<TensorToMesh>>(mod.attr("CppTensorToMesh"));

    auto py_mesh_to_tensor = static_cast<nb::class_<MeshToTensor>>(mod.attr("CppMeshToTensor"));

    mod.def(
        "open_mesh_device",
        nb::overload_cast<
            size_t,
            size_t,
            size_t,
            const tt::tt_metal::DispatchCoreConfig&,
            const std::optional<MeshShape>&,
            const std::optional<MeshCoordinate>&,
            const std::vector<int>&,
            size_t>(&open_mesh_device),
        nb::kw_only(),
        nb::arg("l1_small_size"),
        nb::arg("trace_region_size"),
        nb::arg("num_command_queues"),
        nb::arg("dispatch_core_config"),
        nb::arg("mesh_shape") = nb::none(),
        nb::arg("offset") = nb::none(),
        nb::arg("physical_device_ids") = nb::cast(std::vector<int>{}),
        nb::arg("worker_l1_size") = DEFAULT_WORKER_L1_SIZE);
    mod.def("close_mesh_device", &close_mesh_device, nb::arg("mesh_device"));

    auto py_placement_shard = static_cast<nb::class_<MeshMapperConfig::Shard>>(mod.attr("PlacementShard"));
    py_placement_shard.def(nb::init<int>())
        .def(
            "__repr__",
            [](const MeshMapperConfig::Shard& shard) {
                std::ostringstream str;
                str << shard;
                return str.str();
            })
        .def_ro("dim", &MeshMapperConfig::Shard::dim);

    auto py_placement_replicate = static_cast<nb::class_<MeshMapperConfig::Replicate>>(mod.attr("PlacementReplicate"));
    py_placement_replicate
        .def("__init__", [](MeshMapperConfig::Replicate* t) { new (t) MeshMapperConfig::Replicate{}; })
        .def(
            "__repr__",
            [](const MeshMapperConfig::Replicate& replicate) {
                std::ostringstream str;
                str << replicate;
                return str.str();
            })
        .def(
            "__eq__",
            [](const MeshMapperConfig::Replicate& /*lhs*/, const MeshMapperConfig::Replicate& /*rhs*/) { return true; })
        .def("__ne__", [](const MeshMapperConfig::Replicate& /*lhs*/, const MeshMapperConfig::Replicate& /*rhs*/) {
            return false;
        });
    auto py_mesh_mapper_config = static_cast<nb::class_<MeshMapperConfig>>(mod.attr("MeshMapperConfig"));

    py_mesh_mapper_config.def(
        "__init__",
        [](MeshMapperConfig* t,
           tt::stl::SmallVector<MeshMapperConfig::Placement> placements,
           const std::optional<MeshShape>& mesh_shape_override) {
            new (t) MeshMapperConfig{.placements = std::move(placements), .mesh_shape_override = mesh_shape_override};
        },
        nb::arg("placements"),
        nb::arg("mesh_shape_override") = nb::none(),
        R"doc(
           Creates a MeshMapperConfig object with the given placements and mesh shape override.

           Args:
               placements (List[Placement]): A set of placements to use for the mapper.
               mesh_shape_override (MeshShape): If provided, overrides distribution shape of the mesh device.
               Used for distributing a tensor over ND shape that doesn't match the shape of the mesh device:
               when the shape fits within a mesh device, the tensor shards are distributed within the submesh
               region. Otherwise, the tensor shards are distributed across mesh in row-major order.
           )doc");

    using mmc_dim_t = decltype(MeshMapperConfig::Shard::dim);
    py_mesh_mapper_config
        .def(
            "__init__",
            [](MeshMapperConfig* t,
               std::optional<mmc_dim_t> row_dim,
               std::optional<mmc_dim_t> col_dim,
               const std::optional<MeshShape>& /*mesh_shape_override*/) {
                new (t) MeshMapperConfig;
                t->placements.push_back(
                    row_dim ? MeshMapperConfig::Placement{MeshMapperConfig::Shard{*row_dim}}
                            : MeshMapperConfig::Placement{MeshMapperConfig::Replicate{}});
                t->placements.push_back(
                    col_dim ? MeshMapperConfig::Placement{MeshMapperConfig::Shard{*col_dim}}
                            : MeshMapperConfig::Placement{MeshMapperConfig::Replicate{}});
            },
            nb::arg("row_dim") = nb::none(),
            nb::arg("col_dim") = nb::none(),
            nb::arg("mesh_shape_override") = nb::none(),
            R"doc(
           Creates a 2D MeshMapperConfig with the given placements and mesh shape override.

           Placements are supplied as optional integers: when set to None, the mapper will replicate over the dimension.
           Otherwise, the mapper will shard over the dimension.

           Args:
               row_dim Optional[int]: The row dimension to shard / replicate over.
               col_dim Optional[int]: The column dimension to shard / replicate over.
               mesh_shape_override Optional[MeshShape]: If provided, overrides distribution shape of the mesh device.
               )doc")
        .def("__repr__", [](const MeshMapperConfig& config) {
            std::ostringstream str;
            str << config;
            return str.str();
        });
    auto py_mesh_composer_config = static_cast<nb::class_<MeshComposerConfig>>(mod.attr("MeshComposerConfig"));
    py_mesh_composer_config
        .def(
            nb::init<tt::stl::SmallVector<int>, const std::optional<MeshShape>&>(),
            nb::arg("dims"),
            nb::arg("mesh_shape_override") = nb::none(),
            R"doc(
           Creates a MeshComposerConfig object with the given dimensions.

           Args:
               dims (List[int]): The dimensions to concat over.
               mesh_shape_override Optional[MeshShape]: If provided, overrides distribution shape of the mesh device.
           )doc")
        .def(
            "__init__",
            [](MeshComposerConfig* t,
               mmc_dim_t row_dim,
               mmc_dim_t col_dim,
               const std::optional<MeshShape>& mesh_shape_override) {
                new (t) MeshComposerConfig;
                t->dims.push_back(row_dim);
                t->dims.push_back(col_dim);
                t->mesh_shape_override = mesh_shape_override;
            },
            nb::arg("row_dim"),
            nb::arg("col_dim"),
            nb::arg("mesh_shape_override") = nb::none(),
            R"doc(
           Creates a 2D MeshComposerConfig object with the given dimensions.

           Args:
               row_dim (int): The dimension to concat over.
               col_dim (int): The dimension to concat over.
               mesh_shape_override Optional[MeshShape]: If provided, overrides distribution shape of the mesh device.
           )doc")
        .def("__repr__", [](const MeshComposerConfig& config) {
            std::ostringstream str;
            str << config;
            return str.str();
        });

    auto py_distributed_host_buffer = static_cast<nb::class_<DistributedHostBuffer>>(mod.attr("DistributedHostBuffer"));
    py_distributed_host_buffer.def("is_local", &DistributedHostBuffer::is_local, nb::arg("coord"))
        .def("shape", &DistributedHostBuffer::shape, nb::rv_policy::reference_internal);

    auto py_tensor_topology = static_cast<nb::class_<TensorTopology>>(mod.attr("TensorTopology"));
    py_tensor_topology
        .def(
            nb::init<
                tt::tt_metal::distributed::MeshShape,
                ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>,
                std::vector<tt::tt_metal::distributed::MeshCoordinate>>(),
            nb::arg("distribution_shape"),
            nb::arg("placements"),
            nb::arg("mesh_coords"),
            "Constructor for TensorTopology")
        .def("distribution_shape", &TensorTopology::distribution_shape, nb::rv_policy::reference_internal)
        .def("placements", &TensorTopology::placements, nb::rv_policy::reference_internal)
        .def("mesh_coords", &TensorTopology::mesh_coords, nb::rv_policy::reference_internal)
        .def("__eq__", [](const TensorTopology& self, const TensorTopology& other) { return self == other; })
        .def("__ne__", [](const TensorTopology& self, const TensorTopology& other) { return self != other; })
        .def("__repr__", [](const TensorTopology& self) {
            std::ostringstream oss;
            oss << self;
            return oss.str();
        });

    mod.def(
        "get_device_tensors",
        &get_device_tensors,
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
        [](MeshDevice& mesh_device) -> nbh::unique_ptr<TensorToMesh> {
            return nbh::steal_rewrap_unique<TensorToMesh>(replicate_tensor_to_mesh_mapper(mesh_device));
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
        [](MeshDevice& mesh_device, int dim) -> nbh::unique_ptr<TensorToMesh> {
            return nbh::steal_rewrap_unique<TensorToMesh>(shard_tensor_to_mesh_mapper(mesh_device, dim));
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
        "create_mesh_mapper",
        [](MeshDevice& mesh_device, const MeshMapperConfig& config) -> nbh::unique_ptr<TensorToMesh> {
            return nbh::steal_rewrap_unique<TensorToMesh>(create_mesh_mapper(mesh_device, config));
        },
        nb::arg("mesh_device"),
        nb::arg("config"),
        R"doc(
       Returns an ND mapper for sharding and replicating a tensor over the given dimensions for the given mesh.

       Args:
           mesh_device (MeshDevice): The mesh to create the mapper for.
           config (MeshMapperConfig): A config object representing a set of placements.

       Returns:
           TensorToMesh: A mapper providing the desired sharding.
   )doc");
    mod.def(
        "compute_distribution_to_mesh_mapping",
        [](const tt::tt_metal::distributed::MeshShape& distribution_shape,
           const tt::tt_metal::distributed::MeshShape& mesh_shape)
            -> std::vector<tt::tt_metal::distributed::MeshCoordinate> {
            return ttnn::distributed::compute_distribution_to_mesh_mapping(distribution_shape, mesh_shape);
        },
        nb::arg("distribution_shape"),
        nb::arg("mesh_shape"),
        R"doc(
       Compute ordered mesh coordinates for distribution coordinate mapping.

       This function computes how distribution coordinates should map to mesh coordinates
       based on the distribution mode. For ROW_MAJOR mode, it returns mesh coordinates
       in row-major order. For SUBMESH mode, coordinates map directly.

       Args:
           distribution_shape (MeshShape): The distribution (override) mesh shape.
           mesh_shape (MeshShape): The physical device mesh shape.

       Returns:
           list[MeshCoordinate]: Vector of mesh coordinates in the order they should be mapped to distribution coordinates.
   )doc");
    mod.def(
        "concat_mesh_to_tensor_composer",
        [](MeshDevice& mesh_device, int dim) -> nbh::unique_ptr<MeshToTensor> {
            return nbh::steal_rewrap_unique<MeshToTensor>(concat_mesh_to_tensor_composer(mesh_device, dim));
        },
        nb::arg("mesh_device"),
        nb::arg("dim"));
    mod.def(
        "create_mesh_composer",
        [](MeshDevice& mesh_device, const MeshComposerConfig& config) -> nbh::unique_ptr<MeshToTensor> {
            return nbh::steal_rewrap_unique<MeshToTensor>(create_mesh_composer(mesh_device, config));
        },
        nb::arg("mesh_device"),
        nb::arg("config"),
        R"doc(
            Returns an ND composer that concatenates a tensor across the given dimensions.

            Args:
                mesh_device (MeshDevice): The mesh device to create the composer for.
                config (MeshComposerConfig): A config object representing the dimensions to concat over.
                shape (MeshShape): If provided, overrides the logical shape of the mesh.

            Returns:
                TensorToMesh: A composer providing the desired concatenation.
   )doc");
    mod.def(
        "distribute_tensor",
        [](const Tensor& tensor, const TensorToMesh& mapper, std::optional<MeshDevice*> mesh_device = std::nullopt)
            -> Tensor { return distribute_tensor(tensor, mapper, nbh::rewrap_optional(mesh_device)); },
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
        "from_host_shards",
        [](const std::vector<Tensor>& tensors, const MeshShape& mesh_shape) -> Tensor {
            return from_host_shards(tensors, mesh_shape);
        },
        nb::arg("tensors"),
        nb::arg("mesh_shape"),
        R"doc(
            Creates a multi-device host tensor from a set of individual host shards.

            Args:
                tensors (List[Tensor]): The tensor shards to aggregate.
                mesh_shape (MeshShape): The shape of the mesh to aggregate the shards over.

            Returns:
                Tensor: The multi-device host tensor.
            )doc");
    mod.def(
        "combine_device_tensors",
        [](const std::vector<Tensor>& tensors) -> Tensor { return combine_device_tensors(tensors); },
        nb::kw_only(),
        nb::arg("tensors"),
        R"doc(
            Combines tensor shards allocated on individual devices into a single multi-device tensor. All tensors shards must be allocated on the same mesh buffer.

            Args:
                tensors (List[Tensor]): The tensor shards to combine.

            Returns:
                Tensor: The combined tensor.
            )doc");
    mod.def("using_distributed_env", &tt::tt_metal::distributed::UsingDistributedEnvironment);

    // Distributed context bindings (moved from distributed_context.cpp)
    using namespace tt::tt_metal::distributed::multihost;

    // Initialize the distributed context
    mod.def(
        "init_distributed_context",
        []() {
            // In Python context, we typically don't have argc/argv in the same way
            // This is a simplified initialization that works with default MPI settings
            static char prog_name[] = "python";
            static char* argv[] = {prog_name, nullptr};
            DistributedContext::create(1, argv);
        },
        R"doc(
            Initialize the distributed context with default settings.

            This is a convenience function for Python that initializes the distributed
            context without requiring command-line arguments.

            Example:
                >>> import ttnn
                >>> ttnn.init_distributed_context()
                >>> rank = ttnn.distributed_context_get_rank()
                >>> size = ttnn.distributed_context_get_size()
                >>> print(f"Rank {int(rank)} of {int(size)}")
        )doc");

    // Check if distributed context is initialized
    mod.def(
        "is_initialized",
        &DistributedContext::is_initialized,
        R"doc(
            Check if the distributed context has been initialized.

            Returns:
                bool: True if initialized, False otherwise.

            Example:
                >>> import ttnn
                >>> if ttnn.distributed_context_is_initialized():
                >>>     rank = ttnn.distributed_context_get_rank()
        )doc");

    // Get the rank of the current process
    mod.def(
        "get_rank",
        []() -> Rank {
            if (!DistributedContext::is_initialized()) {
                throw std::runtime_error("Distributed context not initialized. Call init_distributed_context() first.");
            }
            return DistributedContext::get_current_world()->rank();
        },
        R"doc(
            Get the rank of the current process.

            Returns:
                Rank: The rank of the current process (0-indexed).

            Raises:
                RuntimeError: If the distributed context has not been initialized.

            Example:
                >>> import ttnn
                >>> rank = ttnn.distributed_context_get_rank()
                >>> print(f"This is rank {rank}")
                >>> # Convert to int if needed
                >>> rank_int = int(rank)
        )doc");

    // Get the total number of processes
    mod.def(
        "get_size",
        []() -> Size {
            if (!DistributedContext::is_initialized()) {
                throw std::runtime_error("Distributed context not initialized. Call init_distributed_context() first.");
            }
            return DistributedContext::get_current_world()->size();
        },
        R"doc(
            Get the total number of processes.

            Returns:
                Size: The total number of processes.

            Raises:
                RuntimeError: If the distributed context has not been initialized.

            Example:
                >>> import ttnn
                >>> size = ttnn.distributed_context_get_size()
                >>> print(f"Total processes: {size}")
                >>> # Convert to int if needed
                >>> size_int = int(size)
        )doc");

    // Synchronize all processes
    mod.def(
        "barrier",
        []() {
            if (!DistributedContext::is_initialized()) {
                throw std::runtime_error("Distributed context not initialized. Call init_distributed_context() first.");
            }
            DistributedContext::get_current_world()->barrier();
        },
        R"doc(
            Synchronize all processes.

            This function blocks until all processes have reached this point.

            Raises:
                RuntimeError: If the distributed context has not been initialized.

            Example:
                >>> import ttnn
                >>> # Do some work...
                >>> ttnn.distributed_context_barrier()
                >>> # All processes continue from here
        )doc");
}

}  // namespace ttnn::distributed
