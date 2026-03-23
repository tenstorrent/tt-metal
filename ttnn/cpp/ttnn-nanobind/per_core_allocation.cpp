// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "per_core_allocation.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/experimental/per_core_allocation/allocator_mode.hpp>
#include <tt-metalium/experimental/per_core_allocation/mesh_device.hpp>
#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include <ttnn/tensor/memory_config/memory_config.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace nb = nanobind;

using namespace tt::tt_metal;
namespace per_core = tt::tt_metal::experimental::per_core_allocation;

namespace ttnn::per_core_allocation {

void py_module(nb::module_& mod) {
    nb::enum_<AllocatorMode>(mod, "AllocatorMode", "Enum of L1 allocator modes.")
        .value("LOCKSTEP", AllocatorMode::LOCKSTEP)
        .value("HYBRID", AllocatorMode::HYBRID);

    mod.def(
        "CreateDevice",
        [](int device_id,
           uint8_t num_command_queues,
           size_t l1_small_size,
           size_t trace_region_size,
           const DispatchCoreConfig& dispatch_core_config,
           size_t worker_l1_size,
           AllocatorMode allocator_mode) {
            return per_core::create_unit_mesh(
                device_id,
                l1_small_size,
                trace_region_size,
                num_command_queues,
                dispatch_core_config,
                /*l1_bank_remap=*/{},
                worker_l1_size,
                allocator_mode);
        },
        nb::arg("device_id"),
        nb::arg("num_command_queues") = 1,
        nb::arg("l1_small_size") = DEFAULT_L1_SMALL_SIZE,
        nb::arg("trace_region_size") = DEFAULT_TRACE_REGION_SIZE,
        nb::arg("dispatch_core_config") = nb::cast(DispatchCoreConfig{}),
        nb::kw_only(),
        nb::arg("worker_l1_size") = DEFAULT_WORKER_L1_SIZE,
        nb::arg("allocator_mode") = AllocatorMode::HYBRID);

    mod.def(
        "CreateDevices",
        [](const std::vector<int>& device_ids,
           uint8_t num_command_queues,
           size_t l1_small_size,
           size_t trace_region_size,
           const DispatchCoreConfig& dispatch_core_config,
           size_t worker_l1_size,
           AllocatorMode allocator_mode) {
            return per_core::create_unit_meshes(
                device_ids,
                l1_small_size,
                trace_region_size,
                num_command_queues,
                dispatch_core_config,
                /*l1_bank_remap=*/{},
                worker_l1_size,
                allocator_mode);
        },
        nb::arg("device_ids"),
        nb::arg("num_command_queues") = 1,
        nb::arg("l1_small_size") = DEFAULT_L1_SMALL_SIZE,
        nb::arg("trace_region_size") = DEFAULT_TRACE_REGION_SIZE,
        nb::arg("dispatch_core_config") = nb::cast(DispatchCoreConfig{}),
        nb::kw_only(),
        nb::arg("worker_l1_size") = DEFAULT_WORKER_L1_SIZE,
        nb::arg("allocator_mode") = AllocatorMode::HYBRID);

    mod.def(
        "open_mesh_device",
        [](size_t l1_small_size,
           size_t trace_region_size,
           size_t num_command_queues,
           const DispatchCoreConfig& dispatch_core_config,
           const std::optional<distributed::MeshShape>& mesh_shape,
           const std::optional<distributed::MeshCoordinate>& offset,
           const std::vector<int>& physical_device_ids,
           size_t worker_l1_size,
           AllocatorMode allocator_mode) -> std::shared_ptr<distributed::MeshDevice> {
            return per_core::create_mesh_device(
                distributed::MeshDeviceConfig(mesh_shape, offset, physical_device_ids),
                l1_small_size,
                trace_region_size,
                num_command_queues,
                dispatch_core_config,
                /*l1_bank_remap=*/{},
                worker_l1_size,
                allocator_mode);
        },
        nb::kw_only(),
        nb::arg("l1_small_size") = DEFAULT_L1_SMALL_SIZE,
        nb::arg("trace_region_size") = DEFAULT_TRACE_REGION_SIZE,
        nb::arg("num_command_queues") = 1,
        nb::arg("dispatch_core_config") = nb::cast(DispatchCoreConfig{}),
        nb::arg("mesh_shape") = nb::none(),
        nb::arg("offset") = nb::none(),
        nb::arg("physical_device_ids") = nb::cast(std::vector<int>{}),
        nb::arg("worker_l1_size") = DEFAULT_WORKER_L1_SIZE,
        nb::arg("allocator_mode") = AllocatorMode::HYBRID);

    mod.def(
        "set_per_core_allocation",
        [](MemoryConfig& config, bool enable) { per_core::set_per_core_allocation(config, enable); },
        nb::arg("config"),
        nb::arg("enable") = true);

    mod.def(
        "MemoryConfig",
        [](TensorMemoryLayout memory_layout,
           BufferType buffer_type,
           std::optional<ShardSpec> shard_spec,
           bool per_core_allocation) {
            MemoryConfig config(memory_layout, buffer_type, std::move(shard_spec));
            if (per_core_allocation) {
                per_core::set_per_core_allocation(config, true);
            }
            return config;
        },
        nb::arg("memory_layout") = TensorMemoryLayout::INTERLEAVED,
        nb::arg("buffer_type") = BufferType::L1,
        nb::arg("shard_spec") = nb::none(),
        nb::arg("per_core_allocation") = true);

    mod.def(
        "per_core_buffer_address",
        [](const ttnn::Tensor& tensor, const CoreCoord& core) -> uint32_t {
            TT_FATAL(is_device_tensor(tensor), "{} doesn't support per_core_buffer_address", tensor.storage_type());
            TT_FATAL(tensor.is_allocated(), "Tensor is not allocated.");
            return per_core::get_per_core_address(tensor.mesh_buffer(), core);
        },
        nb::arg("tensor"),
        nb::arg("core"));

    mod.def(
        "to_single_device",
        [](const Tensor& host_tensor,
           distributed::MeshDevice* mesh_device,
           const distributed::MeshCoordinate& coord,
           const MemoryConfig& mem_config) -> Tensor {
            TT_FATAL(host_tensor.storage_type() == StorageType::HOST, "to_single_device expects a host tensor");

            auto tensor_spec = TensorSpec(
                host_tensor.logical_shape(),
                TensorLayout(host_tensor.dtype(), host_tensor.tensor_spec().page_config(), mem_config));

            auto mesh_buffer = per_core::create_on_single_device(
                tt::tt_metal::distributed::ReplicatedBufferConfig{
                    .size = tensor_spec.compute_packed_buffer_size_bytes()},
                tt::tt_metal::distributed::DeviceLocalBufferConfig{
                    .page_size = tensor_spec.compute_page_size_bytes(),
                    .buffer_type = mem_config.buffer_type(),
                    .sharding_args = tensor_spec.compute_buffer_sharding_args(),
                },
                mesh_device,
                coord);

            // Write host data to the single device via shard transfer
            const auto& host_storage = host_tensor.host_storage();
            auto host_buffer = host_storage.buffer().get_shard(tt::tt_metal::distributed::MeshCoordinate(0, 0));
            TT_FATAL(host_buffer.has_value(), "Host tensor has no data");

            tt::tt_metal::distributed::ShardDataTransfer transfer{coord};
            transfer.host_data(host_buffer->view_bytes().data());
            transfer.region(BufferRegion(0, host_buffer->view_bytes().size()));

            mesh_device->mesh_command_queue().enqueue_write_shards(mesh_buffer, {transfer}, /*blocking=*/true);

            DeviceStorage device_storage(std::move(mesh_buffer), {coord});
            return Tensor(std::move(device_storage), tensor_spec, TensorTopology{});
        },
        nb::arg("host_tensor"),
        nb::arg("mesh_device"),
        nb::arg("coord"),
        nb::arg("memory_config"),
        R"doc(
        Write a host tensor to a single device within a mesh.
        Allocates on the target device's own allocator and writes data.
        Per-core allocations are automatically mirrored into the mesh-level allocator.
    )doc");
}

}  // namespace ttnn::per_core_allocation
