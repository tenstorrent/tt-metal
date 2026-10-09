// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/broadcast/device/broadcast_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"

namespace ttnn::prim {
void BroadcastDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Operands to broadcast need to be on device! Storage type: {}",
        static_cast<int>(input_tensor.storage_type()));
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to broadcast need to be allocated in buffers on device!");

    TT_FATAL(
        operation_attributes.num_links > 0,
        "Error, num_links should be more than 0 but has {}",
        operation_attributes.num_links);
    TT_FATAL(
        operation_attributes.num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelized over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout());
}

tt::tt_metal::TensorSpec BroadcastDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& shape = input_tensor.logical_shape();
    return tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), input_tensor.tensor_spec().page_config(), operation_attributes.output_mem_config));
}

Tensor BroadcastDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, tensor_args.input_tensor.device());
}

BroadcastDeviceOperation::topology_return_value_t BroadcastDeviceOperation::compute_output_topologies(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Every device along the broadcast line ends up holding the sender's bytes. The op only ever runs on a tensor
    // that spans the sender's line: every non-sender's link-0 worker waits on the semaphore the sender's line
    // multicast raises (BroadcastProgramFactory::create_at, `wait_output_semaphore = (link == 0) && !is_sender`), so
    // a device off that line would never be signalled. Along that line the honest label is Replicate on the broadcast
    // axis (`cluster_axis`) with the other axes kept, or Replicate on every axis when the whole distribution is the
    // line.
    //
    // A 1-D label is stated directly: its single placement describes the line -- the whole 1xN mesh, or a sub-range
    // such as `MeshMapperConfig([Shard(3)], MeshShape(4))` on a 2x4 mesh, which the helper's uncollapse_placements
    // refuses as not covering the mesh although it is a perfectly good broadcast input -- so it becomes Replicate
    // whatever `cluster_axis` says. An N-D label goes through the helper (the all_broadcast label: `cluster_axis`
    // becomes Replicate, no contiguity requirement), which edits by axis index only when axis i of the label is axis
    // i of the mesh and refuses otherwise: `MeshMapperConfig([Shard(3), Replicate], MeshShape(4, 1))` on a 2x4 mesh
    // does not fit, distributes row-major over row 0, and its `Shard(3)` axis is no mesh axis, so no per-axis edit
    // describes it (editing index 1 kept `Shard(3)` and claimed four distinct pieces where every device holds the
    // sender's). A refusal (nullopt, already reported by the helper) keeps the input's label -- the same under-claim,
    // now logged, and an error under strict mode -- stated explicitly as the sibling hooks do.
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_topology = input_tensor.tensor_topology();
    if (input_topology.placements().size() <= 1) {
        auto placements = input_topology.placements();
        for (auto& placement : placements) {
            placement = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
        }
        return {tt::tt_metal::TensorTopology(
            input_topology.distribution_shape(), std::move(placements), input_topology.mesh_coords())};
    }
    return {
        ttnn::operations::ccl::common::all_broadcast_output_topology(input_tensor, operation_attributes.cluster_axis)
            .value_or(input_topology)};
}

Tensor broadcast(
    const ttnn::Tensor& input_tensor,
    const MeshCoordinate& sender_coord,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    tt::tt_fabric::Topology topology,
    std::optional<uint32_t> cluster_axis,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    const auto& tensor_topology = input_tensor.tensor_topology();
    const auto& tensor_topology_shape = tensor_topology.distribution_shape();

    if (!cluster_axis.has_value()) {
        TT_FATAL(
            tensor_topology_shape.is_line_topology(),
            "broadcast op is only supported for a linear tensor topology shape");
    }

    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);

    TT_FATAL(
        num_devices > 1,
        "broadcast op will only work for num_devices > 1, but has {}, shape: {}",
        num_devices,
        tensor_topology_shape);

    tt::tt_fabric::Topology ccl_topology = topology;
    if (num_devices == 2) {
        ccl_topology = tt::tt_fabric::Topology::Linear;
    }
    log_debug(tt::LogOp, "DEBUG: creating line_fabric with num devices: {}, num links: {}", num_devices, num_links);
    log_debug(tt::LogOp, "DEBUG: line_fabric is created");

    return ttnn::device_operation::launch<BroadcastDeviceOperation>(
        BroadcastParams(
            sender_coord,
            num_links,
            num_devices,
            memory_config.value_or(input_tensor.memory_config()),
            ccl_topology,
            cluster_axis,
            sub_device_id),
        BroadcastInputs{.input_tensor = input_tensor});
}
}  // namespace ttnn::prim
