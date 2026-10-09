// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/broadcast/device/broadcast_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

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
    // a device off that line would never be signalled. The label is therefore the input's with Replicate on the
    // broadcast axis (`cluster_axis`) and the other axes kept, or Replicate on every axis when the whole (1-D)
    // distribution is the line. Stated directly rather than through the helper: uncollapse_placements reads a label's
    // coordinates to decide whether axis i of the label is axis i of the mesh, and the labels broadcast runs on do
    // not pass that test although the op is fine with them -- a 1-D sub-range such as `MeshMapperConfig([Shard(3)],
    // MeshShape(4))` on a 2x4 mesh ("does not cover the mesh"), and the N-D `(1, C)` / `(R, 1)` sub-range labels
    // from_torch produces for `ShardTensor2dMesh(mesh_shape=(1, C))` on the parent mesh, whose mesh_coords list every
    // device of the mesh ("does not align with the axes"; test_broadcast_nd_along_one_axis_keeps_the_other). Known
    // limit of the direct edit: an N-D label whose axes are not the mesh axes (`MeshMapperConfig([Shard(3),
    // Replicate], MeshShape(4, 1))` on a 2x4 mesh does not fit and distributes row-major over row 0) keeps its
    // `Shard(3)` on the non-broadcast axis and claims four distinct pieces where every device holds the sender's;
    // labelling it honestly needs the helper to accept these labels first (follow-up in ccl_topology_utils).
    const auto& input_topology = tensor_args.input_tensor.tensor_topology();
    auto placements = input_topology.placements();
    const auto& cluster_axis = operation_attributes.cluster_axis;
    if (cluster_axis.has_value() && *cluster_axis < placements.size()) {
        placements[*cluster_axis] = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
    } else {
        for (auto& placement : placements) {
            placement = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
        }
    }
    return {tt::tt_metal::TensorTopology(
        input_topology.distribution_shape(), std::move(placements), input_topology.mesh_coords())};
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
