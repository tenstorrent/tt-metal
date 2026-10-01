// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "llama_reduce_scatter_create_heads_device_op.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::ccl {
void LlamaReduceScatterCreateHeadsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;

    TT_FATAL(attributes.dim == 3, "dim must be 3, got {}", attributes.dim);
    TT_FATAL(attributes.cluster_axis == 1, "cluster_axis must be 1, got {}", attributes.cluster_axis);
    TT_FATAL(
        attributes.ring_devices == 4 or attributes.ring_devices == 2,
        "ring_devices must be 4 or 2, got {}",
        attributes.ring_devices);
    TT_FATAL(attributes.cross_device_semaphore.has_value(), "Cross device semaphore is not present");

    TT_FATAL(input_tensor.shard_spec().has_value(), "input_tensor must have a shard spec");
    TT_FATAL(
        input_tensor.shard_spec().value().shape[0] == 32,
        "input_tensor shard height must be 32 but got {}",
        input_tensor.shard_spec().value().shape[0]);

    TT_FATAL(
        tensor_args.intermediate_packet_buffer.shard_spec().has_value(),
        "intermediate_packet_buffer must have a shard spec");
    TT_FATAL(
        tensor_args.intermediate_packet_buffer.shard_spec().value().shape[0] == 32,
        "intermediate_packet_buffer shard height must be 32 but got {}",
        tensor_args.intermediate_packet_buffer.shard_spec().value().shape[0]);
    if (attributes.qkv_memory_config.has_value()) {
        TT_FATAL(
            attributes.qkv_memory_config.value().shard_spec().has_value(), "qkv_memory_config must have a shard spec");
        TT_FATAL(
            attributes.qkv_memory_config.value().shard_spec().value().shape[0] == 32,
            "qkv_memory_config shard height must be 32 but got {}",
            attributes.qkv_memory_config.value().shard_spec().value().shape[0]);
    }
}

void LlamaReduceScatterCreateHeadsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& /*tensor_args*/) {}

LlamaReduceScatterCreateHeadsDeviceOperation::spec_return_value_t
LlamaReduceScatterCreateHeadsDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;

    // input is unpadded, output is padded. Ex, input: 3584, 112 tiles, padded to 5 tiles per core, total width is 120
    // tiles (3840). this should be changed to use unpadded output in the future.
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.logical_shape();
    const auto batch = attributes.slice_size;
    const auto head_dim = attributes.head_dim;
    const Shape q_output_shape({input_shape[0], batch, attributes.num_heads, head_dim});
    CoreRangeSet q_shard_grid, k_shard_grid, v_shard_grid;
    auto sub_core_grid = attributes.qkv_memory_config.value().shard_spec()->grid;
    auto start_core_coord = sub_core_grid.bounding_box().start_coord;
    auto next_core_coord = start_core_coord;

    q_shard_grid =
        tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(start_core_coord, batch, sub_core_grid, true);

    CoreRangeSet q_batch_grid =
        tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(start_core_coord, batch + 1, sub_core_grid, true);
    if (!q_batch_grid.ranges().empty()) {
        next_core_coord = q_batch_grid.ranges().back().end_coord;
    }
    k_shard_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(next_core_coord, batch, sub_core_grid, true);

    CoreRangeSet q_two_batch_grid =
        tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(start_core_coord, (2 * batch) + 1, sub_core_grid, true);
    if (!q_two_batch_grid.ranges().empty()) {
        next_core_coord = q_two_batch_grid.ranges().back().end_coord;
    }
    v_shard_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(next_core_coord, batch, sub_core_grid, true);

    tt::tt_metal::ShardSpec q_shard_spec{q_shard_grid, {attributes.num_heads, head_dim}};
    tt::tt_metal::ShardSpec k_shard_spec{k_shard_grid, {attributes.num_heads, head_dim}};
    tt::tt_metal::ShardSpec v_shard_spec{v_shard_grid, {attributes.num_heads, head_dim}};
    const auto& qkv_memory_config = attributes.qkv_memory_config.value();
    tt::tt_metal::MemoryConfig q_mem_config =
        tt::tt_metal::MemoryConfig(qkv_memory_config.memory_layout(), qkv_memory_config.buffer_type(), q_shard_spec);
    tt::tt_metal::MemoryConfig k_mem_config =
        tt::tt_metal::MemoryConfig(qkv_memory_config.memory_layout(), qkv_memory_config.buffer_type(), k_shard_spec);
    tt::tt_metal::MemoryConfig v_mem_config =
        tt::tt_metal::MemoryConfig(qkv_memory_config.memory_layout(), qkv_memory_config.buffer_type(), v_shard_spec);

    return {
        tt::tt_metal::TensorSpec(
            q_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), q_mem_config)),
        tt::tt_metal::TensorSpec(
            q_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), k_mem_config)),
        tt::tt_metal::TensorSpec(
            q_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), v_mem_config))};
}

LlamaReduceScatterCreateHeadsDeviceOperation::tensor_return_value_t
LlamaReduceScatterCreateHeadsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    std::vector<ttnn::Tensor> tensors{};
    tensors.reserve(output_specs.size());
    for (auto& output_spec : output_specs) {
        auto tensor = create_device_tensor(output_spec, tensor_args.input_tensor.device());
        tensors.push_back(std::move(tensor));
    }
    return tensors;
}

std::vector<tt::tt_metal::TensorTopology> LlamaReduceScatterCreateHeadsDeviceOperation::compute_output_topologies(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    // Along `cluster_axis` the op sums the partial QKV projections and scatters the ROWS: device c keeps rows
    // [c * slice_size, (c + 1) * slice_size) of input dim 2 as output dim 1 (batch); input dim 0 stays dim 0; the
    // fused QKV width (input dim 3) is reshaped into heads (dims 2-3) and split across the q/k/v outputs
    // (test_llama_reduce_scatter_create_heads_async_TG composes q/k/v with ConcatMesh2dToTensor dims=(0, 1) against
    // the row-sliced reduction). So every output carries the reduce_scatter label for scatter dim 2, renumbered to the
    // output layout: Shard{2} -> Shard{1}, Shard{0} stays, Replicate stays. A Shard of dim 1 (summed away) or dim 3
    // (reshaped into heads) on a non-cluster axis has no output dim this hook can vouch for: {} (union default).
    const auto rs_topology = ttnn::operations::ccl::common::reduce_scatter_output_topology(
        tensor_args.input_tensor, attributes.cluster_axis, /*scatter_dim=*/2);
    if (!rs_topology.has_value()) {
        return {};
    }
    const auto rank = static_cast<uint32_t>(tensor_args.input_tensor.logical_shape().rank());
    ttnn::operations::ccl::common::TopologyPlacements placements;
    placements.reserve(rs_topology->placements().size());
    for (const auto& placement : rs_topology->placements()) {
        const auto* shard = std::get_if<Shard>(&placement);
        if (shard == nullptr) {
            placements.push_back(placement);
            continue;
        }
        const auto dim = ttnn::operations::ccl::common::normalize_tensor_dim(shard->dim, rank);
        if (dim == 2u) {
            placements.push_back(Shard{1});
        } else if (dim == 0u) {
            placements.push_back(Shard{0});
        } else {
            return {};
        }
    }
    const tt::tt_metal::TensorTopology heads_topology(
        rs_topology->distribution_shape(), std::move(placements), rs_topology->mesh_coords());
    return {heads_topology, heads_topology, heads_topology};
}

tt::tt_metal::operation::Hash LlamaReduceScatterCreateHeadsDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<LlamaReduceScatterCreateHeadsDeviceOperation>(
        attributes.dim,
        attributes.cluster_axis,
        attributes.ring_devices,
        attributes.num_links,
        attributes.num_heads,
        attributes.num_kv_heads,
        attributes.head_dim,
        attributes.slice_size,
        attributes.topology,
        attributes.use_noc1_only,
        attributes.use_optimal_ccl_for_llama,
        tensor_args.input_tensor.dtype(),
        tensor_args.input_tensor.memory_config(),
        tensor_args.input_tensor.device()->id());
}

}  // namespace ttnn::operations::experimental::ccl

namespace ttnn::prim {

ttnn::operations::experimental::ccl::LlamaReduceScatterCreateHeadsDeviceOperation::tensor_return_value_t
llama_reduce_scatter_create_heads(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& intermediate_packet_buffer,
    int32_t dim,
    const GlobalSemaphore& semaphore,
    tt::tt_metal::SubDeviceId subdevice_id,
    uint32_t cluster_axis,
    uint32_t ring_devices,
    ttnn::ccl::Topology topology,
    uint32_t num_links,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t slice_size,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::MemoryConfig>& qkv_memory_config,
    bool use_noc1_only,
    bool use_optimal_ccl_for_llama) {
    using OperationType = ttnn::operations::experimental::ccl::LlamaReduceScatterCreateHeadsDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .dim = (dim < 0 ? uint32_t(input_tensor.logical_shape().rank() + dim) : (uint32_t)dim),
        .cross_device_semaphore = semaphore,
        .subdevice_id = subdevice_id,
        .cluster_axis = cluster_axis,
        .output_mem_config = memory_config,
        .ring_devices = ring_devices,
        .topology = topology,
        .num_links = num_links,
        .num_heads = num_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .slice_size = slice_size,
        .qkv_memory_config = qkv_memory_config,
        .use_noc1_only = use_noc1_only,
        .use_optimal_ccl_for_llama = use_optimal_ccl_for_llama,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor, .intermediate_packet_buffer = intermediate_packet_buffer};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
