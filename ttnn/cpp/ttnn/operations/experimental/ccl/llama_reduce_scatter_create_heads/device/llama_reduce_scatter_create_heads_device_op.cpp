// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "llama_reduce_scatter_create_heads_device_op.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::ccl {

LlamaReduceScatterCreateHeadsDeviceOperation::program_factory_t
LlamaReduceScatterCreateHeadsDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return LlamaReduceScatterCreateHeads{};
}

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
    tt::tt_metal::MemoryConfig q_mem_config = attributes.qkv_memory_config.value().with_shard_spec(q_shard_spec);
    tt::tt_metal::MemoryConfig k_mem_config = attributes.qkv_memory_config.value().with_shard_spec(k_shard_spec);
    tt::tt_metal::MemoryConfig v_mem_config = attributes.qkv_memory_config.value().with_shard_spec(v_shard_spec);

    return {
        TensorSpec(
            q_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), q_mem_config)),
        TensorSpec(
            q_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), k_mem_config)),
        TensorSpec(
            q_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), v_mem_config))};
}

LlamaReduceScatterCreateHeadsDeviceOperation::tensor_return_value_t
LlamaReduceScatterCreateHeadsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    std::vector<ttnn::Tensor> tensors{};
    for (auto& output_spec : output_specs) {
        auto tensor = create_device_tensor(output_spec, tensor_args.input_tensor.device());
        tensors.push_back(tensor);
    }
    return tensors;
}

tt::tt_metal::operation::Hash LlamaReduceScatterCreateHeadsDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto program_factory = select_program_factory(attributes, tensor_args);

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
        tensor_args.input_tensor.device()->id(),
        program_factory.index());
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
