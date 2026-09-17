// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_fanout_reach_device_operation.hpp"

#include <tt_stl/small_vector.hpp>

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fanout_reach {

namespace {

void validate_dram_row_major(const ttnn::Tensor& t, const char* name) {
    TT_FATAL(t.buffer() != nullptr, "moe_fanout_reach: {} has no device buffer", name);
    TT_FATAL(
        t.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "moe_fanout_reach: {} must be ROW_MAJOR, got {}",
        name,
        t.layout());
    TT_FATAL(
        t.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "moe_fanout_reach: {} must be interleaved; the kernel addresses it by page index",
        name);
}

uint32_t axis_extent(const MoeFanoutReachParams& args) {
    return static_cast<uint32_t>(args.device->shape()[args.axis]);
}

}  // namespace

void MoeFanoutReachDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(args.device != nullptr, "moe_fanout_reach requires a mesh device in attributes");
    TT_FATAL(
        args.axis < args.device->shape().dims(),
        "moe_fanout_reach: cluster_axis {} is out of range for a {} mesh",
        args.axis,
        args.device->shape());

    const uint32_t extent = axis_extent(args);
    // The same bound dispatch_fabric2d puts on the axis it consumes this table over. An odd extent has
    // no chip at exactly half the ring, so the clockwise tie rule would never fire and the table would
    // describe a ring this op's consumer cannot run on.
    TT_FATAL(
        extent >= 4 && extent % 2 == 0,
        "moe_fanout_reach: cluster_axis {} extent {} must be even and at least 4, matching the axis "
        "dispatch_fabric2d relays over",
        args.axis,
        extent);
    TT_FATAL(
        args.dispatch_group_size == extent,
        "moe_fanout_reach: dispatch_group_size is {} but cluster_axis {} has extent {}. The reach row "
        "is measured in hops around THAT ring, so a mismatch means the wrong axis was named",
        args.dispatch_group_size,
        args.axis,
        extent);
    TT_FATAL(args.num_routed_experts > 0, "moe_fanout_reach: num_routed_experts must be > 0");
    // Expert ids arrive as UINT16 and the kernel packs a compacted expert index into 16 bits of a
    // per-expert word beside the destination row.
    TT_FATAL(
        args.num_routed_experts <= 0xFFFFu,
        "moe_fanout_reach: num_routed_experts {} does not fit the UINT16 expert ids this op reads",
        args.num_routed_experts);
    TT_FATAL(
        args.num_experts_per_tok > 0 && args.num_experts_per_tok <= args.num_routed_experts,
        "moe_fanout_reach: num_experts_per_tok {} must be in 1..num_routed_experts {}",
        args.num_experts_per_tok,
        args.num_routed_experts);
    TT_FATAL(args.max_dispatch_buffer_token_size > 0, "moe_fanout_reach: max_dispatch_buffer_token_size must be > 0");
    TT_FATAL(
        args.worker_core_range_set.num_cores() > 0,
        "moe_fanout_reach: the op was given no worker cores; the token walk has nowhere to run");
    TT_FATAL(!args.output_mem_config.is_sharded(), "moe_fanout_reach: output memory config must be interleaved");

    const auto& indices = tensor_args.indices_tensor;
    validate_dram_row_major(indices, "indices_tensor");
    TT_FATAL(
        indices.dtype() == tt::tt_metal::DataType::UINT16,
        "moe_fanout_reach: indices must be UINT16, got {}",
        indices.dtype());
    TT_FATAL(
        indices.logical_shape().rank() >= 2,
        "moe_fanout_reach: indices must be at least 2D [.., seq_len, num_experts_per_tok], got {}",
        indices.logical_shape());
    TT_FATAL(
        indices.logical_shape()[-1] == static_cast<int32_t>(args.num_experts_per_tok),
        "moe_fanout_reach: indices last dim is {} but num_experts_per_tok is {}",
        indices.logical_shape()[-1],
        args.num_experts_per_tok);
    // One page per token is what makes a page id a token index. A leading dim above 1 would put a
    // second sequence in the same buffer and the walk would silently cover only the first.
    TT_FATAL(
        indices.logical_volume() == static_cast<uint64_t>(indices.logical_shape()[-2]) * args.num_experts_per_tok,
        "moe_fanout_reach: indices must hold exactly one chip's sequence, [1, .., seq_len, {}], got {}",
        args.num_experts_per_tok,
        indices.logical_shape());

    validate_dram_row_major(tensor_args.expert_dispatch_table_tensor, "expert_dispatch_table");
    TT_FATAL(
        tensor_args.expert_dispatch_table_tensor.dtype() == tt::tt_metal::DataType::INT32,
        "moe_fanout_reach: expert_dispatch_table must be INT32, got {}",
        tensor_args.expert_dispatch_table_tensor.dtype());
    // A trailing sentinel column is allowed and ignored: the kernel refuses an expert id at or above
    // num_routed_experts outright, which is the same answer the sentinel's -1 would have given.
    TT_FATAL(
        tensor_args.expert_dispatch_table_tensor.logical_shape()[-1] >= static_cast<int32_t>(args.num_routed_experts),
        "moe_fanout_reach: expert_dispatch_table last dim is {}, expected at least num_routed_experts = {}",
        tensor_args.expert_dispatch_table_tensor.logical_shape()[-1],
        args.num_routed_experts);
    // This device hosts one dispatch group, so the table is one row and the kernel reads page 0. An
    // unsharded all-groups table would pass the width check above and then be read as group 0's, which
    // routes this chip's tokens by another group's map.
    TT_FATAL(
        tensor_args.expert_dispatch_table_tensor.logical_volume() ==
            static_cast<uint64_t>(tensor_args.expert_dispatch_table_tensor.logical_shape()[-1]),
        "moe_fanout_reach: expert_dispatch_table must be this device's single row, got {}",
        tensor_args.expert_dispatch_table_tensor.logical_shape());

    const auto& offsets = tensor_args.global_dispatch_offsets;
    validate_dram_row_major(offsets, "global_dispatch_offsets");
    TT_FATAL(
        offsets.dtype() == tt::tt_metal::DataType::INT32 || offsets.dtype() == tt::tt_metal::DataType::UINT32,
        "moe_fanout_reach: global_dispatch_offsets must be INT32 or UINT32, got {}",
        offsets.dtype());
    TT_FATAL(
        offsets.logical_shape()[-1] == static_cast<int32_t>(args.num_routed_experts),
        "moe_fanout_reach: global_dispatch_offsets last dim is {} but num_routed_experts is {}",
        offsets.logical_shape()[-1],
        args.num_routed_experts);
    // THIS device's row, not the all-rows table dispatch_fabric2d takes. Reach is a property of the
    // tokens this chip owns, so seeding the allocator from another chip's row would mis-decide every
    // drop -- and a reach table that overstates deadlocks the axis.
    TT_FATAL(
        offsets.logical_volume() == args.num_routed_experts,
        "moe_fanout_reach: global_dispatch_offsets must be THIS device's single row [.., {}], got {}. "
        "That is offset_cumsum's global_dispatch_offsets, not its all-rows table",
        args.num_routed_experts,
        offsets.logical_shape());
}

void MoeFanoutReachDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // A hit that skipped this would answer a caller whose tensors break the op's preconditions with
    // wrong data instead of an error, and wrong data here is a stranded axis downstream.
    validate_on_program_cache_miss(args, tensor_args);
}

MoeFanoutReachDeviceOperation::spec_return_value_t MoeFanoutReachDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& /*tensor_args*/) {
    const uint32_t extent = axis_extent(args);
    // extent / 2 + 2: hops 1..m, the unused 0, and the terminating zero at m + 1 that makes
    // reach[h] - reach[h + 1] the count of tokens whose farthest hop is exactly h, for every h up to m.
    const uint32_t hops = extent / 2 + 2;
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 2, hops}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::INT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            args.output_mem_config));
}

MoeFanoutReachDeviceOperation::topology_return_value_t MoeFanoutReachDeviceOperation::compute_output_topologies(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    const auto& input_topology = tensor_args.indices_tensor.tensor_topology();

    // Per-device on every axis: a row differs from its neighbours along the dispatch axis because the
    // hops are measured from a different position, and across groups because the routing differs.
    ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements;
    for (size_t i = 0; i < input_topology.distribution_shape().dims(); i++) {
        placements.push_back(Shard{static_cast<int>(i)});
    }
    return tt::tt_metal::TensorTopology(input_topology.distribution_shape(), placements, input_topology.mesh_coords());
}

MoeFanoutReachDeviceOperation::tensor_return_value_t MoeFanoutReachDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.indices_tensor.device());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fanout_reach

namespace ttnn::prim {

ttnn::Tensor moe_fanout_reach(
    ttnn::MeshDevice* device,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& expert_dispatch_table_tensor,
    const ttnn::Tensor& global_dispatch_offsets,
    uint32_t num_routed_experts,
    uint32_t num_experts_per_tok,
    uint32_t dispatch_group_size,
    uint32_t max_dispatch_buffer_token_size,
    uint32_t axis,
    const tt::tt_metal::MemoryConfig& memory_config,
    const CoreRangeSet& worker_core_range_set) {
    using namespace ttnn::operations::experimental::deepseek_prefill::moe_fanout_reach;
    using OperationType = MoeFanoutReachDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        MoeFanoutReachParams{
            .device = device,
            .num_routed_experts = num_routed_experts,
            .num_experts_per_tok = num_experts_per_tok,
            .dispatch_group_size = dispatch_group_size,
            .max_dispatch_buffer_token_size = max_dispatch_buffer_token_size,
            .axis = axis,
            .output_mem_config = memory_config,
            .worker_core_range_set = worker_core_range_set},
        MoeFanoutReachInputs{
            .indices_tensor = indices_tensor,
            .expert_dispatch_table_tensor = expert_dispatch_table_tensor,
            .global_dispatch_offsets = global_dispatch_offsets});
}

}  // namespace ttnn::prim
