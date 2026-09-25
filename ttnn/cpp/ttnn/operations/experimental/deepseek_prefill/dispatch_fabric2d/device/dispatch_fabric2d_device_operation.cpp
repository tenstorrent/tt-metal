// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_fabric2d_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"

#include "ttnn/device_operation.hpp"
#include "dispatch_fabric2d_assignments.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

namespace {

// A page index computed on one chip addresses the same page on another chip. That holds only while every
// chip's copy of a buffer starts at the same address, which interleaved allocation on a uniform mesh gives.
void validate_interleaved(const ttnn::Tensor& t, const char* name) {
    TT_FATAL(t.buffer() != nullptr, "dispatch_fabric2d: {} has no device buffer", name);
    TT_FATAL(
        t.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "dispatch_fabric2d: {} must be interleaved; a sender addresses it by page index on another chip",
        name);
}

void validate_interleaved_row_major(const ttnn::Tensor& t, const char* name) {
    TT_FATAL(
        t.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "dispatch_fabric2d: {} must be ROW_MAJOR, got {}",
        name,
        t.layout());
    validate_interleaved(t, name);
}

void validate_control_tensor(const ttnn::Tensor& t, uint32_t num_routed_experts, const char* name) {
    validate_interleaved_row_major(t, name);
    TT_FATAL(
        t.dtype() == tt::tt_metal::DataType::INT32 || t.dtype() == tt::tt_metal::DataType::UINT32,
        "dispatch_fabric2d: {} must be INT32 or UINT32, got {}",
        name,
        t.dtype());
    TT_FATAL(
        t.logical_shape()[-1] == static_cast<int32_t>(num_routed_experts),
        "dispatch_fabric2d: {} last dim is {} but num_routed_experts is {}",
        name,
        t.logical_shape()[-1],
        num_routed_experts);
}

uint32_t axis_extent(const DispatchFabric2dParams& args) {
    return static_cast<uint32_t>(args.device->shape()[args.axis]);
}

}  // namespace

void DispatchFabric2dDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(args.device != nullptr, "dispatch_fabric2d requires a mesh device in attributes");
    TT_FATAL(
        args.axis < args.device->shape().dims(),
        "dispatch_fabric2d: axis {} is out of range for a {} mesh",
        args.axis,
        args.device->shape());
    // A DRAM read needs a 64-byte-aligned L1 destination, and the reader stores the offsets table rows
    // back to back.
    TT_FATAL(
        args.num_routed_experts % 16 == 0,
        "dispatch_fabric2d: num_routed_experts must be a multiple of 16 (got {}); a row of the offsets "
        "table has to be a whole number of 64-byte lines or the per-row DRAM reads are misaligned",
        args.num_routed_experts);
    TT_FATAL(
        args.num_links >= 1 && args.num_links <= 4,
        "dispatch_fabric2d: num_links must be between 1 and 4 (got {})",
        args.num_links);
    TT_FATAL(!args.output_mem_config.is_sharded(), "dispatch_fabric2d: output memory config must be interleaved");

    const uint32_t extent = axis_extent(args);
    TT_FATAL(
        extent >= 4 && extent % 2 == 0,
        "dispatch_fabric2d: axis {} extent {} must be even and at least 4; the schedule splits the "
        "diametrically opposite chip across both directions",
        args.axis,
        extent);
    // The reader tells a bucket index from the BUCKET_NOT_HERE sentinel by value alone.
    TT_FATAL(
        static_cast<uint64_t>(extent) * args.experts_per_chip < dspf2d::BUCKET_NOT_HERE,
        "dispatch_fabric2d: this dispatch group has {} x {} experts, but a bucket index must stay below the "
        "BUCKET_NOT_HERE sentinel",
        extent,
        args.experts_per_chip);
    TT_FATAL(
        ttnn::ccl::is_axis_wrap_wired(*args.device, args.axis),
        "dispatch_fabric2d: axis {} has no closing link, so it is not a ring. This op sends single hops "
        "around one; run it on a topology that wraps that axis (e.g. FABRIC_2D_TORUS_Y or _TORUS_XY), not "
        "a line or mesh.",
        args.axis);

    // A chip that forwards a chunk sizes it from the source chip's row, so every chip needs all rows.
    validate_control_tensor(tensor_args.expert_offsets_tensor, args.num_routed_experts, "expert_offsets");
    TT_FATAL(
        tensor_args.expert_offsets_tensor.logical_shape()[-2] == static_cast<int32_t>(extent),
        "dispatch_fabric2d: expert_offsets second-to-last dim is {}, expected {}. It must be the all-rows "
        "table (offset_cumsum's all_global_dispatch_offsets), replicated along axis {}",
        tensor_args.expert_offsets_tensor.logical_shape()[-2],
        extent,
        args.axis);

    // One row each, identical across the dispatch group. Together they give the end of the last source
    // chip's chunk, which expert_offsets has no row for.
    validate_control_tensor(tensor_args.expert_token_counts, args.num_routed_experts, "expert_token_counts");
    validate_control_tensor(tensor_args.expert_region_offsets, args.num_routed_experts, "expert_region_offsets");

    if (tensor_args.padding_config.has_value()) {
        const auto& pc = *tensor_args.padding_config;
        validate_interleaved_row_major(pc, "padding_config");
        TT_FATAL(
            pc.dtype() == tt::tt_metal::DataType::INT32 || pc.dtype() == tt::tt_metal::DataType::UINT32,
            "dispatch_fabric2d: padding_config must be INT32 or UINT32, got {}",
            pc.dtype());
        TT_FATAL(
            pc.logical_volume() >= 2,
            "dispatch_fabric2d: padding_config holds [real_token_count, pad_side] and so needs at least 2 "
            "elements, got {}",
            pc.logical_volume());
    }

    // Padded tokens look up the extra -1 column, so the kernel needs no bounds check.
    validate_interleaved_row_major(tensor_args.expert_dispatch_table_tensor, "expert_dispatch_table");
    TT_FATAL(
        tensor_args.expert_dispatch_table_tensor.dtype() == tt::tt_metal::DataType::INT32,
        "dispatch_fabric2d: expert_dispatch_table must be INT32, got {}",
        tensor_args.expert_dispatch_table_tensor.dtype());
    TT_FATAL(
        tensor_args.expert_dispatch_table_tensor.logical_shape()[-1] >=
            static_cast<int32_t>(args.num_routed_experts) + 1,
        "dispatch_fabric2d: expert_dispatch_table last dim is {}, expected num_routed_experts + 1 = {}. The "
        "extra column is a -1 sentinel for padded tokens.",
        tensor_args.expert_dispatch_table_tensor.logical_shape()[-1],
        args.num_routed_experts + 1);

    const auto& input = tensor_args.input_tensor;
    validate_interleaved(input, "input_tensor");
    TT_FATAL(
        input.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "dispatch_fabric2d: input must be BFLOAT16, got {}",
        input.dtype());
    TT_FATAL(
        input.layout() == tt::tt_metal::Layout::ROW_MAJOR || input.layout() == tt::tt_metal::Layout::TILE,
        "dispatch_fabric2d: input must be ROW_MAJOR or TILE, got {}",
        input.layout());
    if (input.layout() == tt::tt_metal::Layout::TILE) {
        const uint32_t hidden = static_cast<uint32_t>(input.logical_shape()[-1]);
        TT_FATAL(
            hidden % tt::constants::TILE_WIDTH == 0,
            "dispatch_fabric2d: a TILE input needs emb_dim ({}) to be a multiple of {}",
            hidden,
            tt::constants::TILE_WIDTH);
    }

    // The stream cores and, for a TILE input, the untilizers all come from this set.
    TT_FATAL(
        args.worker_core_range_set.num_cores() >=
            stream_count(args.num_links) + (input.layout() == tt::tt_metal::Layout::TILE ? 1u : 0u),
        "dispatch_fabric2d: the op was given {} worker cores but needs {}{}",
        args.worker_core_range_set.num_cores(),
        stream_count(args.num_links),
        input.layout() == tt::tt_metal::Layout::TILE ? " plus at least one for the untilizer" : "");

    const auto& indices = tensor_args.indices_tensor;
    validate_interleaved_row_major(indices, "indices_tensor");
    TT_FATAL(
        indices.dtype() == tt::tt_metal::DataType::UINT16,
        "dispatch_fabric2d: indices must be UINT16, got {}",
        indices.dtype());
    TT_FATAL(
        indices.logical_shape()[-1] == static_cast<int32_t>(args.num_experts_per_tok),
        "dispatch_fabric2d: indices last dim is {} but num_experts_per_tok is {}",
        indices.logical_shape()[-1],
        args.num_experts_per_tok);
    TT_FATAL(
        indices.logical_shape()[-2] == static_cast<int32_t>(args.seq_len_per_chip),
        "dispatch_fabric2d: indices second-to-last dim is {} but seq_len_per_chip is {}",
        indices.logical_shape()[-2],
        args.seq_len_per_chip);

    TT_FATAL(
        args.metadata_len == 3,
        "dispatch_fabric2d: metadata_len must be 3 (source chip, token index, topk index); got {}",
        args.metadata_len);
    TT_FATAL(args.experts_per_chip > 0, "dispatch_fabric2d: experts_per_chip must be non-zero");
    TT_FATAL(
        args.num_experts_per_tok > 0 && args.num_experts_per_tok <= args.num_routed_experts,
        "dispatch_fabric2d: num_experts_per_tok {} must be in 1..num_routed_experts {}",
        args.num_experts_per_tok,
        args.num_routed_experts);
}

void DispatchFabric2dDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Tensors can change between launches that share a cached program, so check them every time.
    validate_on_program_cache_miss(args, tensor_args);
}

DispatchFabric2dDeviceOperation::spec_return_value_t DispatchFabric2dDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const uint32_t hidden_dim = static_cast<uint32_t>(tensor_args.input_tensor.logical_shape()[-1]);
    const auto layout = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);

    auto payload = tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, args.max_dispatch_buffer_token_size, hidden_dim}),
        tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, layout, args.output_mem_config));
    auto metadata = tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, args.max_dispatch_buffer_token_size, args.metadata_len}),
        tt::tt_metal::TensorLayout(tt::tt_metal::DataType::INT32, layout, args.output_mem_config));
    return {payload, metadata};
}

DispatchFabric2dDeviceOperation::topology_return_value_t DispatchFabric2dDeviceOperation::compute_output_topologies(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    const auto& input_topology = tensor_args.input_tensor.tensor_topology();

    // A chip's outputs hold the tokens for the experts it hosts, so they differ on every mesh axis.
    ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements;
    for (size_t i = 0; i < input_topology.distribution_shape().dims(); i++) {
        placements.push_back(Shard{static_cast<int>(i)});
    }
    auto topology =
        tt::tt_metal::TensorTopology(input_topology.distribution_shape(), placements, input_topology.mesh_coords());
    return {topology, topology};
}

DispatchFabric2dDeviceOperation::tensor_return_value_t DispatchFabric2dDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(args, tensor_args);
    return {
        create_device_tensor(specs[0], tensor_args.input_tensor.device()),
        create_device_tensor(specs[1], tensor_args.input_tensor.device())};
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d

namespace ttnn::prim {

std::array<ttnn::Tensor, 2> dispatch_fabric2d(
    ttnn::MeshDevice* device,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& expert_offsets_tensor,
    const ttnn::Tensor& expert_dispatch_table_tensor,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& expert_region_offsets,
    const std::optional<ttnn::Tensor>& padding_config,
    uint32_t experts_per_chip,
    uint32_t num_routed_experts,
    uint32_t num_experts_per_tok,
    uint32_t metadata_len,
    uint32_t max_dispatch_buffer_token_size,
    uint32_t seq_len_per_chip,
    uint32_t axis,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const tt::tt_metal::MemoryConfig& memory_config,
    const CoreRangeSet& worker_core_range_set) {
    using namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d;
    using OperationType = DispatchFabric2dDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        DispatchFabric2dParams{
            .device = device,
            .experts_per_chip = experts_per_chip,
            .num_routed_experts = num_routed_experts,
            .num_experts_per_tok = num_experts_per_tok,
            .metadata_len = metadata_len,
            .max_dispatch_buffer_token_size = max_dispatch_buffer_token_size,
            .seq_len_per_chip = seq_len_per_chip,
            .axis = axis,
            .num_links = num_links,
            .has_padding_config = padding_config.has_value(),
            .topology = topology,
            .output_mem_config = memory_config,
            .worker_core_range_set = worker_core_range_set},
        DispatchFabric2dInputs{
            .input_tensor = input_tensor,
            .indices_tensor = indices_tensor,
            .expert_offsets_tensor = expert_offsets_tensor,
            .expert_dispatch_table_tensor = expert_dispatch_table_tensor,
            .expert_token_counts = expert_token_counts,
            .expert_region_offsets = expert_region_offsets,
            .padding_config = padding_config});
}

}  // namespace ttnn::prim
