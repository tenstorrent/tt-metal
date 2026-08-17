// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include <set>
#include <string>

#include "combine_fabric2d_device_operation.hpp"
#include "kernels/dataflow/combine_fabric2d_kernel_interface.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

namespace {

void validate_dram_row_major(const ttnn::Tensor& t, const char* tensor_name) {
    TT_FATAL(t.storage_type() == ttnn::StorageType::DEVICE, "combine_fabric2d: {} must be on device", tensor_name);
    TT_FATAL(
        t.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM &&
            t.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "combine_fabric2d: {} must be an interleaved DRAM tensor",
        tensor_name);
    TT_FATAL(
        t.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "combine_fabric2d: {} must be ROW_MAJOR so one row is exactly one page",
        tensor_name);
}

// The 32-bit control tensors carry page indices and counts, so either signedness is fine — the values are
// non-negative by construction and the kernels read them as uint32.
void validate_control_tensor(
    const ttnn::Tensor& t, uint32_t rows, uint32_t num_routed_experts, const char* tensor_name) {
    validate_dram_row_major(t, tensor_name);
    TT_FATAL(
        t.dtype() == tt::tt_metal::DataType::INT32 || t.dtype() == tt::tt_metal::DataType::UINT32,
        "combine_fabric2d: {} must be INT32 or UINT32, got {}",
        tensor_name,
        t.dtype());
    const auto shape = t.logical_shape();
    TT_FATAL(
        shape[-1] == static_cast<int32_t>(num_routed_experts),
        "combine_fabric2d: {} last dim is {} but num_routed_experts is {}",
        tensor_name,
        shape[-1],
        num_routed_experts);
    TT_FATAL(
        shape[-2] == static_cast<int32_t>(rows),
        "combine_fabric2d: {} second-to-last dim is {}, expected {}. expert_offsets must be REPLICATED "
        "along the dispatch-group axis (each chip needs every origin chip's run boundaries, not just its "
        "own); the counts and region offsets are already identical across that axis, so they arrive with a "
        "single row.",
        tensor_name,
        shape[-2],
        rows);
}

}  // namespace

void CombineFabric2dDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(args.device != nullptr, "combine_fabric2d requires a mesh device in attributes");
    TT_FATAL(args.num_links >= 1 && args.num_links <= 4, "num_links must be between 1 and 4 (got {})", args.num_links);
    TT_FATAL(
        !args.init_zeros,
        "combine_fabric2d: init_zeros=true is not implemented. Output slots with no expert contribution are "
        "left as allocated, exactly as the production op leaves them with init_zeros=false.");
    TT_FATAL(
        !args.output_mem_config.is_sharded(),
        "combine_fabric2d: output memory config must be interleaved, not sharded");

    TT_FATAL(
        args.axis < args.device->shape().dims(),
        "combine_fabric2d: axis {} is out of range for a {} mesh",
        args.axis,
        args.device->shape());
    const uint32_t extent = args.device->shape()[static_cast<int32_t>(args.axis)];
    TT_FATAL(
        extent >= 3,
        "combine_fabric2d: axis {} extent {} needs 3+ chips for two distinct neighbours",
        args.axis,
        extent);
    TT_FATAL(
        extent % 2 == 0,
        "combine_fabric2d: axis {} extent {} must be even (the opposite chip is split between the two directions)",
        args.axis,
        extent);
    TT_FATAL(
        args.experts_per_chip >= 1, "combine_fabric2d: experts_per_chip must be >= 1 (got {})", args.experts_per_chip);
    TT_FATAL(
        args.num_experts_per_tok >= 1,
        "combine_fabric2d: num_experts_per_tok must be >= 1 (got {})",
        args.num_experts_per_tok);
    TT_FATAL(
        args.seq_len_per_chip >= 1, "combine_fabric2d: seq_len_per_chip must be >= 1 (got {})", args.seq_len_per_chip);

    // ---- Token data. Page = one token, so the last dim is the embedding and the rest is the flat slot
    // index. BFLOAT16 only: the fp8 and TILE paths both need the untilize stage this op does not have.
    const auto& buf = tensor_args.dispatched_buffer;
    validate_dram_row_major(buf, "dispatched_buffer");
    // A token is one page — the op does not take a token size, it reads it off the tensor the caller staged.
    const uint32_t token_page = static_cast<uint32_t>(buf.buffer()->aligned_page_size());
    // A token is the payload of a NoC transfer, which needs 16-byte alignment.
    TT_FATAL(
        token_page % 16 == 0,
        "combine_fabric2d: token page {} B must be 16-byte aligned for NoC transfers",
        token_page);
    // Token + metadata is a DRAM page of the forwarding buffer, which needs DRAM alignment.
    TT_FATAL(
        (token_page + cmbf2d::FORWARDING_METADATA_SIZE) % 64 == 0,
        "combine_fabric2d: token page {} B + {} B of forwarding metadata must be 64-byte aligned for DRAM",
        token_page,
        cmbf2d::FORWARDING_METADATA_SIZE);
    // A forwarded packet carries the token PLUS its routing tail, so the payload the fabric must accept is
    // token + tail, not just token.
    TT_FATAL(
        token_page + cmbf2d::FORWARDING_METADATA_SIZE <= tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(),
        "combine_fabric2d: token page {} B + {} B routing tail exceeds the fabric max payload {}. Raise "
        "max_payload_size in the device's fabric_router_config.",
        token_page,
        cmbf2d::FORWARDING_METADATA_SIZE,
        tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    TT_FATAL(
        buf.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "combine_fabric2d: dispatched_buffer must be BFLOAT16, got {}. The BFLOAT8_B/TILE path needs an "
        "untilize stage this op does not have.",
        buf.dtype());
    const auto buf_shape = buf.logical_shape();
    TT_FATAL(buf_shape.rank() >= 2, "combine_fabric2d: dispatched_buffer must be rank 2 or more");

    const auto& meta = tensor_args.dispatched_metadata;
    validate_dram_row_major(meta, "dispatched_metadata");
    TT_FATAL(
        meta.dtype() == tt::tt_metal::DataType::INT32,
        "combine_fabric2d: dispatched_metadata must be INT32, got {}",
        meta.dtype());
    const auto meta_shape = meta.logical_shape();
    TT_FATAL(
        meta_shape[-1] == 3,
        "combine_fabric2d: dispatched_metadata last dim is {}, expected 3 (linearized_coord, token_idx, "
        "topk_idx). The fp8 scale tail is not supported.",
        meta_shape[-1]);
    TT_FATAL(
        meta_shape[-2] == buf_shape[-2],
        "combine_fabric2d: dispatched_metadata holds {} slots but dispatched_buffer holds {}; they index the "
        "same flat buffer so the two must match",
        meta_shape[-2],
        buf_shape[-2]);

    // ---- Control tensors. num_routed_experts comes from the counts' width, exactly as production derives it.
    const uint32_t num_routed_experts = static_cast<uint32_t>(tensor_args.expert_token_counts.logical_shape()[-1]);
    TT_FATAL(
        num_routed_experts % args.experts_per_chip == 0,
        "combine_fabric2d: num_routed_experts {} must be divisible by experts_per_chip {}",
        num_routed_experts,
        args.experts_per_chip);
    const uint32_t experts_per_group = args.experts_per_chip * extent;
    TT_FATAL(
        num_routed_experts >= experts_per_group,
        "combine_fabric2d: num_routed_experts {} is fewer than experts_per_chip x ring extent = {}, so "
        "there is not even one dispatch group",
        num_routed_experts,
        experts_per_group);
    TT_FATAL(
        num_routed_experts % experts_per_group == 0,
        "combine_fabric2d: num_routed_experts {} must be divisible by experts_per_chip x ring extent "
        "= {}",
        num_routed_experts,
        experts_per_group);
    validate_control_tensor(tensor_args.expert_token_counts, 1, num_routed_experts, "expert_token_counts");
    validate_control_tensor(tensor_args.expert_region_offsets, 1, num_routed_experts, "expert_region_offsets");
    validate_control_tensor(tensor_args.expert_offsets, extent, num_routed_experts, "expert_offsets");
}

void CombineFabric2dDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t&) {}

CombineFabric2dDeviceOperation::spec_return_value_t CombineFabric2dDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // one page per (token, top-k slot), the embedding along the last dim.
    const uint32_t emb_dim = static_cast<uint32_t>(tensor_args.dispatched_buffer.logical_shape()[-1]);
    const ttnn::Shape output_shape({1, 1, args.seq_len_per_chip, args.num_experts_per_tok, emb_dim});
    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            args.output_mem_config));
}

CombineFabric2dDeviceOperation::tensor_return_value_t CombineFabric2dDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.dispatched_buffer.device());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d

namespace ttnn::prim {
ttnn::Tensor combine_fabric2d(
    ttnn::MeshDevice* device,
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& dispatched_metadata,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_offsets,
    uint32_t experts_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    uint32_t axis,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const tt::tt_metal::MemoryConfig& memory_config,
    bool init_zeros) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::CombineFabric2dDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .device = device,
            .experts_per_chip = experts_per_chip,
            .num_experts_per_tok = num_experts_per_tok,
            .seq_len_per_chip = seq_len_per_chip,
            .axis = axis,
            .num_links = num_links,
            .topology = topology,
            .output_mem_config = memory_config,
            .init_zeros = init_zeros},
        OperationType::tensor_args_t{
            .dispatched_buffer = dispatched_buffer,
            .dispatched_metadata = dispatched_metadata,
            .expert_token_counts = expert_token_counts,
            .expert_region_offsets = expert_region_offsets,
            .expert_offsets = expert_offsets});
}
}  // namespace ttnn::prim
