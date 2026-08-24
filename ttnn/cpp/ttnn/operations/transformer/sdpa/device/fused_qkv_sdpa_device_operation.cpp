// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/fused_qkv_sdpa_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/device.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// Heads sit side by side inside a row, so a head's slice must start and end on a tile boundary for
// the reader's window to be pure address arithmetic. Everything else here follows from that.
uint32_t head_dim_of(const FusedQKVSDPAParams& attrs, const Tensor& qkv) {
    const uint32_t fused_width = qkv.logical_shape()[3];
    TT_FATAL(attrs.num_heads > 0, "fused_qkv_sdpa requires num_heads > 0");
    TT_FATAL(
        fused_width % (3 * attrs.num_heads) == 0,
        "Fused qkv width {} must split evenly into 3 x num_heads ({})",
        fused_width,
        attrs.num_heads);
    return fused_width / (3 * attrs.num_heads);
}

}  // namespace

void FusedQKVSDPAOperation::validate_on_program_cache_miss(
    const FusedQKVSDPAParams& attrs, const FusedQKVSDPAInputs& tensors) {
    const Tensor& qkv = tensors.qkv;

    TT_FATAL(qkv.storage_type() == StorageType::DEVICE, "fused_qkv_sdpa requires the qkv tensor on device");
    TT_FATAL(qkv.buffer() != nullptr, "fused_qkv_sdpa requires the qkv tensor to be allocated");
    TT_FATAL(qkv.layout() == Layout::TILE, "fused_qkv_sdpa requires a tile-layout qkv tensor");
    // The reader addresses qkv by tile id through a TensorAccessor, which does not care whether the
    // pages sit in DRAM or L1 -- only that they are interleaved rather than sharded.
    TT_FATAL(
        !qkv.memory_config().is_sharded(), "fused_qkv_sdpa reads qkv by tile id, so it requires an interleaved tensor");

    const auto& shape = qkv.logical_shape();
    TT_FATAL(shape.rank() == 4, "fused_qkv_sdpa expects a rank-4 qkv tensor, got rank {}", shape.rank());
    TT_FATAL(shape[1] == 1, "fused_qkv_sdpa expects the head axis folded into the last dim, got dim1 = {}", shape[1]);

    const uint32_t head_dim = head_dim_of(attrs, qkv);
    TT_FATAL(
        head_dim % tt::constants::TILE_WIDTH == 0,
        "fused_qkv_sdpa needs head_dim ({}) to be a multiple of the tile width ({}); otherwise a head's slice "
        "does not start on a tile boundary and the split is a real transpose rather than address arithmetic",
        head_dim,
        tt::constants::TILE_WIDTH);

    if (tensors.attn_mask.has_value()) {
        const auto& mask = tensors.attn_mask.value();
        TT_FATAL(mask.storage_type() == StorageType::DEVICE, "fused_qkv_sdpa requires the mask on device");
        TT_FATAL(mask.layout() == Layout::TILE, "fused_qkv_sdpa requires a tile-layout mask");
    }

    if (attrs.program_config.has_value()) {
        const auto& pc = attrs.program_config.value();
        const uint32_t seq_len = shape[2];
        TT_FATAL(
            pc.q_chunk_size % tt::constants::TILE_HEIGHT == 0 && pc.k_chunk_size % tt::constants::TILE_HEIGHT == 0,
            "fused_qkv_sdpa requires q_chunk_size ({}) and k_chunk_size ({}) to be tile-height multiples",
            pc.q_chunk_size,
            pc.k_chunk_size);
        TT_FATAL(
            seq_len % pc.q_chunk_size == 0 && seq_len % pc.k_chunk_size == 0,
            "fused_qkv_sdpa does not pad the sequence: seq_len {} must divide by q_chunk_size {} and k_chunk_size {}",
            seq_len,
            pc.q_chunk_size,
            pc.k_chunk_size);
    }
}

FusedQKVSDPAOperation::spec_return_value_t FusedQKVSDPAOperation::compute_output_specs(
    const FusedQKVSDPAParams& attrs, const FusedQKVSDPAInputs& tensors) {
    const auto& qkv = tensors.qkv;
    const auto& in_shape = qkv.logical_shape();
    const ttnn::Shape shape({in_shape[0], attrs.num_heads, in_shape[2], head_dim_of(attrs, qkv)});
    return tt::tt_metal::TensorSpec(
        shape, TensorLayout(qkv.dtype(), PageConfig(Layout::TILE), attrs.output_mem_config));
}

FusedQKVSDPAOperation::tensor_return_value_t FusedQKVSDPAOperation::create_output_tensors(
    const FusedQKVSDPAParams& attrs, const FusedQKVSDPAInputs& tensors) {
    return create_device_tensor(compute_output_specs(attrs, tensors), tensors.qkv.device());
}

Tensor fused_qkv_sdpa(
    const Tensor& input_tensor_qkv,
    const std::optional<Tensor>& attn_mask,
    uint32_t num_heads,
    std::optional<float> scale,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    using OperationType = ttnn::prim::FusedQKVSDPAOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .num_heads = num_heads,
            .scale = scale,
            .output_mem_config = output_mem_config,
            .program_config = std::move(program_config),
            .compute_kernel_config = compute_kernel_config,
        },
        OperationType::tensor_args_t{
            .qkv = input_tensor_qkv,
            .attn_mask = attn_mask,
        });
}

}  // namespace ttnn::prim
