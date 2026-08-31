// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/vsa_sdpa_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <algorithm>
#include <bit>

namespace ttnn::prim {

namespace {
constexpr uint32_t kQChunkTokens = 64;  // one query tile = 64 tokens = one index row (fixed by the VSA contract)

void validate_common(const VsaSdpaParams& attrs, const VsaSdpaInputs& t) {
    const auto& q = t.q;
    const auto& k = t.k;
    const auto& v = t.v;
    const auto& idx = t.indices;
    const auto& counts = t.block_counts;
    TT_FATAL(
        q.device() == k.device() && q.device() == v.device() && q.device() == idx.device() &&
            q.device() == counts.device(),
        "vsa_sdpa: all inputs must be on the same device");
    for (const Tensor* tp : {&q, &k, &v}) {
        TT_FATAL(tp->layout() == Layout::TILE, "vsa_sdpa q/k/v must be TILE");
        TT_FATAL(tp->memory_config().buffer_type() == BufferType::DRAM, "vsa_sdpa q/k/v must be in DRAM");
        TT_FATAL(!tp->memory_config().is_sharded(), "vsa_sdpa q/k/v must be interleaved");
        TT_FATAL(tp->padded_shape() == tp->logical_shape(), "vsa_sdpa q/k/v must not be padded");
    }
    for (const Tensor* tp : {&idx, &counts}) {
        TT_FATAL(tp->layout() == Layout::ROW_MAJOR, "vsa_sdpa indices/block_counts must be ROW_MAJOR");
        TT_FATAL(tp->memory_config().buffer_type() == BufferType::DRAM, "vsa_sdpa indices/block_counts must be in DRAM");
        TT_FATAL(!tp->memory_config().is_sharded(), "vsa_sdpa indices/block_counts must be interleaved");
    }
    const auto qs = q.logical_shape();
    const auto ks = k.logical_shape();
    const auto vs = v.logical_shape();
    const auto is = idx.logical_shape();
    const auto cs = counts.logical_shape();
    TT_FATAL(qs.rank() == 4 && qs[0] == 1, "q must be [1,H,S,d]");
    const uint32_t H = qs[1];
    const uint32_t S = qs[2];
    const uint32_t d = qs[3];
    TT_FATAL(S > 0 && S % kQChunkTokens == 0, "q sequence length ({}) must be a positive multiple of 64", S);
    TT_FATAL(ks.rank() == 4 && ks[0] == 1 && ks[1] == H && ks[3] == d, "k must be [1,H,T,d] matching q's H and d");
    TT_FATAL(vs == ks, "v shape must equal k shape (got {} vs {})", vs, ks);
    const uint32_t T = ks[2];
    TT_FATAL(
        attrs.block_size > 0 && T % attrs.block_size == 0,
        "block_size ({}) must divide T ({})",
        attrs.block_size,
        T);
    const uint32_t n_kv_blocks = T / attrs.block_size;
    const uint32_t n_q_tiles = S / kQChunkTokens;
    TT_FATAL(
        is.rank() == 4 && is[0] == 1 && is[1] == H && is[2] == n_q_tiles,
        "indices must be [1,H,S/64,W] (got {} for H {}, S/64 {})",
        is,
        H,
        n_q_tiles);
    const uint32_t W = is[3];
    TT_FATAL(W >= n_kv_blocks, "indices width ({}) must be >= T/block_size ({})", W, n_kv_blocks);
    TT_FATAL(
        cs.rank() == 4 && cs[0] == 1 && cs[1] == 1 && cs[2] == 1 && cs[3] == W,
        "block_counts must be [1,1,1,W] matching the indices width (got {} for W {})",
        cs,
        W);
}
}  // namespace

void VsaSdpaOperation::validate_on_program_cache_hit(const VsaSdpaParams& attrs, const VsaSdpaInputs& t) {
    validate_common(attrs, t);
}

void VsaSdpaOperation::validate_on_program_cache_miss(const VsaSdpaParams& attrs, const VsaSdpaInputs& t) {
    TT_FATAL(tt::tt_metal::hal::get_arch() == tt::ARCH::BLACKHOLE, "vsa_sdpa is Blackhole-only");
    TT_FATAL(t.q.dtype() == DataType::BFLOAT16, "vsa_sdpa: q must be bf16");
    TT_FATAL(
        t.k.dtype() == DataType::BFLOAT16 || t.k.dtype() == DataType::BFLOAT8_B, "vsa_sdpa: k must be bf16 or bfp8_b");
    TT_FATAL(
        t.v.dtype() == DataType::BFLOAT16 || t.v.dtype() == DataType::BFLOAT8_B, "vsa_sdpa: v must be bf16 or bfp8_b");
    TT_FATAL(t.indices.dtype() == DataType::UINT32, "indices must be uint32");
    TT_FATAL(t.block_counts.dtype() == DataType::UINT32, "block_counts must be uint32");

    validate_common(attrs, t);

    constexpr uint32_t tile_w = tt::constants::TILE_WIDTH;
    const uint32_t d = t.q.logical_shape()[3];
    TT_FATAL(d % tile_w == 0, "d must be a multiple of {} (got {})", tile_w, d);
    TT_FATAL(
        attrs.block_size % (2 * tile_w) == 0,
        "block_size must be a multiple of {} (got {})",
        2 * tile_w,
        attrs.block_size);
    TT_FATAL(attrs.k_chunk_blocks >= 1, "k_chunk_blocks (m) must be >= 1 (got {})", attrs.k_chunk_blocks);
    TT_FATAL(attrs.scale > 0.0f, "scale must be > 0");

    // Row-byte alignment for the ROW-MAJOR index/count row DMAs. Pad W up with sentinels/zeros to satisfy this.
    const uint32_t dram_align = tt::tt_metal::hal::get_dram_alignment();
    const uint32_t W = t.indices.logical_shape()[3];
    TT_FATAL(
        (W * t.indices.element_size()) % dram_align == 0,
        "indices row bytes ({} * 4) must be {}B-aligned; pad W with 0xFFFFFFFF sentinels",
        W,
        dram_align);
}

VsaSdpaOperation::spec_return_value_t VsaSdpaOperation::compute_output_specs(
    const VsaSdpaParams& /*attrs*/, const VsaSdpaInputs& t) {
    // Output matches q: [1, H, S, d] TILE bf16, DRAM interleaved.
    const tt::tt_metal::MemoryConfig out_mem{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    return tt::tt_metal::TensorSpec(
        t.q.logical_shape(), tt::tt_metal::TensorLayout(t.q.dtype(), tt::tt_metal::PageConfig(Layout::TILE), out_mem));
}

VsaSdpaOperation::tensor_return_value_t VsaSdpaOperation::create_output_tensors(
    const VsaSdpaParams& attrs, const VsaSdpaInputs& t) {
    return create_device_tensor(compute_output_specs(attrs, t), t.q.device());
}

ttsl::hash::hash_t VsaSdpaOperation::compute_program_hash(const VsaSdpaParams& attrs, const VsaSdpaInputs& t) {
    // Every shape is hashed: the kernels bake head strides and the index width as compile-time args.
    return tt::tt_metal::operation::hash_operation<VsaSdpaOperation>(
        std::bit_cast<uint32_t>(attrs.scale),
        attrs.block_size,
        attrs.k_chunk_blocks,
        attrs.compute_kernel_config,
        t.q.logical_shape(),
        t.q.dtype(),
        t.k.logical_shape(),
        t.k.dtype(),
        t.v.dtype(),
        t.indices.logical_shape());
}

VsaSdpaOperation::DispatchArgs VsaSdpaOperation::compute_dispatch_args(
    const VsaSdpaParams& /*attrs*/, const VsaSdpaInputs& t) {
    const uint32_t H = t.q.logical_shape()[1];
    const uint32_t S = t.q.logical_shape()[2];
    const tt::tt_metal::CoreCoord grid = t.q.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = grid.x * grid.y;
    const uint32_t total_work = H * (S / kQChunkTokens);
    return DispatchArgs{
        .grid = grid,
        .num_cores = num_cores,
        .base_work = total_work / num_cores,
        .extra = total_work % num_cores,
    };
}

void VsaSdpaOperation::VsaSdpaProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const VsaSdpaParams& attrs,
    const VsaSdpaInputs& t,
    Tensor& tensor_return_value) {
    // Kernel push order in create_descriptor(): reader(0), writer(1), compute(2).
    constexpr uint32_t kReaderKernelIdx = 0;
    constexpr uint32_t kWriterKernelIdx = 1;
    constexpr uint32_t kComputeKernelIdx = 2;

    const auto dyn = compute_dispatch_args(attrs, t);
    const uint32_t q_addr = t.q.buffer()->address();
    const uint32_t k_addr = t.k.buffer()->address();
    const uint32_t v_addr = t.v.buffer()->address();
    const uint32_t idx_addr = t.indices.buffer()->address();
    const uint32_t counts_addr = t.block_counts.buffer()->address();
    const uint32_t out_addr = tensor_return_value.buffer()->address();

    for (uint32_t i = 0; i < dyn.num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i % dyn.grid.x, i / dyn.grid.x};
        const uint32_t work_start = i * dyn.base_work + std::min(i, dyn.extra);
        const uint32_t work_count = dyn.base_work + (i < dyn.extra ? 1u : 0u);

        auto& reader = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx, core);
        TT_FATAL(
            reader.size() == kReaderArgCount,
            "vsa_sdpa reader expected {} runtime args, cached program has {}",
            static_cast<uint32_t>(kReaderArgCount),
            reader.size());
        reader[kReaderQAddr] = q_addr;
        reader[kReaderKAddr] = k_addr;
        reader[kReaderVAddr] = v_addr;
        reader[kReaderIdxAddr] = idx_addr;
        reader[kReaderCountsAddr] = counts_addr;
        reader[kReaderWorkStart] = work_start;
        reader[kReaderWorkCount] = work_count;

        auto& writer = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx, core);
        TT_FATAL(
            writer.size() == kWriterArgCount,
            "vsa_sdpa writer expected {} runtime args, cached program has {}",
            static_cast<uint32_t>(kWriterArgCount),
            writer.size());
        writer[kWriterOutAddr] = out_addr;
        writer[kWriterKAddr] = k_addr;
        writer[kWriterVAddr] = v_addr;
        writer[kWriterWorkStart] = work_start;
        writer[kWriterWorkCount] = work_count;

        auto& compute = tt::tt_metal::GetRuntimeArgs(program, kComputeKernelIdx, core);
        TT_FATAL(
            compute.size() == kComputeArgCount,
            "vsa_sdpa compute expected {} runtime args, cached program has {}",
            static_cast<uint32_t>(kComputeArgCount),
            compute.size());
        compute[kComputeWorkStart] = work_start;
        compute[kComputeWorkCount] = work_count;
    }
}

Tensor vsa_sdpa(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& indices,
    const Tensor& block_counts,
    float scale,
    uint32_t block_size,
    uint32_t k_chunk_blocks,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    using OperationType = ttnn::prim::VsaSdpaOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .scale = scale,
            .block_size = block_size,
            .k_chunk_blocks = k_chunk_blocks,
            .compute_kernel_config = compute_kernel_config,
        },
        OperationType::tensor_args_t{
            .q = q,
            .k = k,
            .v = v,
            .indices = indices,
            .block_counts = block_counts,
        });
}

}  // namespace ttnn::prim
