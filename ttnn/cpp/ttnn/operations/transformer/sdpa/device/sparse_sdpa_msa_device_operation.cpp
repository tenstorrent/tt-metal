// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_msa_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>  // GetRuntimeArgs (override_runtime_arguments patches per-program args)
#include <bit>

namespace ttnn::prim {

namespace {
// Re-check invariants excluded from the program hash. Interleaved K/V shape fields (T, batch slots, and n_kv)
// and cache_batch_idx may vary on a cache hit; tensor layout, padding, memory placement, and device must still
// match kernel assumptions.
void validate_non_hashed(const SparseSDPAMsaParams& attrs, const SparseSDPAMsaInputs& t) {
    const auto& q = t.q;
    const auto& k = t.k;
    const auto& v = t.v;
    const auto& idx = t.indices;
    TT_FATAL(
        q.device() == k.device() && q.device() == v.device() && q.device() == idx.device(),
        "sparse_sdpa_msa: all inputs must be on the same device");
    // q and indices: ROW_MAJOR, DRAM, interleaved, unpadded.
    for (const Tensor* tp : {&q, &idx}) {
        TT_FATAL(tp->layout() == Layout::ROW_MAJOR, "sparse_sdpa_msa q/indices must be ROW_MAJOR");
        TT_FATAL(tp->memory_config().buffer_type() == BufferType::DRAM, "sparse_sdpa_msa q/indices must be in DRAM");
        TT_FATAL(!tp->memory_config().is_sharded(), "sparse_sdpa_msa q/indices must be interleaved");
        TT_FATAL(tp->padded_shape() == tp->logical_shape(), "sparse_sdpa_msa q/indices must not be padded");
    }
    // K/V are pre-tiled DRAM caches. They may be interleaved or ND-sharded.
    for (const Tensor* tp : {&k, &v}) {
        TT_FATAL(tp->layout() == Layout::TILE, "sparse_sdpa_msa k/v must be TILE (pre-tiled cache)");
        TT_FATAL(tp->memory_config().buffer_type() == BufferType::DRAM, "sparse_sdpa_msa k/v must be in DRAM");
        TT_FATAL(tp->padded_shape() == tp->logical_shape(), "sparse_sdpa_msa k/v must not be padded");
    }
    const auto qs = q.logical_shape();
    const auto is = idx.logical_shape();
    TT_FATAL(qs.rank() == 4 && qs[0] == 1, "q must be [1,H,S,d]");
    const uint32_t H = qs[1];
    const uint32_t S = qs[2];
    const uint32_t d = q.logical_shape()[3];
    const auto ks = k.logical_shape();
    const auto vs = v.logical_shape();
    TT_FATAL(ks.rank() == 4 && ks[3] == d, "k must be [B,n_kv,T,d] with d matching q ({})", d);
    TT_FATAL(
        vs.rank() == 4 && vs[0] == ks[0] && vs[1] == ks[1] && vs[2] == ks[2],
        "v must be [B,n_kv,T,v_dim] matching k's B/n_kv/T");
    // v_dim is compiled into all kernels and included in the program hash.
    TT_FATAL(
        vs[3] > 0 && vs[3] % tt::constants::TILE_WIDTH == 0,
        "v_dim (v last dim) must be a positive multiple of {} (got {})",
        tt::constants::TILE_WIDTH,
        vs[3]);
    TT_FATAL(ks[1] > 0, "n_kv must be > 0");
    TT_FATAL(ks[2] > 0, "k/v T (cache length) must be > 0");
    const uint32_t n_kv = ks[1];
    TT_FATAL(H % n_kv == 0, "sparse_sdpa_msa: H ({}) must be divisible by n_kv ({})", H, n_kv);
    const uint32_t heads_per_kv = H / n_kv;
    constexpr uint32_t tile_h = tt::constants::TILE_HEIGHT;
    // Each KV group computes full 32-head tile rows. 16 heads/group is padded internally for the production
    // TP-shard and single-chip GQA cases; larger per-group head counts must already be full tiles.
    TT_FATAL(
        heads_per_kv == 16 || (heads_per_kv % tile_h == 0 && heads_per_kv >= tile_h),
        "sparse_sdpa_msa: H / n_kv must be 16 or a multiple of {} (got H {}, n_kv {}, H/n_kv {})",
        tile_h,
        H,
        n_kv,
        heads_per_kv);
    TT_FATAL(is.rank() == 4 && is[0] == 1 && is[1] == n_kv && is[2] == S, "indices must be [1,n_kv,S,TOPK]");
    TT_FATAL(S > 0 && is[3] > 0, "S/TOPK must be > 0");
    TT_FATAL(
        attrs.block_size > 0 && ks[2] % attrs.block_size == 0,
        "block_size must divide T (got block_size {}, T {})",
        attrs.block_size,
        ks[2]);
    const uint32_t B = ks[0];
    if (attrs.cache_batch_idx.has_value()) {
        TT_FATAL(
            attrs.cache_batch_idx.value() < B,
            "cache_batch_idx ({}) must be < kv batch slots ({})",
            attrs.cache_batch_idx.value(),
            B);
    } else {
        TT_FATAL(B == 1, "k/v batch must be 1 unless cache_batch_idx is set (got {})", B);
    }
}
}  // namespace

void SparseSDPAMsaOperation::validate_on_program_cache_hit(
    const SparseSDPAMsaParams& attrs, const SparseSDPAMsaInputs& t) {
    validate_non_hashed(attrs, t);
}

void SparseSDPAMsaOperation::validate_on_program_cache_miss(
    const SparseSDPAMsaParams& attrs, const SparseSDPAMsaInputs& t) {
    const auto& q = t.q;
    const auto& k = t.k;
    const auto& v = t.v;
    const auto& idx = t.indices;

    TT_FATAL(tt::tt_metal::hal::get_arch() == tt::ARCH::BLACKHOLE, "sparse_sdpa_msa is Blackhole-only");

    // q is bf16 or fp8_e4m3. K/V are tiled bf16 or bfp8_b. Indices are uint32 block ids.
    const bool q_is_fp8 = (q.dtype() == DataType::FP8_E4M3);
    TT_FATAL(q.dtype() == DataType::BFLOAT16 || q_is_fp8, "sparse_sdpa_msa: q must be bf16 or fp8_e4m3");
    // fp8 Q is silently inaccurate with the token-level causal mask (fp8-specific; root cause not yet identified)
    TT_FATAL(
        !(attrs.causal_enabled() && q_is_fp8),
        "sparse_sdpa_msa: causal masking (chunk_start_idx) with fp8_e4m3 q is not supported; use bf16 q");
    TT_FATAL(
        k.dtype() == DataType::BFLOAT16 || k.dtype() == DataType::BFLOAT8_B,
        "sparse_sdpa_msa: k must be bf16 or bfp8_b");
    TT_FATAL(
        v.dtype() == DataType::BFLOAT16 || v.dtype() == DataType::BFLOAT8_B,
        "sparse_sdpa_msa: v must be bf16 or bfp8_b");
    TT_FATAL(idx.dtype() == DataType::UINT32, "indices must be uint32");
    // fp8 Q tilize requires a 32-bit DEST accumulator.
    TT_FATAL(
        !q_is_fp8 || get_fp32_dest_acc_en(attrs.compute_kernel_config),
        "fp8 q requires fp32_dest_acc_en=true (32-bit DEST for the fp8 tilize)");

    validate_non_hashed(attrs, t);

    const auto qs = q.logical_shape();
    const auto is = idx.logical_shape();
    const uint32_t d = qs[3];
    const uint32_t v_dim = v.logical_shape()[3];
    const uint32_t TOPK = is[3];

    constexpr uint32_t tile_w = tt::constants::TILE_WIDTH;
    TT_FATAL(d % tile_w == 0, "d (q/k last dim) must be a multiple of {} (got {})", tile_w, d);
    // v_dim positivity and tile_w alignment are checked on hits and misses.

    // block_size: one chunk == one block of block_size contiguous token rows; must tile the key axis and divide T.
    TT_FATAL(
        attrs.block_size >= tile_w && attrs.block_size % tile_w == 0,
        "block_size must be a multiple of {} (got {})",
        tile_w,
        attrs.block_size);
    TT_FATAL(attrs.scale > 0.0f, "scale must be > 0");

    // Row-byte alignment for the ROW-MAJOR DMAs (q rows, index rows, output rows). K/V are TILE tensors (the
    // reader reads whole tiles, which are inherently aligned), so no row-byte check applies to them.
    const uint32_t dram_align = tt::tt_metal::hal::get_dram_alignment();
    TT_FATAL((d * q.element_size()) % dram_align == 0, "q row bytes must be {}B aligned", dram_align);
    TT_FATAL((TOPK * idx.element_size()) % dram_align == 0, "indices row bytes must be {}B aligned", dram_align);
    TT_FATAL((v_dim * q.element_size()) % dram_align == 0, "output row bytes must be {}B aligned", dram_align);
}

SparseSDPAMsaOperation::spec_return_value_t SparseSDPAMsaOperation::compute_output_specs(
    const SparseSDPAMsaParams& /*attrs*/, const SparseSDPAMsaInputs& t) {
    auto shape = t.q.logical_shape();   // [1, H, S, d]
    shape[3] = t.v.logical_shape()[3];  // [1, H, S, v_dim]
    // Output is DRAM-interleaved ROW_MAJOR, with dtype matching q.
    const tt::tt_metal::MemoryConfig out_mem{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    return TensorSpec(
        shape, tt::tt_metal::TensorLayout(t.q.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), out_mem));
}

SparseSDPAMsaOperation::tensor_return_value_t SparseSDPAMsaOperation::create_output_tensors(
    const SparseSDPAMsaParams& attrs, const SparseSDPAMsaInputs& t) {
    return create_device_tensor(compute_output_specs(attrs, t), t.q.device());
}

ttsl::hash::hash_t SparseSDPAMsaOperation::compute_program_hash(
    const SparseSDPAMsaParams& attrs, const SparseSDPAMsaInputs& t) {
    // Hash compile-time choices. Interleaved K/V T and cache_batch_idx are patched at dispatch.
    // Sharded K/V shapes stay hashed because accessor strides depend on them.
    return tt::tt_metal::operation::hash_operation<SparseSDPAMsaOperation>(
        std::bit_cast<uint32_t>(attrs.scale),
        attrs.block_size,
        attrs.compute_kernel_config,
        t.q.logical_shape(),
        t.q.dtype(),
        t.k.dtype(),
        t.k.memory_config(),
        t.k.memory_config().is_sharded() ? t.k.logical_shape() : tt::tt_metal::Shape{},
        t.v.dtype(),
        t.v.memory_config(),
        t.v.memory_config().is_sharded() ? t.v.logical_shape() : tt::tt_metal::Shape{},
        t.v.logical_shape()[3],
        attrs.has_indexed_kv_cache(),
        attrs.causal_enabled(),
        t.indices.logical_shape(),
        t.indices.dtype());
}

SparseSDPAMsaOperation::program_factory_t SparseSDPAMsaOperation::select_program_factory(
    const SparseSDPAMsaParams& /*attrs*/, const SparseSDPAMsaInputs& /*t*/) {
    return MeshWorkloadFactory{};
}

SparseSDPAMsaOperation::MeshWorkloadFactory::cached_mesh_workload_t
SparseSDPAMsaOperation::MeshWorkloadFactory::create_mesh_workload(
    const SparseSDPAMsaParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const SparseSDPAMsaInputs& tensor_args,
    Tensor& output) {
    // The adapter runs SparseSDPAMsaProgramFactory::create_descriptor once per coordinate, so each device's
    // program bakes its own per-device causal chunk_start (chunk_start_idx + device_index*S) from its coord.
    return descriptor_adapter_t::create_mesh_workload(args, tensor_coords, tensor_args, output);
}

void SparseSDPAMsaOperation::MeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const SparseSDPAMsaParams& attrs,
    const SparseSDPAMsaInputs& t,
    Tensor& output) {
    // Cache-hit fast path: re-apply operand buffer-binding addresses on every cached program.
    descriptor_adapter_t::apply_descriptor(cached_workload, attrs, t, output);

    const uint32_t n_kv = t.k.logical_shape()[1];
    const bool patch_kv = attrs.has_indexed_kv_cache() || n_kv > 1;
    // n_kv==1 non-indexed + non-causal programs have no hash-excluded scalars beyond the buffers.
    if (!patch_kv && !attrs.causal_enabled()) {
        return;
    }

    // Hash-excluded scalars re-applied per dispatch: cache-slot offsets (cache_batch_idx), GQA group strides
    // (interleaved K/V T excluded from the hash), and the per-device causal chunk_start.
    constexpr uint32_t tw = tt::constants::TILE_WIDTH;
    constexpr uint32_t th = tt::constants::TILE_HEIGHT;
    const uint32_t S = t.q.logical_shape()[2];
    const uint32_t T = t.k.logical_shape()[2];
    const uint32_t d = t.q.logical_shape()[3];
    const uint32_t v_dim = t.v.logical_shape()[3];
    const uint32_t s = attrs.cache_batch_idx.value_or(0);
    const uint32_t tiles_per_row = T / th;
    const uint32_t k_group_tile_stride = tiles_per_row * (d / tw);
    const uint32_t v_group_tile_stride = tiles_per_row * (v_dim / tw);
    const uint32_t k_batch_tile_offset = s * n_kv * k_group_tile_stride;
    const uint32_t v_batch_tile_offset = s * n_kv * v_group_tile_stride;
    const bool multi_device = t.q.device_storage().get_coords().size() > 1;

    const tt::tt_metal::CoreCoord grid = t.q.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = grid.x * grid.y;

    // One program per mesh coordinate; recompute chunk_start from each program's coordinate so chunked prefill
    // (chunk_start_idx changes on a cache hit) stays correct per device.
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        uint32_t chunk_start_local = 0;
        if (attrs.causal_enabled()) {
            uint32_t device_index = 0;
            if (multi_device) {
                for (const auto& coord : range) {
                    device_index =
                        ttnn::ccl::get_linearized_index_from_physical_coord(t.q, coord, attrs.cluster_axis);
                    break;  // single-coordinate range: all cores in this program share one SP rank
                }
            }
            chunk_start_local = attrs.chunk_start_idx.value() + device_index * S;
        }
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, sparse_sdpa_msa_rt::kReaderKernelIdx);
        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, sparse_sdpa_msa_rt::kWriterKernelIdx);
        for (uint32_t i = 0; i < num_cores; ++i) {
            const tt::tt_metal::CoreCoord core = {i % grid.x, i / grid.x};
            auto& reader_rt = reader_args[core.x][core.y];
            auto& writer_rt = writer_args[core.x][core.y];
            if (attrs.causal_enabled()) {
                reader_rt[sparse_sdpa_msa_rt::kReaderChunkStartArg] = chunk_start_local;
            }
            if (attrs.has_indexed_kv_cache()) {
                reader_rt[sparse_sdpa_msa_rt::kReaderKBatchOffsetArg] = k_batch_tile_offset;
                reader_rt[sparse_sdpa_msa_rt::kReaderVBatchOffsetArg] = v_batch_tile_offset;
                writer_rt[sparse_sdpa_msa_rt::kWriterKBatchOffsetArg] = k_batch_tile_offset;
                writer_rt[sparse_sdpa_msa_rt::kWriterVBatchOffsetArg] = v_batch_tile_offset;
            }
            if (n_kv > 1) {
                reader_rt[sparse_sdpa_msa_rt::kReaderKGroupStrideArg] = k_group_tile_stride;
                reader_rt[sparse_sdpa_msa_rt::kReaderVGroupStrideArg] = v_group_tile_stride;
                writer_rt[sparse_sdpa_msa_rt::kWriterKGroupStrideArg] = k_group_tile_stride;
                writer_rt[sparse_sdpa_msa_rt::kWriterVGroupStrideArg] = v_group_tile_stride;
            }
        }
    }
}

Tensor sparse_sdpa_msa(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& indices,
    float scale,
    uint32_t block_size,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx,
    std::optional<uint32_t> chunk_start_idx,
    std::optional<uint32_t> cluster_axis) {
    using OperationType = ttnn::prim::SparseSDPAMsaOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .scale = scale,
            .block_size = block_size,
            .compute_kernel_config = compute_kernel_config,
            .cache_batch_idx = cache_batch_idx,
            .chunk_start_idx = chunk_start_idx,
            .cluster_axis = cluster_axis,
        },
        OperationType::tensor_args_t{
            .q = q,
            .k = k,
            .v = v,
            .indices = indices,
        });
}

}  // namespace ttnn::prim
