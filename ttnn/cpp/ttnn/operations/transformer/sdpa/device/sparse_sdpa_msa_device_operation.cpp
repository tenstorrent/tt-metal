// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_msa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_msa_common.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/indexer_score/device/kernels/indexer_score_causal_geometry.hpp"
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <algorithm>
#include <bit>

namespace ttnn::prim {

namespace {
// The SP-sharded read of a block-cyclic chunked-prefill cache: the global positions of this device's query rows
// follow the cache writer's rotation (see compute_causal_geometry) -- q holds its SP rank's chunk_local rows, or,
// seq-sharded across TP as well (chunk_local == tp*S), its TP rank's S-row slice of them.
bool rotation_exact_causal(const SparseSDPAMsaParams& attrs) {
    return attrs.causal_enabled() && attrs.cluster_axis.has_value() && attrs.has_block_cyclic();
}

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
    // chunk_start_idx is hash-excluded (patched per dispatch), hence checked on hits too.
    if (attrs.causal_enabled() && attrs.has_block_cyclic()) {
        const uint32_t chunk_start_idx = attrs.chunk_start_idx.value();
        const auto& bc = attrs.block_cyclic.value();
        // The rotated geometry works in tile-rows (as the KV writer does), so the chunk must start on the tile grid.
        TT_FATAL(
            chunk_start_idx % tt::constants::TILE_HEIGHT == 0,
            "sparse_sdpa_msa: chunk_start_idx ({}) must be a multiple of {} with a block-cyclic cache",
            chunk_start_idx,
            tt::constants::TILE_HEIGHT);
        // Without cluster_axis every device masks at chunk_start_idx + its flat rank * S, which matches the
        // writer's placement only for a slab-aligned start.
        TT_FATAL(
            rotation_exact_causal(attrs) || bc.sp <= 1 || chunk_start_idx % (bc.sp * bc.chunk_local) == 0,
            "sparse_sdpa_msa: a mid-slab chunk_start_idx ({}, slab {} = sp {} x chunk_local {}) with a block-cyclic "
            "cache needs cluster_axis (the SP axis) for the rotated per-device query positions",
            chunk_start_idx,
            bc.sp * bc.chunk_local,
            bc.sp,
            bc.chunk_local);
    }
}
}  // namespace

void SparseSDPAMsaOperation::validate_on_program_cache_hit(
    const SparseSDPAMsaParams& attrs, const SparseSDPAMsaInputs& t) {
    validate_non_hashed(attrs, t);
    validate_kv_cache_request(attrs, t);
}

// An explicit slot count that cannot be honoured at all is a caller error; auto (0) falls back to the streamed
// kernels instead. Checked on hits too: an auto call that fell back and an explicit request with no room resolve
// to the same (streamed) program, so the miss-only validator would not see the second.
void SparseSDPAMsaOperation::validate_kv_cache_request(const SparseSDPAMsaParams& attrs, const SparseSDPAMsaInputs& t) {
    if (attrs.kv_cache_blocks.value_or(0) == 0) {  // off or auto
        return;
    }
    const KvCachePlan kv = resolve_kv_cache(geometry(attrs, t), attrs, t);
    TT_FATAL(
        kv.slots > 0,
        "sparse_sdpa_msa: kv_cache_blocks={} but no L1 is left for one {} B K+V block slot ({} B free after the "
        "op's own CBs and {} B slack)",
        attrs.kv_cache_blocks.value(),
        kv.block_bytes,
        kv.free_l1,
        sparse_sdpa_msa::KV_CACHE_L1_SLACK_BYTES);
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
    validate_kv_cache_request(attrs, t);

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

    // Block-cyclic ("slab") cache: the invP block remap bakes T/sp and chunk_local (in blocks) as compile-time
    // arguments, so the layout must divide cleanly. Miss-only — the constants are hashed.
    if (attrs.has_block_cyclic()) {
        const uint32_t sp = attrs.block_cyclic->sp;
        const uint32_t chunk_local = attrs.block_cyclic->chunk_local;
        const uint32_t T = k.logical_shape()[2];
        TT_FATAL(
            sp > 0 && chunk_local > 0,
            "block_cyclic sp/chunk_local must be > 0 (got sp {}, chunk_local {})",
            sp,
            chunk_local);
        TT_FATAL(T % sp == 0, "block_cyclic: sp ({}) must divide T ({})", sp, T);
        const uint32_t shard_len = T / sp;
        TT_FATAL(
            shard_len % chunk_local == 0,
            "block_cyclic: chunk_local ({}) must divide shard_len T/sp ({})",
            chunk_local,
            shard_len);
        // Remap is block-granular -> chunk_local and shard_len must be whole numbers of blocks.
        TT_FATAL(
            chunk_local % attrs.block_size == 0,
            "block_cyclic: block_size ({}) must divide chunk_local ({})",
            attrs.block_size,
            chunk_local);
        TT_FATAL(
            shard_len % attrs.block_size == 0,
            "block_cyclic: block_size ({}) must divide shard_len T/sp ({})",
            attrs.block_size,
            shard_len);
    }
}

SparseSDPAMsaOperation::spec_return_value_t SparseSDPAMsaOperation::compute_output_specs(
    const SparseSDPAMsaParams& /*attrs*/, const SparseSDPAMsaInputs& t) {
    auto shape = t.q.logical_shape();   // [1, H, S, d]
    shape[3] = t.v.logical_shape()[3];  // [1, H, S, v_dim]
    // Output is DRAM-interleaved ROW_MAJOR, with dtype matching q.
    const tt::tt_metal::MemoryConfig out_mem{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    return tt::tt_metal::TensorSpec(
        shape, tt::tt_metal::TensorLayout(t.q.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), out_mem));
}

SparseSDPAMsaOperation::tensor_return_value_t SparseSDPAMsaOperation::create_output_tensors(
    const SparseSDPAMsaParams& attrs, const SparseSDPAMsaInputs& t) {
    return create_device_tensor(compute_output_specs(attrs, t), t.q.device());
}

SparseSDPAMsaOperation::Geometry SparseSDPAMsaOperation::geometry(
    const SparseSDPAMsaParams& attrs, const SparseSDPAMsaInputs& t) {
    Geometry g;
    const uint32_t H_total = t.q.logical_shape()[1];
    g.n_kv = t.k.logical_shape()[1];
    g.H_logical = g.n_kv ? H_total / g.n_kv : 0;  // query heads per KV group (validation rejects n_kv == 0)
    // Compute always sees whole 32-head tiles; the reader zero-fills the padded heads.
    g.H = ((g.H_logical + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT) * tt::constants::TILE_HEIGHT;
    g.S = t.q.logical_shape()[2];
    g.topk = t.indices.logical_shape()[3];
    g.d = t.q.logical_shape()[3];
    g.v_dim = t.v.logical_shape()[3];
    g.DHt = g.d / tt::constants::TILE_WIDTH;
    g.vDHt = g.v_dim / tt::constants::TILE_WIDTH;
    g.Skt = attrs.block_size / tt::constants::TILE_WIDTH;  // a chunk is exactly one block
    g.Sqt = g.H / tt::constants::TILE_HEIGHT;
    g.k_tiles_per_block = g.Skt * g.DHt;
    g.v_tiles_per_block = g.Skt * g.vDHt;
    g.k_df = tt::tt_metal::datatype_to_dataformat_converter(t.k.dtype());
    g.v_df = tt::tt_metal::datatype_to_dataformat_converter(t.v.dtype());
    g.k_tile_bytes = tt::tile_size(g.k_df);
    g.v_tile_bytes = tt::tile_size(g.v_df);
    // Q is read row-major and tiled on chip; fp8 Q tilizes into bfp8_b. The output matches Q's dtype.
    g.q_rm_df = tt::tt_metal::datatype_to_dataformat_converter(t.q.dtype());
    g.q_is_fp8 = (t.q.dtype() == DataType::FP8_E4M3);
    g.q_in_df = g.q_is_fp8 ? tt::DataFormat::Bfp8_b : g.q_rm_df;
    g.out_df = g.q_rm_df;
    g.q_row_bytes = g.d * t.q.element_size();
    g.idx_row_bytes = g.topk * t.indices.element_size();
    return g;
}

std::vector<SparseSDPAMsaOperation::CbSpec> SparseSDPAMsaOperation::base_cbs(
    const Geometry& g, bool causal, bool block_cache_serves_kv) {
    constexpr tt::DataFormat bf = tt::DataFormat::Float16_b;
    constexpr uint32_t tile_bytes = tt::tile_size(bf);  // intermediates are bf16
    std::vector<CbSpec> cbs = {
        {cb_q_rm, g.q_row_bytes, g.H, g.q_rm_df},
        {cb_q_in, tt::tile_size(g.q_in_df), g.Sqt * g.DHt, g.q_in_df},
        {cb_scale, tile_bytes, 1, bf},
        {cb_qk_im, tile_bytes, g.Sqt * g.Skt, bf},
        {cb_max_a, tile_bytes, g.Sqt, bf},
        {cb_max_b, tile_bytes, g.Sqt, bf},
        {cb_sum_a, tile_bytes, g.Sqt, bf},
        {cb_sum_b, tile_bytes, g.Sqt, bf},
        {cb_out_a, tile_bytes, g.Sqt * g.vDHt, bf},
        {cb_out_b, tile_bytes, g.Sqt * g.vDHt, bf},
        {cb_corr, tile_bytes, g.Sqt, bf},
        {cb_out_im, tile_bytes, g.Sqt * g.vDHt, bf},  // bf16 accumulator, full precision
        {cb_out_rm, tt::tile_size(g.out_df), g.Sqt * g.vDHt, g.out_df},
        {cb_idx, g.idx_row_bytes, 1, bf},
        {cb_ctrl, sparse_sdpa_msa::ctrl::PAGE_BYTES, 2, bf},  // active block count + causal control
        {cb_col_identity, tile_bytes, 1, bf},
        {cb_recip_scratch, tile_bytes, 1, bf},
        {cb_kreq, sparse_sdpa_msa::kreq::PAGE_BYTES, 2, bf},
        {cb_kack, sparse_sdpa_msa::ACK_PAGE_BYTES, 2, bf},
    };
    // Streamed K/V: one block, single-buffered -- the reader reserves it and the writer fills its half into the
    // same L1. Absent when the block cache serves K/V (compute then reads the cache CBs in place).
    if (!block_cache_serves_kv) {
        cbs.push_back({cb_k_in, g.k_tile_bytes, g.k_tiles_per_block, g.k_df});
        cbs.push_back({cb_v_in, g.v_tile_bytes, g.v_tiles_per_block, g.v_df});
    }
    // The mask tiles are touched only under CAUSAL_MASK_ENABLED, so causal-off skips their L1.
    if (causal) {
        cbs.push_back({cb_neginf, tile_bytes, 1, bf});
        cbs.push_back({cb_vmask, tile_bytes, 2, bf});
    }
    return cbs;
}

SparseSDPAMsaOperation::KvCachePlan SparseSDPAMsaOperation::resolve_kv_cache(
    const Geometry& g, const SparseSDPAMsaParams& attrs, const SparseSDPAMsaInputs& t) {
    KvCachePlan plan;
    plan.block_bytes = g.k_tiles_per_block * g.k_tile_bytes + g.v_tiles_per_block * g.v_tile_bytes;
    // The hash runs before validation, so a not-yet-rejected block_size or d below a tile can give block_bytes == 0.
    if (!attrs.kv_cache_blocks.has_value() || plan.block_bytes == 0) {
        return plan;
    }
    uint64_t base_bytes = 0;
    for (const CbSpec& s : base_cbs(g, attrs.causal_enabled(), /*block_cache_serves_kv=*/true)) {
        base_bytes += static_cast<uint64_t>(s.page_size) * s.num_pages;
    }
    // Free L1 for the slots: [CB base, lowest live L1 buffer) minus the base CBs and the slack. L1 buffers fill the
    // interleaved-L1 bank top-down, so its end (not l1_size_per_core(), which also spans L1_SMALL) is the bound
    // when nothing is live. Anything this mesh-level view misses, e.g. a HYBRID allocator's per-device buffers,
    // trips the CB/buffer overlap check at program launch, which fails rather than corrupts.
    auto* device = t.q.device();
    const uint64_t l1_base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint64_t l1_end = l1_base + device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1);
    const uint64_t l1_top = std::min<uint64_t>(device->lowest_occupied_compute_l1_address().value_or(l1_end), l1_end);
    const uint64_t reserved = base_bytes + sparse_sdpa_msa::KV_CACHE_L1_SLACK_BYTES;
    plan.free_l1 = l1_top > l1_base + reserved ? l1_top - l1_base - reserved : 0;
    const uint32_t n_fit =
        static_cast<uint32_t>(std::min<uint64_t>(plan.free_l1 / plan.block_bytes, sparse_sdpa_msa::KV_CACHE_SLOTS_MAX));
    const uint32_t requested = attrs.kv_cache_blocks.value();
    plan.slots = requested == 0 ? n_fit : std::min(requested, n_fit);
    // depth <= slots: the reader's victim search needs one slot outside the in-flight set (its static_assert).
    plan.slot_depth = std::min(sparse_sdpa_msa::KV_CACHE_SLOT_DEPTH_MAX, std::max(plan.slots, 1u));
    return plan;
}

ttsl::hash::hash_t SparseSDPAMsaOperation::compute_program_hash(
    const SparseSDPAMsaParams& attrs, const SparseSDPAMsaInputs& t) {
    // Hash compile-time choices. Interleaved K/V T and cache_batch_idx are patched at dispatch.
    // Sharded K/V shapes stay hashed because accessor strides depend on them. The block-cyclic path also
    // hashes T: the shard stride gap (= (T/sp - chunk_local)/block_size, in blocks) is baked as a compile-time
    // argument, so a different cache size must be a distinct program.
    return tt::tt_metal::operation::hash_operation<SparseSDPAMsaOperation>(
        std::bit_cast<uint32_t>(attrs.scale),
        attrs.block_size,
        attrs.compute_kernel_config,
        t.q.logical_shape(),
        t.q.dtype(),
        t.k.dtype(),
        t.k.memory_config(),
        (t.k.memory_config().is_sharded() || attrs.has_block_cyclic()) ? t.k.logical_shape() : tt::tt_metal::Shape{},
        t.v.dtype(),
        t.v.memory_config(),
        (t.v.memory_config().is_sharded() || attrs.has_block_cyclic()) ? t.v.logical_shape() : tt::tt_metal::Shape{},
        t.v.logical_shape()[3],
        attrs.has_indexed_kv_cache(),
        attrs.causal_enabled(),
        attrs.has_block_cyclic(),
        attrs.block_cyclic.has_value() ? attrs.block_cyclic->sp : 0u,
        attrs.block_cyclic.has_value() ? attrs.block_cyclic->chunk_local : 0u,
        // The RESOLVED slot count: the CB layout and kernels bake it in, and auto depends on free L1 at this call.
        // A request that resolves to no slots aliases the cache-off program (same layout, same kernels).
        resolve_kv_cache(geometry(attrs, t), attrs, t).slots,
        t.indices.logical_shape(),
        t.indices.dtype());
}

SparseSDPAMsaOperation::CausalGeometry SparseSDPAMsaOperation::compute_causal_geometry(
    const SparseSDPAMsaParams& attrs,
    const SparseSDPAMsaInputs& t,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    // Derived exactly as indexer_score_msa's start (same closed form, same device index), so the mask and the
    // indexer's selection share one global-position frame.
    if (!attrs.causal_enabled()) {
        return {};
    }
    const uint32_t S = t.q.logical_shape()[2];
    const uint32_t chunk_start_idx = attrs.chunk_start_idx.value();
    const uint32_t device_index =
        mesh_dispatch_coordinate.has_value()
            ? ttnn::ccl::get_linearized_index_from_physical_coord(t.q, *mesh_dispatch_coordinate, attrs.cluster_axis)
            : 0;
    if (!rotation_exact_causal(attrs)) {
        return {.chunk_start = chunk_start_idx + device_index * S};
    }
    // The chunk [chunk_start_idx, +sp*chunk_local) was written round-robin by update_padded_kv_cache, so this
    // device's S query rows start at the writer's rotated position, not chunk_start_idx + rank*S, and on the
    // boundary chip of a mid-slab start they cross a slab boundary (the straddle). With q also seq-sharded over
    // TP (chunk_local == tp*S) the device holds its TP rank's S-row slice of its SP rank's rows: the same
    // [SP, TP] geometry indexer_score uses with seq_shard_axes=[SP, TP].
    const auto& bc = attrs.block_cyclic.value();
    TT_FATAL(
        device_index < bc.sp,
        "sparse_sdpa_msa: cluster_axis rank {} out of range for block-cyclic sp={} (cluster_axis must be the "
        "block-cyclic SP axis)",
        device_index,
        bc.sp);
    uint32_t tp_index = 0;
    if (bc.chunk_local != S) {
        const auto mesh_shape = t.q.device()->get_view().shape();
        TT_FATAL(
            mesh_shape.dims() == 2 && bc.chunk_local % S == 0 && mesh_dispatch_coordinate.has_value(),
            "sparse_sdpa_msa: a TP-sub-sharded q (block_cyclic_chunk_local {} = tp * q seq-len {}) needs a 2D mesh",
            bc.chunk_local,
            S);
        const uint32_t tp_axis = 1 - attrs.cluster_axis.value();
        tp_index = ttnn::ccl::get_linearized_index_from_physical_coord(t.q, *mesh_dispatch_coordinate, tp_axis);
        TT_FATAL(
            tp_index < bc.chunk_local / S,
            "sparse_sdpa_msa: TP rank {} out of range for block_cyclic_chunk_local {} / q seq-len {}",
            tp_index,
            bc.chunk_local,
            S);
    }
    constexpr uint32_t TW = tt::constants::TILE_WIDTH;
    const auto g = ttnn::operations::experimental::indexer_score::causal_geometry_tiles(
        chunk_start_idx,
        /*has_block_cyclic=*/true,
        /*rotation_exact=*/true,
        bc.sp,
        bc.chunk_local,
        device_index,
        tp_index,
        S);
    return {
        .chunk_start = g.chunk_start_tiles * TW,
        .straddle_row = g.straddle_q_tile * TW,
        .straddle_jump = g.straddle_jump_tiles * TW,
    };
}

SparseSDPAMsaOperation::DispatchArgs SparseSDPAMsaOperation::compute_dispatch_args(
    const SparseSDPAMsaParams& attrs,
    const SparseSDPAMsaInputs& t,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const uint32_t S = t.q.logical_shape()[2];
    const uint32_t n_kv = t.k.logical_shape()[1];
    const uint32_t T = t.k.logical_shape()[2];
    const uint32_t d = t.q.logical_shape()[3];
    const uint32_t v_dim = t.v.logical_shape()[3];
    const uint32_t tiles_per_row = T / tt::constants::TILE_HEIGHT;
    const uint32_t k_group_tile_stride = tiles_per_row * (d / tt::constants::TILE_WIDTH);
    const uint32_t v_group_tile_stride = tiles_per_row * (v_dim / tt::constants::TILE_WIDTH);
    const uint32_t slot = attrs.cache_batch_idx.value_or(0);
    const tt::tt_metal::CoreCoord grid = t.q.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = grid.x * grid.y;
    const uint32_t total_work = S * n_kv;
    return DispatchArgs{
        .grid = grid,
        .num_cores = num_cores,
        .base_work = total_work / num_cores,
        .extra = total_work % num_cores,
        .k_batch_tile_offset = slot * n_kv * k_group_tile_stride,
        .v_batch_tile_offset = slot * n_kv * v_group_tile_stride,
        .k_group_tile_stride = k_group_tile_stride,
        .v_group_tile_stride = v_group_tile_stride,
        .causal = compute_causal_geometry(attrs, t, mesh_dispatch_coordinate),
    };
}

void SparseSDPAMsaOperation::SparseSDPAMsaProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const SparseSDPAMsaParams& attrs,
    const SparseSDPAMsaInputs& t,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    // Patch the cached program in place. Calling create_descriptor() here instead would pay the cache-MISS host
    // cost on every hit (work split, CoreRangeSet, kernel sources, compute-config arch queries, accessor args,
    // one heap-allocated arg vector per core) on a grid-wide op with three kernels.
    //
    // Every slot written below either holds a buffer address - the override supersedes resolve_bindings, so all
    // of them are ours to re-apply - or derives from a value compute_program_hash excludes: interleaved K/V T
    // and batch slots, n_kv, cache_batch_idx, and chunk_start_idx/cluster_axis (patched per coordinate, from
    // the coordinate this program was built for). Nothing else is dynamic: kernel geometry and CB sizes are
    // hash-pinned, all CBs are locally allocated (no `.buffer`/`.tensor` backing to re-point), and the only
    // common runtime args come from the K/V TensorAccessorArgs, which exist solely when K/V are sharded - and
    // sharded K/V shape and memory config are hashed.
    //
    // Kernel push order in create_descriptor(): reader(0), writer(1), compute(2).
    constexpr uint32_t kReaderKernelIdx = 0;
    constexpr uint32_t kWriterKernelIdx = 1;
    constexpr uint32_t kComputeKernelIdx = 2;

    const auto dyn = compute_dispatch_args(attrs, t, mesh_dispatch_coordinate);
    const uint32_t q_addr = t.q.buffer()->address();
    const uint32_t k_addr = t.k.buffer()->address();
    const uint32_t v_addr = t.v.buffer()->address();
    const uint32_t idx_addr = t.indices.buffer()->address();
    const uint32_t out_addr = tensor_return_value.buffer()->address();

    for (uint32_t i = 0; i < dyn.num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i % dyn.grid.x, i / dyn.grid.x};
        const uint32_t work_start = i * dyn.base_work + std::min(i, dyn.extra);
        const uint32_t work_count = dyn.base_work + (i < dyn.extra ? 1u : 0u);

        auto& reader = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx, core);
        TT_FATAL(
            reader.size() == kReaderArgCount,
            "sparse_sdpa_msa reader expected {} runtime args, cached program has {}",
            static_cast<uint32_t>(kReaderArgCount),
            reader.size());
        reader[kReaderQAddr] = q_addr;
        reader[kReaderKAddr] = k_addr;
        reader[kReaderVAddr] = v_addr;
        reader[kReaderIdxAddr] = idx_addr;
        reader[kReaderWorkStart] = work_start;
        reader[kReaderWorkCount] = work_count;
        reader[kReaderKBatchOffset] = dyn.k_batch_tile_offset;
        reader[kReaderVBatchOffset] = dyn.v_batch_tile_offset;
        reader[kReaderKGroupStride] = dyn.k_group_tile_stride;
        reader[kReaderVGroupStride] = dyn.v_group_tile_stride;
        reader[kReaderChunkStart] = dyn.causal.chunk_start;
        reader[kReaderStraddleRow] = dyn.causal.straddle_row;
        reader[kReaderStraddleJump] = dyn.causal.straddle_jump;

        auto& writer = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx, core);
        TT_FATAL(
            writer.size() == kWriterArgCount,
            "sparse_sdpa_msa writer expected {} runtime args, cached program has {}",
            static_cast<uint32_t>(kWriterArgCount),
            writer.size());
        writer[kWriterOutAddr] = out_addr;
        writer[kWriterWorkStart] = work_start;
        writer[kWriterWorkCount] = work_count;
        writer[kWriterKAddr] = k_addr;
        writer[kWriterVAddr] = v_addr;
        writer[kWriterKBatchOffset] = dyn.k_batch_tile_offset;
        writer[kWriterVBatchOffset] = dyn.v_batch_tile_offset;
        writer[kWriterKGroupStride] = dyn.k_group_tile_stride;
        writer[kWriterVGroupStride] = dyn.v_group_tile_stride;

        auto& compute = tt::tt_metal::GetRuntimeArgs(program, kComputeKernelIdx, core);
        TT_FATAL(
            compute.size() == kComputeArgCount,
            "sparse_sdpa_msa compute expected {} runtime args, cached program has {}",
            static_cast<uint32_t>(kComputeArgCount),
            compute.size());
        compute[kComputeWorkStart] = work_start;
        compute[kComputeWorkCount] = work_count;
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
    std::optional<uint32_t> cluster_axis,
    std::optional<BlockCyclicLayout> block_cyclic,
    std::optional<uint32_t> kv_cache_blocks) {
    using OperationType = ttnn::prim::SparseSDPAMsaOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .scale = scale,
            .block_size = block_size,
            .compute_kernel_config = compute_kernel_config,
            .cache_batch_idx = cache_batch_idx,
            .block_cyclic = block_cyclic,
            .chunk_start_idx = chunk_start_idx,
            .cluster_axis = cluster_axis,
            .kv_cache_blocks = kv_cache_blocks,
        },
        OperationType::tensor_args_t{
            .q = q,
            .k = k,
            .v = v,
            .indices = indices,
        });
}

}  // namespace ttnn::prim
