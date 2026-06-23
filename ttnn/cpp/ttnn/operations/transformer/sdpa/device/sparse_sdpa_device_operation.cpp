// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <bit>

namespace ttnn::prim {

namespace {
// All input invariants the program hash does NOT key on, so they can vary while hitting the same cached
// program and must be re-checked on EVERY dispatch (miss AND hit). The hash keys only on q/indices shape+dtype
// and kv dtype+memory_config — NOT on tensor layout, padding, q/indices buffer type/sharding, the kv logical
// shape (its T rides on the kv TensorAccessor's RuntimeTensorShape runtime metadata, not a kernel scalar),
// cache_batch_idx (a dynamic runtime arg), or device
// placement. The accessors assume ROW_MAJOR, unpadded, DRAM, interleaved q/indices, so a cache hit with a
// tiled/padded/L1/sharded/off-device tensor would otherwise run on wrong assumptions. K_DIM comes from q,
// whose shape IS hashed, so it is pinned. Shared by validate_on_program_cache_miss and _hit.
void validate_non_hashed(const SparseSDPAParams& attrs, const SparseSDPAInputs& t) {
    const auto& q = t.q;
    const auto& kv = t.kv;
    const auto& idx = t.indices;
    TT_FATAL(
        q.device() == kv.device() && q.device() == idx.device(), "sparse_sdpa: all inputs must be on the same device");
    // q and indices: ROW_MAJOR, DRAM, interleaved, unpadded (row-major paged-accessor assumptions).
    for (const Tensor* tp : {&q, &idx}) {
        TT_FATAL(tp->layout() == Layout::ROW_MAJOR, "sparse_sdpa q/indices must be ROW_MAJOR");
        TT_FATAL(tp->memory_config().buffer_type() == BufferType::DRAM, "sparse_sdpa q/indices must be in DRAM");
        TT_FATAL(!tp->memory_config().is_sharded(), "sparse_sdpa q/indices must be interleaved");
        TT_FATAL(tp->padded_shape() == tp->logical_shape(), "sparse_sdpa q/indices must not be padded");
    }
    // kv: ROW_MAJOR and unpadded; may be interleaved OR ND-sharded DRAM (the accessor resolves the per-page
    // bank/shard either way — kv's DRAM/shard layout is pinned via the hashed kv.memory_config(), its
    // ROW_MAJOR layout and padding are not).
    TT_FATAL(kv.layout() == Layout::ROW_MAJOR, "sparse_sdpa kv must be ROW_MAJOR");
    TT_FATAL(kv.padded_shape() == kv.logical_shape(), "sparse_sdpa kv must not be padded");
    const uint32_t K_DIM = q.logical_shape()[3];
    const auto kvs = kv.logical_shape();
    // kv is [B,1,T,K_DIM]: B == 1 normally; when indexed (cache_batch_idx set) B is the cache's batch slots
    // and cache_batch_idx selects one. q is always batch-1, so the selected slot serves the whole query.
    TT_FATAL(
        kvs.rank() == 4 && kvs[1] == 1 && kvs[3] == K_DIM,
        "kv must be [B,1,T,K_DIM] with K_DIM matching q ({})",
        K_DIM);
    TT_FATAL(kvs[2] > 0, "kv T (cache length) must be > 0");
    const uint32_t B = kvs[0];
    if (attrs.cache_batch_idx.has_value()) {
        TT_FATAL(
            attrs.cache_batch_idx.value() < B,
            "cache_batch_idx ({}) must be < kv batch slots ({})",
            attrs.cache_batch_idx.value(),
            B);
    } else {
        TT_FATAL(B == 1, "kv batch must be 1 unless cache_batch_idx is set (got {})", B);
    }
}
}  // namespace

void SparseSDPAOperation::validate_on_program_cache_hit(const SparseSDPAParams& attrs, const SparseSDPAInputs& t) {
    validate_non_hashed(attrs, t);
}

void SparseSDPAOperation::validate_on_program_cache_miss(const SparseSDPAParams& attrs, const SparseSDPAInputs& t) {
    const auto& q = t.q;
    const auto& kv = t.kv;
    const auto& idx = t.indices;

    TT_FATAL(tt::tt_metal::hal::get_arch() == tt::ARCH::BLACKHOLE, "sparse_sdpa is Blackhole-only");

    // dtypes. q and kv may each be bf16 or fp8_e4m3 — fp8 halves that input's K/Q-gather bytes; the reader
    // gathers it row-major and compute tilizes fp8 -> bfp8_b cb_*_in (near-lossless, also halves the L1).
    // fp8 inputs require fp32_dest_acc_en (the unpack-to-dest path); enforced below.
    const bool kv_is_fp8 = (kv.dtype() == DataType::FP8_E4M3);
    const bool q_is_fp8 = (q.dtype() == DataType::FP8_E4M3);
    TT_FATAL(q.dtype() == DataType::BFLOAT16 || q_is_fp8, "q must be bf16 or fp8_e4m3");
    TT_FATAL(kv.dtype() == DataType::BFLOAT16 || kv_is_fp8, "kv must be bf16 or fp8_e4m3");
    TT_FATAL(idx.dtype() == DataType::UINT32, "indices must be uint32");

    // kv DRAM residency is keyed by the hash (kv.memory_config()), so it is checked on miss only. The
    // remaining layout/memory/padding/device invariants (q/indices layout+memory+padding, kv layout+padding,
    // same-device) are NOT hashed and are checked in validate_non_hashed (called below, run on miss AND hit).
    TT_FATAL(kv.memory_config().buffer_type() == BufferType::DRAM, "sparse_sdpa kv must be in DRAM");

    const auto qs = q.logical_shape();
    const auto is = idx.logical_shape();
    TT_FATAL(qs.rank() == 4 && qs[0] == 1, "q must be [1,H,S,K_DIM]");
    const uint32_t H = qs[1];
    const uint32_t S = qs[2];
    const uint32_t K_DIM = qs[3];  // head dim, taken from the tensor (not hardcoded)
    // H heads map to H/TILE_HEIGHT query tile-rows (compute processes them in DST-sized groups, so DST
    // doesn't cap H). The upper bound is the per-core L1 budget — the flash state (out/max/sum) and Q both
    // scale with H — so a too-large H simply fails CB allocation at program creation rather than here.
    constexpr uint32_t tile_h = tt::constants::TILE_HEIGHT;
    TT_FATAL(H % tile_h == 0 && H >= tile_h, "sparse_sdpa: H must be a multiple of {} (got {})", tile_h, H);
    // All non-hashed invariants (q/indices/kv layout+memory+padding, kv shape, cache_batch_idx, same-device).
    // kv may be oversized: a persistent max-size cache whose logical T far exceeds the keys any query attends
    // to. No valid-length bound is needed — reads are index-driven (indices < populated length, sentinels mark
    // unused slots), so the unpopulated suffix is never addressed.
    validate_non_hashed(attrs, t);
    TT_FATAL(is.rank() == 4 && is[0] == 1 && is[1] == 1 && is[2] == S, "indices must be [1,1,S,TOPK]");
    const uint32_t TOPK = is[3];
    TT_FATAL(S > 0 && TOPK > 0, "S/TOPK must be > 0");

    // V is the leading v_dim cols of the K_DIM-wide KV cache.
    TT_FATAL(attrs.v_dim > 0 && attrs.v_dim <= K_DIM, "v_dim must be in (0, K_DIM={}] (got {})", K_DIM, attrs.v_dim);
    TT_FATAL(
        attrs.v_dim % tt::constants::TILE_WIDTH == 0,
        "v_dim must be a multiple of {} (got {})",
        tt::constants::TILE_WIDTH,
        attrs.v_dim);
    TT_FATAL(
        K_DIM % tt::constants::TILE_WIDTH == 0,
        "K_DIM (q/kv last dim) must be a multiple of {} (got {})",
        tt::constants::TILE_WIDTH,
        K_DIM);

    // k_chunk_size: multiple of the tile width (it tiles the key axis), divides TOPK
    constexpr uint32_t tile_w = tt::constants::TILE_WIDTH;
    TT_FATAL(
        attrs.k_chunk_size >= tile_w && attrs.k_chunk_size % tile_w == 0,
        "k_chunk_size must be a multiple of {} (got {})",
        tile_w,
        attrs.k_chunk_size);
    TT_FATAL(TOPK % attrs.k_chunk_size == 0, "k_chunk_size must divide TOPK");
    TT_FATAL(attrs.scale > 0.0f, "scale must be > 0");

    // Row-byte alignment: each Q/K/index/output row is a DRAM page the reader DMAs from / the writer DMAs
    // to (and tensors are unpadded, so the page == one row). The row byte count must be a multiple of the
    // DRAM access alignment; that also covers the in-L1 row offsets (DRAM alignment >= L1 alignment).
    // Each Q/K row is one DRAM page in its native dtype (fp8 -> 1 byte, bf16 -> 2). The output dtype
    // matches q (compute_output_specs), so its row width uses q's element size.
    const uint32_t dram_align = tt::tt_metal::hal::get_dram_alignment();
    TT_FATAL((K_DIM * q.element_size()) % dram_align == 0, "Q row bytes must be {}B aligned", dram_align);
    TT_FATAL((K_DIM * kv.element_size()) % dram_align == 0, "K row bytes must be {}B aligned", dram_align);
    TT_FATAL((TOPK * idx.element_size()) % dram_align == 0, "indices row bytes must be {}B aligned", dram_align);
    TT_FATAL((attrs.v_dim * q.element_size()) % dram_align == 0, "output row bytes must be {}B aligned", dram_align);

    // fp8 q/kv tilizes through a 32-bit dest accumulator (fp8 unpacks to fp32 in DEST, packs to bf16/bfp8),
    // so it requires fp32_dest_acc_en. The op factory defaults it on for fp8 inputs (see sparse_sdpa.cpp).
    // (fp32_dest_acc_en is otherwise free: the compute kernel processes the H/TILE_HEIGHT query tile-rows in
    // DST-sized groups, so DST never caps H — only per-core L1 does.)
    TT_FATAL(
        !(kv_is_fp8 || q_is_fp8) || get_fp32_dest_acc_en(attrs.compute_kernel_config),
        "fp8 q/kv requires fp32_dest_acc_en=true (32-bit dest for the fp8 tilize)");
}

SparseSDPAOperation::spec_return_value_t SparseSDPAOperation::compute_output_specs(
    const SparseSDPAParams& attrs, const SparseSDPAInputs& t) {
    auto shape = t.q.logical_shape();  // [1, H, S, K_DIM]
    shape[3] = attrs.v_dim;            // [1, H, S, v_dim]
    // Output dtype matches q (bf16 -> bf16, fp8_e4m3 -> fp8_e4m3): the final untilize packs the bf16
    // accumulator to the output dtype (fp8 is a regular float8, not block-float, so it untilizes fine).
    // The output is always DRAM-interleaved ROW_MAJOR — the writer drains it with per-head-row paged
    // noc writes, so no caller-supplied memory_config is exposed (it could only ever be this).
    const tt::tt_metal::MemoryConfig out_mem{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    return TensorSpec(
        shape, tt::tt_metal::TensorLayout(t.q.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), out_mem));
}

SparseSDPAOperation::tensor_return_value_t SparseSDPAOperation::create_output_tensors(
    const SparseSDPAParams& attrs, const SparseSDPAInputs& t) {
    return create_device_tensor(compute_output_specs(attrs, t), t.q.device());
}

ttsl::hash::hash_t SparseSDPAOperation::compute_program_hash(const SparseSDPAParams& attrs, const SparseSDPAInputs& t) {
    // dtypes + compute_kernel_config MUST be in the hash: q/kv may be bf16 or fp8_e4m3 (different CB
    // formats, row byte widths, fp32-dest/unpack modes, and output dtype), and fp32_dest_acc_en changes
    // the program. Same-shape-different-dtype (or -config) calls otherwise alias to one program and the
    // second silently reuses the first's kernel (wrong results).
    //
    // kv's shape is hashed ONLY when kv is sharded. For an INTERLEAVED kv the per-page address is
    // shape-independent (just page_id, page_size, num_banks), so T rides on the accessor's common runtime
    // args and may change without recompiling. For a SHARDED kv the per-page bank mapping derives from the
    // tensor shape (the shard-grid strides), which is baked into the accessor at create time and is NOT
    // re-emitted on a cache-hit fast path — so a sharded kv MUST recompile when its shape changes, else a hit
    // would reuse stale strides and read the wrong banks. (K_DIM is pinned via q's hashed shape regardless.)
    return tt::tt_metal::operation::hash_operation<SparseSDPAOperation>(
        std::bit_cast<uint32_t>(attrs.scale),
        attrs.v_dim,
        attrs.k_chunk_size,
        attrs.compute_kernel_config,
        t.q.logical_shape(),
        t.q.dtype(),
        t.kv.dtype(),
        // kv.memory_config(): an ND-sharded kv produces different TensorAccessor compile-time args than an
        // interleaved one, so they must be distinct programs.
        t.kv.memory_config(),
        // kv.logical_shape() only when sharded (see above); a fixed sentinel otherwise so interleaved T does
        // not recompile.
        t.kv.memory_config().is_sharded() ? t.kv.logical_shape() : tt::tt_metal::Shape{},
        // Only whether kv is indexed (not which slot): cache_batch_idx's VALUE is a dynamic runtime arg
        // (see get_dynamic_runtime_args), so indexing into a different slot reuses the same program.
        attrs.has_indexed_kv_cache(),
        t.indices.logical_shape(),
        t.indices.dtype());
}

std::vector<tt::tt_metal::DynamicRuntimeArg> SparseSDPAOperation::get_dynamic_runtime_args(
    const SparseSDPAParams& attrs,
    const SparseSDPAInputs& t,
    Tensor& /*output*/,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Non-indexed: the gather page offset is the baked 0 from create_descriptor (and the non-indexed program
    // is never shared with an indexed one — has_indexed_kv_cache() is in the hash — so nothing to re-apply).
    if (!attrs.has_indexed_kv_cache()) {
        return {};
    }
    // Indexed: re-apply kv_batch_page_offset = cache_batch_idx * T to the reader (kernel 0, arg 5) and writer
    // (kernel 1, arg 4) on every dispatch. The value is the same on every core but the slot is per-core, so
    // emit one entry per core per kernel. The core partition mirrors the factory exactly.
    const uint32_t T = t.kv.logical_shape()[2];
    const uint32_t kv_batch_page_offset = attrs.cache_batch_idx.value() * T;
    const tt::tt_metal::CoreCoord grid = t.q.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = grid.x * grid.y;
    // Arg slots/kernel indices are defined once in sparse_sdpa_rt (header), shared with the program factory's
    // emit order so a reorder can't silently desync this re-apply path from the factory.
    std::vector<tt::tt_metal::DynamicRuntimeArg> args;
    args.reserve(2 * num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i % grid.x, i / grid.x};
        args.push_back(
            {sparse_sdpa_rt::kReaderKernelIdx,
             core,
             sparse_sdpa_rt::kReaderBatchOffsetArg,
             kv_batch_page_offset,
             /*is_common=*/false});
        args.push_back(
            {sparse_sdpa_rt::kWriterKernelIdx,
             core,
             sparse_sdpa_rt::kWriterBatchOffsetArg,
             kv_batch_page_offset,
             /*is_common=*/false});
    }
    return args;
}

Tensor sparse_sdpa(
    const Tensor& q,
    const Tensor& kv,
    const Tensor& indices,
    float scale,
    uint32_t v_dim,
    uint32_t k_chunk_size,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx) {
    using OperationType = ttnn::prim::SparseSDPAOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .scale = scale,
            .v_dim = v_dim,
            .k_chunk_size = k_chunk_size,
            .compute_kernel_config = compute_kernel_config,
            .cache_batch_idx = cache_batch_idx,
        },
        OperationType::tensor_args_t{
            .q = q,
            .kv = kv,
            .indices = indices,
        });
}

}  // namespace ttnn::prim
