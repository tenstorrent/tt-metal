// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Program factory for TurboQuant SDPA decode.
// Creates CBs, kernels, and sets compile/runtime args for the fused
// BFP4-dequantize + SDPA decode kernel.

#include "sdpa_tq_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

namespace ttnn::operations::experimental::turbo_quant {

namespace {
uint32_t sdpa_float_to_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits;
}
}  // namespace

SDPATQDeviceOperation::MultiCore::cached_program_t SDPATQDeviceOperation::MultiCore::create(
    const operation_attributes_t& attrs, const tensor_args_t& args, tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    Program program{};

    const auto& q = args.q;
    const auto& k_idx = args.k_indices;
    const auto& k_norms = args.k_norms;
    const auto& v_idx = args.v_indices;
    const auto& v_norms = args.v_norms;
    const auto& page_table = args.page_table;
    const auto& cur_pos = args.cur_pos;

    IDevice* device = q.device();
    [[maybe_unused]] auto grid = device->compute_with_storage_grid_size();

    // ── Dimensions ──
    const uint32_t B = q.padded_shape()[0];
    const uint32_t NQH = q.padded_shape()[1];
    const uint32_t NKH = k_idx.padded_shape()[1];
    const uint32_t DHt = q.padded_shape()[3] / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = v_idx.padded_shape()[3] / tt::constants::TILE_WIDTH;

    // Paged detection: if K indices' batch dim (0) != Q's batch dim, we have the
    // paged layout [max_num_blocks, NKH, block_size, DH]. Otherwise contiguous
    // [B, NKH, max_seq, DH]. Logical Skt comes from the page_table's per-batch
    // page count (paged) or from K_indices' seq dim (contiguous).
    const bool is_paged_attention = (k_idx.padded_shape()[0] != B);
    const bool hybrid_mode = (attrs.recent_window > 0);
    const uint32_t block_size_t = is_paged_attention ? (k_idx.padded_shape()[2] / tt::constants::TILE_HEIGHT) : 0;
    const uint32_t Skt = is_paged_attention ? (page_table.padded_shape()[-1] * block_size_t)
                                            : (k_idx.padded_shape()[2] / tt::constants::TILE_HEIGHT);

    // Chunk sizes (hardcoded for decode: Sq=1 tile, Sk=4 tiles = 128 positions)
    const uint32_t Sq_chunk_t = 1;
    const uint32_t Sk_chunk_t = std::min(Skt, (uint32_t)4);
    const uint32_t k_num_chunks = (Skt + Sk_chunk_t - 1) / Sk_chunk_t;

    const uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    const uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    const uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    const uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    const uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    // ── Tile sizes ──
    auto q_df = datatype_to_dataformat_converter(q.dtype());
    auto k_idx_df = datatype_to_dataformat_converter(k_idx.dtype());
    auto k_norms_df = datatype_to_dataformat_converter(k_norms.dtype());
    auto bf16_df = tt::DataFormat::Float16_b;
    auto im_df = tt::DataFormat::Float16_b;
    // K-split BF16-boundary workaround (2026-05-06): all SDPA intermediates
    // and cross-core merge CBs stay at BF16. The chunk-loop math runs in FP32
    // inside DST (via fp32_dest_acc_en=true), packing back to BF16 on every CB
    // store. The fully-FP32 CB config was tried earlier and ran into an LLK
    // bug in the FP32 unpack-to-dest fast path that produced an interleaved
    // DST layout in the merge cluster — see auto-memory
    // `project_kvcache_kksplit_llk_bug` for the diagnostic trail.

    uint32_t q_tile_size = tile_size(q_df);
    [[maybe_unused]] uint32_t k_idx_tile_size = tile_size(k_idx_df);
    [[maybe_unused]] uint32_t k_norms_tile_size = tile_size(k_norms_df);
    uint32_t bf16_tile_size = tile_size(bf16_df);
    uint32_t im_tile_size = tile_size(im_df);

    // ── Work distribution: batch × heads across compute grid ──
    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores = grid_size.x * grid_size.y;
    uint32_t num_cores_y = grid_size.y;
    CoreRangeSet all_cores({CoreRange(CoreCoord(0, 0), CoreCoord(grid_size.x - 1, grid_size.y - 1))});

    // For decode: q_num_chunks=1, so parallelize over batch × heads only
    uint32_t total_work = B * NQH;
    uint32_t active_cores = std::min(num_cores, total_work);
    uint32_t batch_parallel_factor = std::min(B, active_cores);
    uint32_t nh_parallel_factor = std::min(active_cores / batch_parallel_factor, NQH);
    uint32_t batch_per_core = (B + batch_parallel_factor - 1) / batch_parallel_factor;
    uint32_t nh_per_core = (NQH + nh_parallel_factor - 1) / nh_parallel_factor;

    // Tier 2A: clamp the requested num_cores_per_head to what the grid permits.
    // K = 1 means legacy single-core behavior (no chunk-loop parallelism).
    // Used in Phase 2.2+ to assign K consecutive cores to each (B, NQH) tuple.
    const uint32_t max_cores_per_head = total_work > 0 ? (num_cores / total_work) : 1;
    const uint32_t cores_per_head = std::min(std::max(attrs.num_cores_per_head, (uint32_t)1), max_cores_per_head);
    log_info(
        tt::LogOp,
        "[TQ Phase 2.3] num_cores_per_head req={}, max={}, clamped={} (B={}, NQH={}, num_cores={}, total_work={})",
        attrs.num_cores_per_head,
        max_cores_per_head,
        cores_per_head,
        B,
        NQH,
        num_cores,
        total_work);

    // ── Circular Buffers ──
    // Standard SDPA CBs
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(q_chunk_tiles * q_tile_size, {{CBIndex::c_0, q_df}})
            .set_page_size(CBIndex::c_0, q_tile_size));

    // K/V CBs: both paths now use O(chunk) L1 for 128K+ context.
    //
    // Pre-rescaled: reader pushes native KV format (BFP4) directly to c_1/c_2;
    //   sdpa_standard consumes tile-by-tile. Capacity = 1 chunk.
    //
    // Full dequant: compute interleaves dequant (BFP4+norm → BF16) with SDPA
    //   per-chunk using Flash-Attention online softmax. Capacity = 2 chunks
    //   to double-buffer reader↔compute.
    auto kv_cb_df = attrs.pre_rescaled ? k_idx_df : bf16_df;
    auto kv_cb_tile_size = attrs.pre_rescaled ? k_idx_tile_size : bf16_tile_size;
    uint32_t kv_cb_chunks = attrs.pre_rescaled ? 1 : 2;

    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(k_chunk_tiles * kv_cb_chunks * kv_cb_tile_size, {{CBIndex::c_1, kv_cb_df}})
            .set_page_size(CBIndex::c_1, kv_cb_tile_size));

    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(v_chunk_tiles * kv_cb_chunks * kv_cb_tile_size, {{CBIndex::c_2, kv_cb_df}})
            .set_page_size(CBIndex::c_2, kv_cb_tile_size));

    // Scale + column identity
    uint32_t scalar_tile_size = tile_size(tt::DataFormat::Float16_b);
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(scalar_tile_size, {{CBIndex::c_5, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_5, scalar_tile_size));

    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(scalar_tile_size, {{CBIndex::c_7, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_7, scalar_tile_size));

    // BFP4 index CBs for reader → compute dequant pipeline
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(k_chunk_tiles * k_idx_tile_size, {{CBIndex::c_10, k_idx_df}})
            .set_page_size(CBIndex::c_10, k_idx_tile_size));

    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(Sk_chunk_t * k_norms_tile_size, {{CBIndex::c_11, k_norms_df}})
            .set_page_size(CBIndex::c_11, k_norms_tile_size));

    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(v_chunk_tiles * k_idx_tile_size, {{CBIndex::c_12, k_idx_df}})
            .set_page_size(CBIndex::c_12, k_idx_tile_size));

    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(Sk_chunk_t * k_norms_tile_size, {{CBIndex::c_13, k_norms_df}})
            .set_page_size(CBIndex::c_13, k_norms_tile_size));

    // Dequantize temp: holds one chunk of centroid-gathered BF16 tiles for norm-multiply pass.
    uint32_t dq_temp_tiles = std::max(k_chunk_tiles, v_chunk_tiles);
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(dq_temp_tiles * bf16_tile_size, {{CBIndex::c_14, bf16_df}})
            .set_page_size(CBIndex::c_14, bf16_tile_size));

    // BF16 typecast scratch CBs for norms when stored as BFP8 in DRAM.
    // Compute kernel typecasts c_11 (BFP8) → c_15 (BF16), and c_13 → c_17,
    // then mul_bcast_cols uses the BF16 versions. When norms are already BF16
    // these CBs are unused (the kernel takes a fast path).
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(Sk_chunk_t * bf16_tile_size, {{CBIndex::c_15, bf16_df}})
            .set_page_size(CBIndex::c_15, bf16_tile_size));

    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(Sk_chunk_t * bf16_tile_size, {{CBIndex::c_17, bf16_df}})
            .set_page_size(CBIndex::c_17, bf16_tile_size));

    // SDPA intermediates
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(qk_chunk_tiles * im_tile_size, {{CBIndex::c_24, im_df}})
            .set_page_size(CBIndex::c_24, im_tile_size));

    // Output ping-pong (c_25, c_26)
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(out_chunk_tiles * im_tile_size, {{CBIndex::c_25, im_df}})
            .set_page_size(CBIndex::c_25, im_tile_size));
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(out_chunk_tiles * im_tile_size, {{CBIndex::c_26, im_df}})
            .set_page_size(CBIndex::c_26, im_tile_size));

    // Max/sum ping-pong (c_27-c_30)
    for (auto cb : {CBIndex::c_27, CBIndex::c_28, CBIndex::c_29, CBIndex::c_30}) {
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(Sq_chunk_t * im_tile_size, {{cb, im_df}}).set_page_size(cb, im_tile_size));
    }

    // exp_max_diff (c_31)
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(Sq_chunk_t * im_tile_size, {{CBIndex::c_31, im_df}})
            .set_page_size(CBIndex::c_31, im_tile_size));

    // Output (c_16) — output[0] is always the main attention output. When
    // attrs.return_lse is set, output[1] is the LSE tensor (1 tile per (B, NQH)
    // BF16, populated by the writer kernel from cb_lse_out=c_3).
    auto out_df = datatype_to_dataformat_converter(output[0].dtype());
    uint32_t out_tile_size = tile_size(out_df);
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(out_chunk_tiles * out_tile_size, {{CBIndex::c_16, out_df}})
            .set_page_size(CBIndex::c_16, out_tile_size));

    // ── Tier 2A Phase 2.3: cross-core partial-state CBs ──
    // When num_cores_per_head > 1, each (B, NQH) tuple is split across K worker
    // cores. Worker idx > 0 packs its final (max, sum, out) into c_18/c_19/c_20
    // (cb_partial_max/sum/out, single-slot) and signals a semaphore. The
    // reducer (idx == 0) holds K slots in c_21/c_22/c_23 (cb_remote_max/sum/out),
    // each worker NoC-writes to its own slot (offset = idx * tile_bytes). The
    // reducer compute then merges all K-1 peer slots via online-softmax
    // correction. Slot 0 is unused (reducer keeps its own state in alias_prev_*).
    // c_18 / c_19 — dual role:
    //   Tier-2A mode (num_cores_per_head > 1): cb_partial_max / cb_partial_sum.
    //     Sized as 1 tile each; workers pack their final softmax state here
    //     and the writer NoC-sends it to the reducer's cb_remote_*.
    //   Hybrid mode (recent_window > 0, num_cores_per_head forced 1): repurposed
    //     as the reader's ring K (c_18) and ring V (c_19) data CBs. Sized for
    //     one chunk of the ring's tile format. The Tier-2A and hybrid modes are
    //     mutually exclusive (validation enforces num_cores_per_head == 1 in
    //     hybrid mode), so the reuse is safe.
    if (hybrid_mode) {
        const auto ring_k_df = datatype_to_dataformat_converter(args.k_ring->dtype());
        const auto ring_v_df = datatype_to_dataformat_converter(args.v_ring->dtype());
        const uint32_t ring_k_tile_size = tile_size(ring_k_df);
        const uint32_t ring_v_tile_size = tile_size(ring_v_df);
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(k_chunk_tiles * ring_k_tile_size, {{CBIndex::c_18, ring_k_df}})
                .set_page_size(CBIndex::c_18, ring_k_tile_size));
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(v_chunk_tiles * ring_v_tile_size, {{CBIndex::c_19, ring_v_df}})
                .set_page_size(CBIndex::c_19, ring_v_tile_size));
    } else {
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(Sq_chunk_t * im_tile_size, {{CBIndex::c_18, im_df}})
                .set_page_size(CBIndex::c_18, im_tile_size));
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(Sq_chunk_t * im_tile_size, {{CBIndex::c_19, im_df}})
                .set_page_size(CBIndex::c_19, im_tile_size));
    }
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(out_chunk_tiles * im_tile_size, {{CBIndex::c_20, im_df}})
            .set_page_size(CBIndex::c_20, im_tile_size));
    // Reducer-side CBs: K slots (one per worker) so peers can write in parallel.
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(cores_per_head * Sq_chunk_t * im_tile_size, {{CBIndex::c_21, im_df}})
            .set_page_size(CBIndex::c_21, im_tile_size));
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(cores_per_head * Sq_chunk_t * im_tile_size, {{CBIndex::c_22, im_df}})
            .set_page_size(CBIndex::c_22, im_tile_size));
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(cores_per_head * out_chunk_tiles * im_tile_size, {{CBIndex::c_23, im_df}})
            .set_page_size(CBIndex::c_23, im_tile_size));
    // Scratch CBs for the merge: cb_merge_new_max (c_3) holds max(prev, peer);
    // cb_merge_peer_diff (c_4) holds exp((peer_max - new_max) * scale).
    // cb_exp_max_diff (c_31) is reused as cb_merge_self_diff during the merge.
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(Sq_chunk_t * im_tile_size, {{CBIndex::c_3, im_df}})
            .set_page_size(CBIndex::c_3, im_tile_size));
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(Sq_chunk_t * im_tile_size, {{CBIndex::c_4, im_df}})
            .set_page_size(CBIndex::c_4, im_tile_size));

    // Per-group reducer semaphore: workers `noc_semaphore_inc` to signal they've
    // finished packing partial state; reducer `noc_semaphore_wait` for K-1
    // increments before pulling. One semaphore for the whole program is enough
    // because each core's local L1 holds an independent counter.
    const uint32_t reducer_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, 0);

    // ── Compute kernel ──
    // Matmul config for QK and out
    uint32_t qk_in0_block_w = DHt;
    uint32_t qk_subblock_w = std::min(Sk_chunk_t, (uint32_t)4);
    uint32_t qk_subblock_h = 1;
    uint32_t qk_in0_num_subblocks = 1;
    uint32_t qk_in1_num_subblocks = (Sk_chunk_t + qk_subblock_w - 1) / qk_subblock_w;
    uint32_t qk_num_blocks = 1;

    uint32_t out_in0_block_w = Sk_chunk_t;
    uint32_t out_subblock_w = std::min(vDHt, (uint32_t)4);
    uint32_t out_subblock_h = 1;
    uint32_t out_in0_num_subblocks = 1;
    uint32_t out_in1_num_subblocks = (vDHt + out_subblock_w - 1) / out_subblock_w;
    uint32_t out_num_blocks = 1;

    uint32_t num_levels = static_cast<uint32_t>(attrs.centroids.size());

    std::vector<uint32_t> compute_ct_args = {
        B,
        NQH,
        NKH,
        Skt,
        DHt,
        vDHt,
        Sq_chunk_t,
        1 /*q_num_chunks*/,
        Sk_chunk_t,
        k_num_chunks,
        qk_in0_block_w,
        qk_subblock_w,
        qk_subblock_h,
        qk_in0_num_subblocks,
        qk_in1_num_subblocks,
        qk_num_blocks,
        out_in0_block_w,
        out_subblock_w,
        out_subblock_h,
        out_in0_num_subblocks,
        out_in1_num_subblocks,
        out_num_blocks,
        num_cores,
        0,                                // is_causal = false
        0,                                // use_provided_mask = false
        0,                                // use_padded_mask = false
        1,                                // is_chunked = true (paged)
        sdpa_float_to_bits(attrs.scale),  // scale
        0,                                // sliding_window_size
        0,                                // use_attention_sink
        0,                                // use_streaming_compute
        Skt,                              // valid_Skt
        cores_per_head > 1 ? 1u : 0u,     // fp32_dst_mode (was uniform_dataformat — repurposed; TQ kernel only)
        // TQ args (index 33+)
        num_levels,
    };
    // Append centroid bit-patterns, then pre_rescaled flag
    for (float c : attrs.centroids) {
        compute_ct_args.push_back(sdpa_float_to_bits(c));
    }
    compute_ct_args.push_back(attrs.pre_rescaled ? 1 : 0);
    // Whether norms are BFP8_B (1) or BF16 (0) — compute kernel typecasts when 1.
    const bool norms_are_bfp8 = (k_norms.dtype() == tt::tt_metal::DataType::BFLOAT8_B);
    compute_ct_args.push_back(norms_are_bfp8 ? 1 : 0);
    // return_lse: when 1, kernel packs LSE = max + log(sum) to cb_lse_out (c_3)
    // before the final divide. Used by the sliding-window hybrid host-level
    // combine (see turbo_quant/LSE_COMBINE_DESIGN.md).
    compute_ct_args.push_back(attrs.return_lse ? 1 : 0);
    // Hybrid SDPA: recent_window (=0 disables hybrid) and ring_W_padded
    // (block-aligned ring tensor capacity in tokens). The compute kernel
    // computes the same split_chunk_idx the reader uses and branches the
    // chunk-loop body on TQ vs ring source.
    compute_ct_args.push_back(attrs.recent_window);
    compute_ct_args.push_back(hybrid_mode ? args.k_ring->padded_shape()[0] * args.k_ring->padded_shape()[2] : 0);

    // K-split FP32 merge-boundary DPRINTs: enable by setting TT_TQ_DPRINT_KSPLIT=1
    // in the host environment. Off by default (no overhead in production builds).
    std::map<std::string, std::string> compute_defines = {{"SDPA_TQ_DECODE", "1"}};
    if (const char* dbg = std::getenv("TT_TQ_DPRINT_KSPLIT"); dbg != nullptr && std::string(dbg) != "0") {
        compute_defines["TQ_DPRINT_KSPLIT"] = "1";
    }
    KernelHandle compute_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/turbo_quant/sdpa/kernels/compute/sdpa_tq_decode.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            // FP32 dst-acc is required for the K-split (Tier-2A) merge path because the
            // copy_dest_values cascade in dequant_*_chunk reads back DST registers, which
            // need FP32 width to preserve the integer-encoded indices across the cascade.
            // For the legacy single-core path (K==1 = Track A) and the new hybrid fused
            // path (also K==1), enabling FP32 dst-acc regresses end-to-end token accuracy
            // from ~82% top-1 to 0% — bisected to commit 8fe227b. Default OFF; flip ON
            // only when the K-split merge actually runs.
            .fp32_dest_acc_en = (cores_per_head > 1),
            .math_approx_mode = true,
            .compile_args = compute_ct_args,
            .defines = compute_defines});

    // ── Page table CB (paged mode only) ──
    // Single page_table row (covers one batch slot). For B cores each reading their
    // own batch, 1 page_table page of size page_size_bytes = max_pages_per_batch * 4.
    const uint32_t page_table_page_size = is_paged_attention ? (page_table.padded_shape()[-1] * sizeof(uint32_t)) : 0;
    if (is_paged_attention) {
        auto page_table_df = datatype_to_dataformat_converter(page_table.dtype());
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(page_table_page_size, {{CBIndex::c_9, page_table_df}})
                .set_page_size(CBIndex::c_9, page_table_page_size));
    }

    // ── cur_pos CB (c_8) ──
    // Holds B int32 values (current decode position per batch). Reader reads
    // the cur_pos tensor into this CB once at kernel start; compute consumes
    // to derive a valid_k_chunks loop bound (vs. iterating the full padded
    // cache, which is the whole point of this fix).
    const uint32_t cur_pos_stick_size = cur_pos.buffer()->aligned_page_size();
    auto cur_pos_df = datatype_to_dataformat_converter(cur_pos.dtype());
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(cur_pos_stick_size, {{CBIndex::c_8, cur_pos_df}})
            .set_page_size(CBIndex::c_8, cur_pos_stick_size));

    // ── Reader kernel ──
    // Hybrid plumbing: when recent_window > 0, the reader needs accessors for
    // the ring tensors. We always append ring tensor accessor CT args (using
    // the TQ tensors as a placeholder when not in hybrid mode) so the kernel's
    // CT-arg layout is constant. ring_W_padded is the block-aligned ring
    // capacity; when recent_window=0 it's also 0.
    const auto* k_ring_buffer = hybrid_mode ? args.k_ring->buffer() : k_idx.buffer();
    const auto* v_ring_buffer = hybrid_mode ? args.v_ring->buffer() : v_idx.buffer();
    const auto* ring_pt_buffer = hybrid_mode ? args.ring_page_table->buffer() : page_table.buffer();
    const uint32_t ring_block_size_t =
        hybrid_mode ? (args.k_ring->padded_shape()[2] / tt::constants::TILE_HEIGHT) : block_size_t;
    const uint32_t ring_W_padded = hybrid_mode ? args.k_ring->padded_shape()[0] * args.k_ring->padded_shape()[2] : 0;

    std::vector<uint32_t> reader_ct_args = {
        B,
        NQH,
        NKH,
        1 /*Sqt*/,
        Skt,
        DHt,
        vDHt,
        Sq_chunk_t,
        Sk_chunk_t,
        k_num_chunks,
        num_cores,
        attrs.pre_rescaled ? 1u : 0u,
        is_paged_attention ? 1u : 0u,
        block_size_t,
        attrs.recent_window,
        ring_W_padded,
        ring_block_size_t,
    };
    TensorAccessorArgs(*q.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*k_idx.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*k_norms.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*v_idx.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*v_norms.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*cur_pos.buffer()).append_to(reader_ct_args);
    // page_table TensorAccessor args — always append (even in contiguous mode, reader
    // ignores them behind the `if constexpr (is_paged_attention)` gate, but the offset
    // calculation must be consistent).
    TensorAccessorArgs(*page_table.buffer()).append_to(reader_ct_args);
    // Ring tensor accessor CT args. Always appended so the reader's CT-arg
    // layout is identical in legacy and hybrid modes; placeholders point at
    // TQ tensors when recent_window == 0.
    TensorAccessorArgs(*k_ring_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(*v_ring_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(*ring_pt_buffer).append_to(reader_ct_args);

    KernelHandle reader_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/turbo_quant/sdpa/kernels/dataflow/reader_tq_decode.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args));

    // ── Writer kernel (custom simple writer) ──
    std::vector<uint32_t> writer_ct_args = {
        B,
        NQH,
        Sq_chunk_t,
        vDHt,
        num_cores,
        [&]() -> uint32_t {
            // Pack scale as BF16 doubled: (bf16 << 16) | bf16
            // This matches what generate_reduce_scaler expects.
            uint32_t f32_bits = sdpa_float_to_bits(1.0f);  // identity scalar = 1.0
            uint16_t bf16 = static_cast<uint16_t>(f32_bits >> 16);
            return (static_cast<uint32_t>(bf16) << 16) | bf16;
        }(),
        attrs.return_lse ? 1u : 0u,
    };
    TensorAccessorArgs(*output[0].buffer()).append_to(writer_ct_args);
    // Always append a TensorAccessorArgs for the LSE buffer. When return_lse
    // is false, output[1] doesn't exist — fall back to output[0]'s accessor as
    // a placeholder (the kernel's `if constexpr (return_lse)` branch elides
    // the LSE write so the placeholder is never used).
    const auto& lse_buffer_for_accessor = attrs.return_lse ? *output[1].buffer() : *output[0].buffer();
    TensorAccessorArgs(lse_buffer_for_accessor).append_to(writer_ct_args);

    KernelHandle writer_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/turbo_quant/sdpa/kernels/dataflow/writer_tq_decode.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    // ── Runtime args: distribute (batch × heads) tuples across cores ──
    // Tier 2A Phase 2.2b: each tuple gets `cores_per_head` consecutive cores.
    // With cores_per_head == 1, this is identical to the legacy mapping
    // (group_id == i, core_idx_in_group == 0). Phase 2.3+ will activate K > 1
    // and add the cross-core reduce; for now the cores_per_head runtime arg
    // sent to the kernel is forced to 1, so multi-core-per-tuple groups still
    // process the full chunk range on the worker (idx 0) and the other cores
    // receive empty work ranges.
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        const uint32_t group_id = i / cores_per_head;
        const uint32_t core_idx_in_group = i % cores_per_head;

        uint32_t batch_start = (group_id / nh_parallel_factor) * batch_per_core;
        uint32_t batch_end = std::min(batch_start + batch_per_core, B);
        uint32_t head_start = (group_id % nh_parallel_factor) * nh_per_core;
        uint32_t head_end = std::min(head_start + nh_per_core, NQH);

        // Clamp: cores beyond active_cores get empty ranges
        batch_start = std::min(batch_start, B);
        head_start = std::min(head_start, NQH);

        // Tier 2A Phase 2.3 step 5: workers (idx > 0) now share the same
        // (batch, head) range as their reducer (idx == 0) — both process the
        // SAME (B, NQH) tuple, just different chunk slices. The empty-range
        // guard from Phase 2.2b is removed.

        // Tier 2A Phase 2.3 step 5: pass real (core_idx_in_group, cores_per_head)
        // to the kernels. The reader/compute slice the chunk loop accordingly,
        // workers pack-and-skip, and the reducer waits-pulls-merges.
        const uint32_t kernel_core_idx_in_group = core_idx_in_group;
        const uint32_t kernel_cores_per_head = cores_per_head;

        // Ring runtime addresses. When recent_window == 0, fall back to TQ
        // addresses so the placeholder accessors don't read garbage. The
        // reader gates ring usage on the recent_window CT arg, so these
        // addresses are dormant unless hybrid mode is active.
        const uint32_t k_ring_addr = hybrid_mode ? args.k_ring->buffer()->address() : k_idx.buffer()->address();
        const uint32_t v_ring_addr = hybrid_mode ? args.v_ring->buffer()->address() : v_idx.buffer()->address();
        const uint32_t ring_pt_addr = hybrid_mode ? args.ring_page_table->buffer()->address()
                                                  : (is_paged_attention ? page_table.buffer()->address() : 0u);

        SetRuntimeArgs(
            program,
            reader_kernel,
            core,
            {
                q.buffer()->address(),
                k_idx.buffer()->address(),
                k_norms.buffer()->address(),
                v_idx.buffer()->address(),
                v_norms.buffer()->address(),
                is_paged_attention ? page_table.buffer()->address() : 0u,
                page_table_page_size,
                cur_pos.buffer()->address(),
                cur_pos_stick_size,
                i,  // core_id
                batch_start,
                batch_end,
                head_start,
                head_end,
                kernel_core_idx_in_group,  // Tier 2A: chunk-slice routing (mirrors compute kernel slot [7])
                kernel_cores_per_head,     // Tier 2A: chunk-slice routing (mirrors compute kernel slot [8])
                k_ring_addr,               // Hybrid: ring K base address (= k_idx.address() in legacy mode)
                v_ring_addr,               // Hybrid: ring V base address (= v_idx.address() in legacy mode)
                ring_pt_addr,              // Hybrid: ring page-table base address
            });

        SetRuntimeArgs(
            program,
            compute_kernel,
            core,
            {
                i,                         // [0]  core_id
                batch_start,               // [1]
                batch_end,                 // [2]
                head_start,                // [3]
                head_end,                  // [4]
                (uint32_t)0,               // [5]  local_q_start
                (uint32_t)1,               // [6]  local_q_end
                kernel_core_idx_in_group,  // [7]  Tier 2A: chunk-slice routing (forced 0 until reduce lands)
                kernel_cores_per_head,     // [8]  Tier 2A: chunk-slice routing (forced 1 until reduce lands)
                reducer_semaphore_id,      // [9]  Tier 2A: per-program semaphore (worker→reducer signal)
            });

        // Tier 2A: physical NoC coords of this group's reducer (always
        // core_idx_in_group == 0 within the group). Workers send their partials
        // to this address; reducer self-writes (no-op via NoC).
        const uint32_t reducer_logical_core_id = group_id * cores_per_head;
        const CoreCoord reducer_logical = {
            reducer_logical_core_id % grid_size.x, reducer_logical_core_id / grid_size.x};
        const auto reducer_physical = device->worker_core_from_logical_core(reducer_logical);

        const uint32_t lse_addr = attrs.return_lse ? output[1].buffer()->address() : 0u;
        SetRuntimeArgs(
            program,
            writer_kernel,
            core,
            {
                output[0].buffer()->address(),
                i,
                batch_start,
                batch_end,
                head_start,
                head_end,
                kernel_core_idx_in_group,                   // [6] Tier 2A: chunk-slice routing
                kernel_cores_per_head,                      // [7] Tier 2A: chunk-slice routing
                static_cast<uint32_t>(reducer_physical.x),  // [8] Tier 2A: this group's reducer NoC x
                static_cast<uint32_t>(reducer_physical.y),  // [9] Tier 2A: this group's reducer NoC y
                reducer_semaphore_id,                       // [10] Tier 2A: per-program semaphore id
                lse_addr,                                   // [11] LSE output buffer (0 if !return_lse)
            });
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel,
         .compute_kernel_id = compute_kernel,
         .writer_kernel_id = writer_kernel,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y,
         .grid_size_x = grid_size.x}};
}

void SDPATQDeviceOperation::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attrs,
    const tensor_args_t& args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto num_cores = cached_program.shared_variables.num_cores;
    auto grid_size_x = cached_program.shared_variables.grid_size_x;

    const bool hybrid_mode = (attrs.recent_window > 0);
    const bool is_paged = (args.k_indices.padded_shape()[0] != args.q.padded_shape()[0]);
    const uint32_t k_ring_addr = hybrid_mode ? args.k_ring->buffer()->address() : args.k_indices.buffer()->address();
    const uint32_t v_ring_addr = hybrid_mode ? args.v_ring->buffer()->address() : args.v_indices.buffer()->address();
    const uint32_t ring_pt_addr =
        hybrid_mode ? args.ring_page_table->buffer()->address() : (is_paged ? args.page_table.buffer()->address() : 0u);

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i % grid_size_x, i / grid_size_x};

        auto& reader_args = GetRuntimeArgs(program, cached_program.shared_variables.reader_kernel_id, core);
        reader_args[0] = args.q.buffer()->address();
        reader_args[1] = args.k_indices.buffer()->address();
        reader_args[2] = args.k_norms.buffer()->address();
        reader_args[3] = args.v_indices.buffer()->address();
        reader_args[4] = args.v_norms.buffer()->address();
        // args[5] = page_table buffer address (0 if not paged).
        // args[6] = page_table_page_size (constant — set in create, not touched here).
        reader_args[5] = is_paged ? args.page_table.buffer()->address() : 0u;
        // args[7] = cur_pos buffer address (refresh per call so the kernel reads
        // the latest position values).
        // args[8] = cur_pos_stick_size (constant — set in create).
        reader_args[7] = args.cur_pos.buffer()->address();
        // args[9..15] = core-distribution / Tier 2A routing slots (constant — set in create).
        // args[16..18] = ring K, V, page_table addresses (refresh per call so a
        // moved ring buffer is picked up; in legacy mode they alias the TQ
        // tensors and the reader doesn't read them).
        reader_args[16] = k_ring_addr;
        reader_args[17] = v_ring_addr;
        reader_args[18] = ring_pt_addr;

        auto& writer_args = GetRuntimeArgs(program, cached_program.shared_variables.writer_kernel_id, core);
        writer_args[0] = output[0].buffer()->address();
        // writer_args[11] = LSE buffer address (refresh in case it moved between
        // launches). Untouched when return_lse=false because the kernel never reads it.
        if (output.size() > 1) {
            writer_args[11] = output[1].buffer()->address();
        }
    }
}

}  // namespace ttnn::operations::experimental::turbo_quant
