// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Gemma4 sliding ring SDPA's copy of sdpa_ring_v2 (compute_streaming.hpp). The rest of the streaming
// primitives stay shared; only the ring loop is forked so Gemma4-only changes do not touch the shared op.
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"

namespace gemma4_sliding {

/**
 * Streaming Ring SDPA (v2): Ring-aware variant of sdpa_standard_v2 with deferred normalization.
 * Accumulates raw (un-normalized) softmax state across ring iterations; normalizes once on the
 * last K chunk of the last ring iteration. Single Q-chunk: accumulators persist in L1 across
 * ring iterations (no DRAM traffic). Multi Q-chunk: accumulators round-trip through DRAM.
 *
 * @tparam Sq_chunk_t       Q chunk size in tiles
 * @tparam Sk_chunk_t       K chunk size in tiles
 * @tparam Skt              Not used for ring (pass 0)
 * @tparam DHt              Head dimension in tiles
 * @tparam vDHt             V head dimension in tiles (== DHt for ring)
 * @tparam scale_fp32       Attention scale factor as raw uint32_t bits
 * @tparam qkt_subblock_h   QK matmul subblock height
 * @tparam cb_max_in      CB for restoring row-max from DRAM (multi Q-chunk, c_6)
 * @tparam cb_max_out     CB for saving row-max to DRAM (multi Q-chunk, c_17)
 * @tparam cb_normalized_out CB for normalized output rows (written by normalize_row_streaming)
 * @tparam cb_sum_out       CB for saving row-sum to DRAM (multi Q-chunk, c_10)
 * @tparam cb_sum_in        CB for restoring row-sum from DRAM (multi Q-chunk, c_11)
 * @tparam local_padded_Nt   Per-device KV padded sequence length in tiles (N_local / TILE_H)
 * @tparam q_local_padded_Nt Per-device Q padded sequence length in tiles.
 * @tparam kv_region_Nt      Per-source block-cyclic KV region; Q slab divided by the stripe split.
 * @tparam chunk_size_t      Per-chunk Q/K extent in tiles (chunked-only)
 *
 * @param global_q_start     First global Q chunk index for this core
 * @param global_q_end       One-past-last global Q chunk index for this core
 * @param num_kv_chunks      Total K chunks this ring iter (local + joint if applicable)
 * @param ring_iter          Current ring iteration (0..ring_size-1)
 * @param ring_id            Device ID within the ring that owns this iter's KV shard
 * @param num_local_k_chunks Number of K chunks from the local (non-joint) sequence
 * @param logical_nt         Actual (unpadded) global sequence length in tiles
 * @param acc_state          Persistent accumulator state (prev/cur CB halves for ping-pong)
 * @param is_last_ring_iter  True on the final ring iteration — triggers normalization
 * @param q_per_core         Number of Q chunks per core (1 = L1-only, >1 = DRAM round-trip)
 * @param lw_mask            Lightweight mask context for partial-tile padding
 * @param chunked            Chunked-prefill runtime state (q_start_idx_t, ring_index); ignored when
 * chunked_enabled=false
 * @param is_first_active_iter Set on the first ring iter that does work (decoupled from ring_iter==0 for skipped-chain
 * bounds)
 */
template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t Skt,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t scale_fp32,
    uint32_t qkt_subblock_h,
    uint32_t qkt_subblock_w,
    uint32_t qktv_subblock_h,
    uint32_t qktv_subblock_w,
    uint32_t cb_q_in,
    uint32_t cb_kt_in,
    uint32_t cb_v_in,
    uint32_t cb_qkt_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_exp_max_diff,
    uint32_t cb_col_identity,
    uint32_t cb_recip_scratch,
    uint32_t cb_mask_in,
    uint32_t cb_scale_in,
    uint32_t cb_max_in,
    uint32_t cb_max_out,
    uint32_t cb_prev_out,
    uint32_t cb_out,
    uint32_t cb_normalized_out = 0,
    uint32_t cb_sum_out = 0,
    uint32_t cb_sum_in = 0,
    uint32_t cb_signal = 0,
    bool lightweight_mask_enabled = false,
    bool is_causal_sdpa = false,
    bool is_balanced_sdpa = false,
    bool chunked_enabled = false,
    uint32_t local_padded_Nt = 0,
    uint32_t q_local_padded_Nt = 0,
    uint32_t chunk_size_t = 0,
    bool global_n_mask_enabled = false,
    bool local_n_mask_enabled = false,
    bool joint_n_mask_enabled = false,
    bool straddle_mask_enabled = false,
    // Head-serial passes: keep each pass's flash-attention state in an L1 FIFO
    // ({cb_sum_in, cb_max_in, cb_prev_out}, num_passes+1 entries deep) instead of the caller's
    // ping-pong halves. Compile-time so non-FIFO callers (normal ring joint) pay zero MATH-thread
    // instructions for the FIFO branches — the utilization-band perf gates resolve even
    // fraction-of-a-percent overhead on this path.
    bool use_l1_state_fifo = false,
    bool kv_pad_rotation_enabled = false,
    uint32_t v_cb_physical_width_t = vDHt,
    bool v_shares_k_buffer = false,
    bool kt_inplace_v = false,
    uint32_t sliding_window_size = 0,
    // Circular sliding KV cache slab count (0/1 = unbounded); wraps the sliding work
    // plan's local slab addressing. See sliding_window_work_plan.hpp.
    uint32_t circular_kv_slab_count = 0,
    uint32_t ring_size = 1,
    bool use_attention_sink = false,
    uint32_t cb_attention_sink = INVALID_CB,
    // Sharded joint: skip joint K chunks that lie entirely in the padded joint tail, mirroring the
    // reader's kv_chunk_is_beyond_logical_l skip so the K/V CB producer/consumer counts stay aligned.
    bool joint_n_skip_enabled = false,
    uint32_t joint_local_padded_Nt = 0,  // Lt_local: per-device joint tile count (sharded path)
    // Rotated Q split: map q-loop positions through the fixed base range and moving remainder.
    // Off by default (position is the flat id).
    bool rotated_q_split_enabled = false,
    // Dense chunked prefill: skip K chunks past this device's last Q row (the reader mirrors it).
    bool causal_k_skip = false,
    uint32_t kv_region_Nt = q_local_padded_Nt,
    // Split-KV: GROUPED_KV_SOURCE_COUNT > 1. Keeps the per-source path free of packed branches.
    bool packed_kv_enabled = false,
    typename MaskCtx = LightweightMaskContext>
void sdpa_ring_v2(
    const uint32_t global_q_start,
    const uint32_t global_q_end,
    const uint32_t num_kv_chunks,
    const uint32_t num_q_chunks,
    const uint32_t ring_iter,
    const uint32_t ring_id,
    const uint32_t num_local_k_chunks,
    const uint32_t logical_nt,
    const bool ring_iter_needs_global_n_mask,
    const bool ring_iter_needs_joint_n_mask,
    const bool local_n_needs_masking,
    const uint32_t global_n_mask_chunk_id,
    const uint32_t local_n_mask_chunk_id,
    const uint32_t joint_n_mask_chunk_id,
    RingAccumulatorState& acc_state,
    const bool is_last_ring_iter = false,
    const uint32_t q_per_core = 1,
    const MaskCtx& lw_mask = {},
    const bool skip_first_half_q = false,
    const bool use_zigzag_balancing = false,
    const ChunkedContext& chunked = {},
    const bool is_first_active_iter = true,
    // True (unpadded) joint length in tiles; joint K chunks starting at/after it are pure padding.
    const uint32_t logical_lt = 0,
    // Tile offset of this call's Q chunk within cb_q_in (head-serial passes; 0 otherwise).
    const uint32_t q_base_tiles = 0,
    // Rotated Q split only: fixed base range plus this iteration's remainder unit.
    [[maybe_unused]] const RotatedQSlots& rotated_slots = {},
    // K split only: the local K chunks this core attends to (the reader skips the same).
    const uint32_t k_split_begin = 0,
    const uint32_t k_split_end = 0xFFFFFFFFu,
    // Sliding K split only: this core's band of the work plan (ring_joint_ksplit.hpp).
    const uint32_t sliding_split_idx = 0,
    const uint32_t sliding_split_count = 1,
    const uint32_t* packed_source_ids = nullptr,
    const uint32_t packed_source_count = 0,
    // Rows of newest slabs this pass appends (packed_kv_pass_rows).
    const uint32_t packed_newest_rows = 0,
    // Padded multicast slots the reader runs after this core's Q chunks; their K/V entries are drained.
    const uint32_t padded_q_slots = 0) {
    init_sdpa_streaming_semaphores();
    const bool packed_sources = packed_kv_enabled && packed_source_count > 1;
    const ring_joint::PackedKVGroupPlan packed_kv = ring_joint::packed_kv_pass_plan(
        {ring_joint::packed_kv_source_tiles(local_padded_Nt, logical_nt, kv_region_Nt, ring_size),
         packed_sources ? packed_source_count : 1,
         Sk_chunk_t,
         kv_region_Nt},
        packed_sources ? packed_newest_rows : 0);
    // Global end of the head slabs: every head tile lies below it.
    const uint32_t packed_head_global_end = packed_sources ? packed_kv.head_tiles() / kv_region_Nt * chunk_size_t : 0;

    constexpr bool has_sliding_window = sliding_window_size > 0;
    static_assert(!has_sliding_window || chunked_enabled, "Sliding windows require chunked prefill");
    // is_causal: diagonal stamp only on iter 0 (K is local-frame). Chunked: every iter (absolute coords).
    const bool is_causal_iter = (is_causal_sdpa && (ring_iter == 0)) || chunked_enabled;

    // reduce_trigger enables early reduce start via semaphore signaling from packer to unpacker.
    // All conditions are compile-time except the active_Sk == Sk_chunk_t guard (padded chunks).
    constexpr bool can_reduce_trigger = reduce_trigger_supported && (Sk_chunk_t % qkt_subblock_w == 0) &&
                                        (Sk_chunk_t / qkt_subblock_w > 1) && (Sk_chunk_t % 2 == 0);

    // Subblock width for the non-padded (common) case — compile-time constant.
    constexpr uint32_t full_sbw = qkt_subblock_w;

    // Pre-compute subblock widths for each mask type (hoisted out of per-chunk loop).
    uint32_t global_n_sbw = full_sbw;
    uint32_t local_n_sbw = full_sbw;
    uint32_t joint_n_sbw = full_sbw;
    uint32_t straddle_sbw = full_sbw;
    if constexpr (global_n_mask_enabled) {
        global_n_sbw = lw_mask.global_n_padded_tiles
                           ? largest_factor_le(Sk_chunk_t - lw_mask.global_n_padded_tiles, qkt_subblock_w)
                           : full_sbw;
    }
    if constexpr (local_n_mask_enabled) {
        local_n_sbw = lw_mask.local_n_padded_tiles
                          ? largest_factor_le(Sk_chunk_t - lw_mask.local_n_padded_tiles, qkt_subblock_w)
                          : full_sbw;
    }
    if constexpr (joint_n_mask_enabled) {
        joint_n_sbw = lw_mask.joint_n_padded_tiles
                          ? largest_factor_le(Sk_chunk_t - lw_mask.joint_n_padded_tiles, qkt_subblock_w)
                          : full_sbw;
    }
    if constexpr (straddle_mask_enabled) {
        straddle_sbw = lw_mask.straddle_num_padded_tiles
                           ? largest_factor_le(Sk_chunk_t - lw_mask.straddle_num_padded_tiles, qkt_subblock_w)
                           : full_sbw;
    }

    uint32_t KV_chunks_processed_in_iter = 0;

    // ---- Q-loop helpers ---------------------------------------------------

    // Non-last ring iter: drain restored staging CBs and skip this Q chunk.
    auto try_balanced_skip = [&](uint32_t q_chunk) -> bool {
        if constexpr (is_balanced_sdpa) {
            if (skip_first_half_q && q_chunk < num_q_chunks / 2 && !is_last_ring_iter) {
                return true;
            }
        }
        return false;
    };

    auto normalize_only = [&]() {
        AccumulatorHalf q_prev_norm = acc_state.prev;
        if (q_per_core > 1 && ring_iter > 0) {
            q_prev_norm = {cb_sum_in, cb_max_in, cb_prev_out};
        }
        constexpr uint32_t norm_dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
        normalize_row_streaming<
            false,
            vDHt,
            norm_dst_size,
            cb_col_identity,
            cb_recip_scratch,
            cb_normalized_out,
            scale_fp32,
            use_attention_sink,
            cb_attention_sink>(q_prev_norm.sum, q_prev_norm.out, Sq_chunk_t, q_prev_norm.max);
        if constexpr (use_attention_sink) {
            CircularBuffer(cb_attention_sink).pop_front(1);
        }
        sdpa_cb_pop_front_out_of_line(q_prev_norm.max, Sq_chunk_t);
        if (q_per_core > 1) {
            CircularBuffer(cb_signal).reserve_back(1);
            sdpa_cb_push_back_out_of_line(cb_signal, 1);
        }
    };

    // Last ring iter: normalize accumulated state and signal writer, then skip K-loop.
    auto try_normalize_only = [&](uint32_t q_chunk) -> bool {
        if constexpr (is_balanced_sdpa) {
            if (skip_first_half_q && q_chunk < num_q_chunks / 2 && is_last_ring_iter) {
                normalize_only();
                return true;
            }
        }
        return false;
    };

    // ---- K-loop helpers ---------------------------------------------------

    [[maybe_unused]] const uint32_t causal_end_nt =
        !causal_k_skip ? 0
                       : chunked_q_global_end_tile<kv_pad_rotation_enabled, q_local_padded_Nt>(
                             logical_nt,
                             chunked.ring_index,
                             ring_size,
                             chunked.kv_pad_rotation.q_pre_wrap_start_tile,
                             chunked.kv_pad_rotation.q_pre_wrap_tile_count,
                             chunked.kv_pad_rotation.q_post_wrap_start_tile,
                             chunked.kv_pad_rotation.q_valid_tile_count);

    // Skip KV chunks beyond the logical sequence length (padding tiles).
    auto try_skip_oob_kv = [&](uint32_t source_ring_id, uint32_t k_chunk, bool kv_chunk_is_joint) -> bool {
        if (kv_chunk_is_joint) {
            // Skip joint chunk at/after logical_lt - pure padding
            if constexpr (joint_n_skip_enabled) {
                const uint32_t joint_global_start_tile =
                    ring_id * joint_local_padded_Nt + (k_chunk - num_local_k_chunks) * Sk_chunk_t;
                return joint_global_start_tile >= logical_lt;
            }
            return false;
        }
        if (k_chunk < k_split_begin || k_chunk >= k_split_end) {
            return true;
        }
        if constexpr (chunked_enabled && !has_sliding_window && causal_k_skip) {
            return !chunked_kv_chunk_is_live<
                kv_pad_rotation_enabled,
                chunked_enabled,
                local_padded_Nt,
                chunk_size_t,
                q_local_padded_Nt>(source_ring_id, k_chunk, Sk_chunk_t, logical_nt, causal_end_nt);
        }
        return !kv_chunk_starts_before_logical_end<
            kv_pad_rotation_enabled,
            chunked_enabled,
            local_padded_Nt,
            chunk_size_t,
            kv_region_Nt>(source_ring_id, k_chunk * Sk_chunk_t, logical_nt);
    };

    // Causal skip: K chunks fully above the diagonal — drain K/V from CBs and skip.
    auto try_skip_causal_above_diag = [&](uint32_t k_chunk, uint32_t causal_k_limit) -> bool {
        if constexpr (is_causal_sdpa) {
            if constexpr (!has_sliding_window) {
                if (is_causal_iter && k_chunk >= causal_k_limit) {
                    CircularBuffer(cb_kt_in).wait_front(DHt * Sk_chunk_t);
                    sdpa_cb_pop_front_out_of_line(cb_kt_in, DHt * Sk_chunk_t);
                    // In-place latent-V never pushes a V entry, so only K^T needs draining.
                    if constexpr (!kt_inplace_v) {
                        CircularBuffer(cb_v_in).wait_front(Sk_chunk_t * v_cb_physical_width_t);
                        sdpa_cb_pop_front_out_of_line(cb_v_in, Sk_chunk_t * v_cb_physical_width_t);
                    }
                    KV_chunks_processed_in_iter++;
                    return true;
                }
            }
        }
        return false;
    };

    // -----------------------------------------------------------------------

    for (uint32_t q = global_q_start; q < global_q_end; q++) {
        // Compute Q chunk index (with optional zigzag remapping for causal balancing).
        // num_q_chunks is total per-head chunks (local + joint), matching the divisor the
        // writer/reader use to flatten (batch, head, q_chunk) — see ring_joint_sdpa.cpp.
        uint32_t q_flat = q;
        if constexpr (rotated_q_split_enabled) {
            q_flat = rotated_slots.at(q);
        }
        uint32_t q_chunk = remap_q_index(q_flat, num_q_chunks, use_zigzag_balancing) % num_q_chunks;

        // Causal K-chunk limit and Q start tile for this Q chunk
        uint32_t causal_k_limit = num_kv_chunks;
        uint32_t q_start_tile = 0;
        if constexpr (has_sliding_window) {
            q_start_tile = logical_nt - ring_size * q_local_padded_Nt + chunked.ring_index * q_local_padded_Nt +
                           q_chunk * Sq_chunk_t;
        } else if constexpr (is_causal_sdpa) {
            if (is_causal_iter) {
                q_start_tile = q_chunk * Sq_chunk_t;
                causal_k_limit = (q_start_tile + Sq_chunk_t + Sk_chunk_t - 1) / Sk_chunk_t;
            }
        } else if constexpr (chunked_enabled) {
            // Absolute Q tile row. Diag stamp masks K past Q's range; logical_n skip handles K past the cache.
            q_start_tile = chunked.q_start_idx_t + chunked.ring_index * q_local_padded_Nt + q_chunk * Sq_chunk_t;
        }

        if (try_balanced_skip(q_chunk)) {
            continue;
        }
        if (try_normalize_only(q_chunk)) {
            continue;
        }

        ttnn::operations::transformer::sdpa::ring_joint::SlidingQWorkPlan<sliding_q_plan_max_source_ranges>
            sliding_q_plan;
        constexpr bool circular_kv_cache = circular_kv_slab_count > 1;
        if constexpr (has_sliding_window) {
            const ttnn::operations::transformer::sdpa::ring_joint::ChunkedQMapping q_mapping{
                chunked.kv_pad_rotation.q_pre_wrap_start_tile,
                chunked.kv_pad_rotation.q_pre_wrap_tile_count,
                chunked.kv_pad_rotation.q_post_wrap_start_tile,
                chunked.kv_pad_rotation.q_valid_tile_count};
            sliding_q_plan = ttnn::operations::transformer::sdpa::ring_joint::build_sliding_q_work_plan<
                sliding_q_plan_max_source_ranges>(
                q_chunk * Sq_chunk_t,
                Sq_chunk_t,
                chunked.ring_index,
                q_local_padded_Nt,
                ring_size,
                sliding_window_size,
                TILE_HEIGHT,
                local_padded_Nt,
                Sk_chunk_t,
                logical_nt,
                circular_kv_slab_count,
                kv_pad_rotation_enabled ? &q_mapping : nullptr,
                sliding_local_first && sliding_split_count == 1);
            ASSERT(sliding_q_plan.is_valid);
            acc_state.sliding_plan_k_chunks = sliding_q_plan.total_k_chunk_count;
            ASSERT(sliding_q_plan.total_k_chunk_count > 0);
        }
        // Per-Q pre-scan: count K chunks that will actually be processed.
        // Placed after balanced-skip guards so skipped Q chunks don't pay for the scan.
        const auto sliding_band = ttnn::operations::transformer::sdpa::ring_joint::sliding_ksplit_range(
            sliding_q_plan.total_k_chunk_count, sliding_split_idx, sliding_split_count);
        uint32_t per_q_valid_kv =
            packed_sources ? packed_kv.chunk_count() : (has_sliding_window ? sliding_band.end - sliding_band.begin : 0);
        for (uint32_t k = 0; !packed_sources && !has_sliding_window && k < num_kv_chunks; ++k) {
            const uint32_t source_ring_id = ring_id;
            const uint32_t source_k_chunk = k;
            const bool is_joint = k >= num_local_k_chunks;
            if (try_skip_oob_kv(source_ring_id, source_k_chunk, is_joint)) {
                continue;
            }
            if constexpr (is_causal_sdpa) {
                if (is_causal_iter && k >= causal_k_limit) {
                    continue;
                }
            }
            per_q_valid_kv++;
        }
        // Use persistent accumulator state from caller (single Q-chunk),
        // restore from DRAM staging (multi Q-chunk), or pop from the L1 FIFO (head-serial passes).
        AccumulatorHalf q_prev = acc_state.prev, q_cur = acc_state.cur;

        const bool is_first_kv_for_this_q = is_first_active_iter;

        // Multi Q-chunk restore: K0 reads prev accumulators directly from staging buffers
        // (cb_prev_out, cb_max_in, cb_sum_in) — no copy_block needed.
        // After K0's swap, reset q_cur to original accumulator CBs for normal ping-pong.
        const AccumulatorHalf original_prev = q_prev;
        const bool restore_from_staging = (q_per_core > 1 && !is_first_active_iter);
        ASSERT(!has_sliding_window || !restore_from_staging);
        // FIFO entry: this pass's own state sits at the front (the fixed cyclic pass order
        // guarantees it), so the first K chunk merges straight out of the FIFO and pops it.
        // The staging path builds the identical triple, so both share one redirect.
        const bool fifo_entry = use_l1_state_fifo && !is_first_active_iter;
        if (restore_from_staging || fifo_entry) {
            q_prev = {cb_sum_in, cb_max_in, cb_prev_out};
        }

        uint32_t KV_chunks_processed = 0;

        // Global K bounds of this Q chunk's rows, shared by all its packed K chunks.
        static_assert(
            !packed_kv_enabled || Sk_chunk_t <= ring_joint::kMaxPackedKVChunkTiles,
            "packed mask runs live on the TRISC stack; the host groups only chunks up to the cap");
        static_assert(ring_joint::kPackedKVInvalidRowTile == KV_PAD_ROTATION_INVALID_TILE);
        ring_joint::PackedKVRowBounds packed_row_bounds{};
        if (packed_sources) {
            const uint32_t mask_q_start = kv_pad_rotation_enabled ? q_chunk * Sq_chunk_t : q_start_tile;
            packed_row_bounds =
                ring_joint::packed_kv_row_bounds(Sq_chunk_t, logical_nt, packed_head_global_end, [&](uint32_t row) {
                    return q_global_tile_for_mask_row<kv_pad_rotation_enabled>(
                        row, mask_q_start, chunked.kv_pad_rotation);
                });
        }

        const uint32_t q_k_loop_count = has_sliding_window ? sliding_band.end : num_kv_chunks;
        for (uint32_t k_chunk = has_sliding_window ? sliding_band.begin : 0; k_chunk < q_k_loop_count; ++k_chunk) {
            const auto sliding_k_chunk = sliding_q_plan.k_chunk_at(k_chunk);
            // Circular cache: the mask's K origin is the plan's absolute K-chunk index (inverting local
            // cache rows is ambiguous once several chunk groups alias one local slab); unbounded caches
            // keep the inversion. Resolved once here, outside the [&] lambdas below: anything new they
            // capture changes the unbounded kernels' register allocation.
            uint32_t circular_k_origin_tile = 0;
            if constexpr (circular_kv_cache) {
                circular_k_origin_tile = sliding_q_plan.global_k_chunk_at(k_chunk) * Sk_chunk_t;
            }
            const uint32_t source_ring_id = has_sliding_window ? sliding_k_chunk.source_ring_id : ring_id;
            const uint32_t source_k_chunk = has_sliding_window ? sliding_k_chunk.source_k_chunk : k_chunk;
            const bool kv_chunk_is_joint = !packed_sources && !has_sliding_window && k_chunk >= num_local_k_chunks;
            if (!packed_sources && !has_sliding_window &&
                try_skip_oob_kv(source_ring_id, source_k_chunk, kv_chunk_is_joint)) {
                continue;
            }
            if (try_skip_causal_above_diag(source_k_chunk, causal_k_limit)) {
                continue;
            }
            KV_chunks_processed++;
            KV_chunks_processed_in_iter++;

            const bool is_first = is_first_kv_for_this_q && (KV_chunks_processed == 1);
            const bool is_last_k = (KV_chunks_processed == per_q_valid_kv);

            // Last K chunk of last ring_iter triggers per-row normalization
            const bool is_last_k_of_last_ring_iter = is_last_ring_iter && is_last_k;

            // Signal writer that last K-chunk is starting (for row-by-row DMA save/restore).
            if (is_last_k && q_per_core > 1) {
                CircularBuffer(cb_signal).reserve_back(1);
                sdpa_cb_push_back_out_of_line(cb_signal, 1);
            }

            // Determine if this K chunk needs masking (partial tile within a tile boundary)
            bool is_global_n_mask_chunk = false;
            bool is_local_n_mask_chunk = false;
            bool is_joint_n_mask_chunk = false;
            if constexpr (global_n_mask_enabled) {
                is_global_n_mask_chunk = ring_iter_needs_global_n_mask && source_k_chunk == global_n_mask_chunk_id;
            }
            if constexpr (local_n_mask_enabled) {
                is_local_n_mask_chunk = local_n_needs_masking && source_k_chunk == local_n_mask_chunk_id;
            }
            if constexpr (joint_n_mask_enabled) {
                is_joint_n_mask_chunk = ring_iter_needs_joint_n_mask && kv_chunk_is_joint &&
                                        (k_chunk - num_local_k_chunks) == joint_n_mask_chunk_id;
            }
            // Straddle chunk (rix > rid balanced-causal with k_chunk_size ∤ coarse_chunk_size):
            // only the early-half columns attend; late-half columns must be dropped. Narrow
            // active_Sk like local_n; no partial-tile stamp needed for tile-aligned straddle.
            bool is_straddle_mask_chunk = false;
            if constexpr (straddle_mask_enabled) {
                is_straddle_mask_chunk =
                    lw_mask.straddle_num_padded_tiles > 0 && source_k_chunk == lw_mask.straddle_mask_chunk_id;
            }

            bool apply_mask = kv_pad_rotation_enabled;
            if constexpr (global_n_mask_enabled) {
                apply_mask = apply_mask || is_global_n_mask_chunk;
            }
            if constexpr (joint_n_mask_enabled) {
                // Mirror the spatial local_n strategy for the joint chunk. The exp variant's LightweightMaskContext
                // keeps it runtime.
                if (lw_mask.joint_l_partial_col > 0) {
                    apply_mask = apply_mask || is_joint_n_mask_chunk;
                }
            }

            // Resolve lightweight mask params for partial tile masking
            uint32_t lw_partial_tile_idx = 0;
            if constexpr (lightweight_mask_enabled) {
                if (apply_mask) {
                    bool partial_tile_selected = false;
                    // Live column 0 = tile-aligned boundary this dispatch: no partial stamp.
                    if constexpr (global_n_mask_enabled) {
                        if (!packed_sources && is_global_n_mask_chunk) {
                            lw_partial_tile_idx =
                                lw_mask.global_n_partial_col > 0 ? lw_mask.global_n_partial_tile_idx : 0u;
                            partial_tile_selected = true;
                        }
                    }
                    if constexpr (joint_n_mask_enabled) {
                        if (!partial_tile_selected && is_joint_n_mask_chunk) {
                            lw_partial_tile_idx =
                                lw_mask.joint_l_partial_col > 0 ? lw_mask.joint_l_partial_tile_idx : 0u;
                        }
                    }
                }
            }

            // Tile-level matmul skip for global_n, local_n, or joint_n padding.
            // Runtime reduce narrows to active_Sk; matmul/sub_exp/V also need narrowing.
            // Also select pre-computed subblock width for this chunk's mask type.
            uint32_t active_Sk_param = packed_sources ? packed_kv.valid_tiles(k_chunk) : Sk_chunk_t;
            uint32_t chunk_sbw = packed_sources && active_Sk_param != Sk_chunk_t
                                     ? largest_factor_le(active_Sk_param, qkt_subblock_w)
                                     : full_sbw;
            bool narrowed_by_mask = packed_sources;
            if constexpr (global_n_mask_enabled) {
                if (!packed_sources && is_global_n_mask_chunk) {
                    active_Sk_param = Sk_chunk_t - lw_mask.global_n_padded_tiles;
                    chunk_sbw = global_n_sbw;
                    narrowed_by_mask = true;
                }
            }
            if constexpr (local_n_mask_enabled) {
                if (!narrowed_by_mask && is_local_n_mask_chunk) {
                    active_Sk_param = Sk_chunk_t - lw_mask.local_n_padded_tiles;
                    chunk_sbw = local_n_sbw;
                    narrowed_by_mask = true;
                }
            }
            if constexpr (joint_n_mask_enabled) {
                if (!narrowed_by_mask && is_joint_n_mask_chunk) {
                    active_Sk_param = Sk_chunk_t - lw_mask.joint_n_padded_tiles;
                    chunk_sbw = joint_n_sbw;
                    narrowed_by_mask = true;
                }
            }
            if constexpr (straddle_mask_enabled) {
                // A short balanced shard can end inside the same K chunk as its
                // causal half boundary. Padding does not supersede that tighter
                // boundary: intersect both masks rather than keeping the first one.
                const uint32_t straddle_active = Sk_chunk_t - lw_mask.straddle_num_padded_tiles;
                if (is_straddle_mask_chunk && straddle_active < active_Sk_param) {
                    active_Sk_param = straddle_active;
                    chunk_sbw = straddle_sbw;
                }
            }

            // Causal narrowing on the diagonal-crossing K-chunk: cols past the last Q-row's
            // diag tile are -inf for every row in the Q-chunk, so skip matmul/sub_exp/V there.
            // Composes with any prior padding narrowing via min().
            if constexpr (is_causal_sdpa && !kv_pad_rotation_enabled) {
                if constexpr (has_sliding_window) {
                    const uint32_t k_global_start =
                        circular_kv_cache ? circular_k_origin_tile
                                          : kv_global_tile_for_local<true, local_padded_Nt, chunk_size_t, kv_region_Nt>(
                                                source_ring_id, source_k_chunk * Sk_chunk_t);
                    const uint32_t q_global_end = q_start_tile + Sq_chunk_t;
                    if (k_global_start < q_global_end && q_global_end - k_global_start < active_Sk_param) {
                        active_Sk_param = q_global_end - k_global_start;
                        chunk_sbw = largest_factor_le(active_Sk_param, qkt_subblock_w);
                    }
                } else if (is_causal_iter && k_chunk == causal_k_limit - 1) {
                    const uint32_t causal_active = q_start_tile + Sq_chunk_t - k_chunk * Sk_chunk_t;
                    if (causal_active < active_Sk_param) {
                        active_Sk_param = causal_active;
                        chunk_sbw = largest_factor_le(active_Sk_param, qkt_subblock_w);
                    }
                }
            }

            // On last K-chunk of non-last ring iters (multi-Q), redirect output, sum, and max
            // to writer-staging CBs, eliminating post-loop copy_block calls.
            // Writer drains cb_out row-by-row during SALAD; cb_sum_out and cb_max_out bulk after.
            const bool save_to_staging = is_last_k && !is_last_ring_iter && q_per_core > 1;
            ASSERT(!has_sliding_window || !save_to_staging);
            // FIFO exit: sum/out redirect into the FIFO (pack-only, write-pointer relative). max
            // cannot — the step reads cur.max front-relative — so mirror it via step_save_max_cb.
            const bool fifo_exit = use_l1_state_fifo && is_last_k && !is_last_ring_iter;
            const uint32_t step_save_out_cb = save_to_staging ? cb_out : (fifo_exit ? cb_prev_out : INVALID_CB);
            const uint32_t step_save_max_cb = save_to_staging ? cb_max_out : (fifo_exit ? cb_max_in : INVALID_CB);
            if (save_to_staging) {
                q_cur.sum = cb_sum_out;
            } else if (fifo_exit) {
                q_cur.sum = cb_sum_in;
            }

            // K start tile fed to diag stamp must share Q's coord frame (local for is_causal, global for chunked).
            const uint32_t step_k_start_tile_local = [&]() {
                if constexpr (has_sliding_window) {
                    return kv_global_tile_for_local<true, local_padded_Nt, chunk_size_t, kv_region_Nt>(
                        source_ring_id, source_k_chunk * Sk_chunk_t);
                } else if constexpr (chunked_enabled) {
                    return kv_global_tile_for_local<true, local_padded_Nt, chunk_size_t, kv_region_Nt>(
                        ring_id, k_chunk * Sk_chunk_t);
                } else {
                    return k_chunk * Sk_chunk_t;
                }
            }();
            // Circular cache: the plan's absolute origin replaces the inversion above.
            const uint32_t step_k_start_tile = circular_kv_cache ? circular_k_origin_tile : step_k_start_tile_local;
            const bool step_apply_causal = [&]() {
                if constexpr (has_sliding_window) {
                    return q_start_tile < step_k_start_tile + Sk_chunk_t &&
                           q_start_tile + Sq_chunk_t > step_k_start_tile;
                }
                return is_causal_iter;
            }();

            // Chunked-prefill straddle. Background: each device's K cache holds the per-chunk
            // K region for every chunk back-to-back, kv_region_Nt tiles per region. When
            // k_chunk_size does not divide kv_region_Nt, a single K-chunk can begin in
            // one region (chunk j) and end one or more regions later. Because adjacent regions
            // map to *non-adjacent* global K positions (jumping by chunk_size_t between them),
            // the global K coord is no longer contiguous across the K-chunk's columns. We
            // signal this to the diag stamp via:
            //   - straddle_col: column index at which the jump happens (= tiles remaining in
            //     region j from this K-chunk's start)
            //   - straddle_jump: the global-K increment at that boundary
            //     (= chunk_size_t - kv_region_Nt, i.e. the gap between region j's end and
            //     region j+1's start in global K).
            // When straddle_col > 0 the stamp evaluates the diagonal per column instead of
            // per row, applying the jump at straddle_col and every kv_region_Nt tiles thereafter.
            uint32_t step_straddle_col = 0;
            uint32_t step_straddle_jump = 0;
            if constexpr (chunked_enabled) {
                if (kv_region_Nt > 0) {
                    const uint32_t local_start = source_k_chunk * Sk_chunk_t;
                    const uint32_t slab_end_local = (local_start / kv_region_Nt + 1) * kv_region_Nt;
                    if (local_start + Sk_chunk_t > slab_end_local) {
                        step_straddle_col = slab_end_local - local_start;
                        step_straddle_jump = chunk_size_t - kv_region_Nt;
                    }
                }
            }
            KVPadRotationContext step_kv_pad_rotation = chunked.kv_pad_rotation;
            step_kv_pad_rotation.k_local_start_tile = source_k_chunk * Sk_chunk_t;
            step_kv_pad_rotation.ring_id = source_ring_id;
            step_kv_pad_rotation.logical_tile_count = logical_nt;

            // Packed chunks mostly hold prefix slabs that every row of this Q chunk sees in full; the
            // plan narrows the live width and says whether the chunk needs stamping.
            ring_joint::PackedKVMaskRun packed_runs[packed_kv_enabled ? Sk_chunk_t : 1];
            ring_joint::PackedKVChunkMask packed_mask;
            if (packed_sources) {
                const ring_joint::PackedKVChunkPlan plan = ring_joint::packed_kv_chunk_plan(
                    packed_kv,
                    k_chunk,
                    packed_source_ids,
                    chunk_size_t,
                    packed_row_bounds,
                    qkt_subblock_w,
                    active_Sk_param,
                    packed_runs);
                if (plan.active_tiles < active_Sk_param) {
                    active_Sk_param = plan.active_tiles;
                    chunk_sbw = largest_factor_le(active_Sk_param, qkt_subblock_w);
                }
                packed_mask = plan.mask;
            }

            sdpa_inner_loop_step<
                false,  // profiling_enabled
                Sq_chunk_t,
                Sk_chunk_t,
                Skt,
                DHt,
                vDHt,
                scale_fp32,
                qkt_subblock_h,
                qkt_subblock_w,
                qktv_subblock_h,
                qktv_subblock_w,
                false,                              // use_padded_mask — ring uses ring mask instead
                true,                               // ring_mode
                is_causal_sdpa || chunked_enabled,  // chunked re-enables causal masking with absolute coords
                cb_q_in,
                cb_kt_in,
                cb_v_in,
                cb_qkt_im,
                cb_identity_scale_in,
                cb_exp_max_diff,
                cb_col_identity,
                cb_recip_scratch,
                cb_normalized_out,
                cb_mask_in,
                Sk_chunk_t,
                kv_pad_rotation_enabled,
                kv_region_Nt,
                chunk_size_t,
                local_padded_Nt,
                v_cb_physical_width_t,
                kt_inplace_v,
                sliding_window_size,
                use_attention_sink,
                cb_attention_sink,
                false,              // use_provided_mask
                use_l1_state_fifo,  // has_q_base_tiles: head-serial passes read Q at q_base_tiles
                circular_kv_cache,
                packed_kv_enabled>(
                q_prev,
                q_cur,
                is_last_k_of_last_ring_iter,
                is_first,
                apply_mask,
                lw_partial_tile_idx,
                active_Sk_param,
                can_reduce_trigger && (active_Sk_param == Sk_chunk_t),
                chunk_sbw,
                step_save_out_cb,
                step_save_max_cb,
                step_apply_causal,
                kv_pad_rotation_enabled ? q_chunk * Sq_chunk_t : q_start_tile,
                step_k_start_tile,
                lw_mask.neginf_tile_idx,
                has_sliding_window ? 1 : lw_mask.causal_diag_tile_idx,
                has_sliding_window ? 2 : 0,
                has_sliding_window ? 3 : 0,
                has_sliding_window ? 4 : 0,
                has_sliding_window,
                step_straddle_col,
                step_straddle_jump,
                step_kv_pad_rotation,
                q_base_tiles,
                packed_mask);

            // Post-iteration cleanup: pop previous values and swap aliases
            // prev.out and cb_exp_max_diff are already popped row-by-row inside salad_correct_row.
            if (!is_first) {
                sdpa_cb_pop_front_out_of_line(q_prev.max, Sq_chunk_t);
                sdpa_cb_pop_front_out_of_line(q_prev.sum, Sq_chunk_t);
            }

            if (is_last_k_of_last_ring_iter) {
                // Normalization consumed cur.sum and cur.out; pop cur.max.
                sdpa_cb_pop_front_out_of_line(q_cur.max, Sq_chunk_t);
            } else {
                std::swap(q_prev, q_cur);
                // After K0's swap, q_cur holds the staging/FIFO buffers. Reset to the scratch
                // accumulator CBs so the rest of the pass ping-pongs between scratch halves.
                if ((restore_from_staging || fifo_entry) && KV_chunks_processed == 1) {
                    q_cur = {original_prev.sum, original_prev.max, original_prev.out};
                }
            }
        }
        ASSERT(!has_sliding_window || KV_chunks_processed == per_q_valid_kv);

        // Pop Q — not popped inside step since ring_mode gates the early Q pop.
        // When q_per_core == 1, Q is identical across ring iterations so we keep it
        // fronted in the CB and only pop on the last iteration to avoid redundant DRAM re-reads.
        // Head-serial passes own the pop themselves: all resident chunks are popped together after
        // the last pass of the last ring iteration, so nothing is popped here.
        if constexpr (!use_l1_state_fifo) {
            if (q_per_core > 1 || is_last_ring_iter) {
                sdpa_cb_pop_front_out_of_line(cb_q_in, Sq_chunk_t * DHt);
            }
        }

        // Persist or save accumulators for next ring iteration.
        //
        // FIFO passes persist nothing. Only the scratch max (mirrored, not redirected) is left;
        // after the last swap it is q_prev.max — pop it.
        if constexpr (use_l1_state_fifo) {
            if (!is_last_ring_iter) {
                sdpa_cb_pop_front_out_of_line(q_prev.max, Sq_chunk_t);
            }
        } else if (q_per_core == 1) {
            // Single Q-chunk: persist in L1 (no DRAM round-trip)
            acc_state.prev = q_prev;
            acc_state.cur = q_cur;
            acc_state.last_call_k_chunks = KV_chunks_processed;
        } else if (!is_last_ring_iter) {
            // Multi Q-chunk: save raw accumulators to DRAM via writer CBs.
            // Out tiles already saved row-by-row via cb_out during last K-chunk SALAD.
            // Sum already in cb_sum_out (redirected via q_cur.sum on last K-chunk).
            // Max already in cb_max_out (dual-write from DST during reduce on last K-chunk).
            // Pop the accumulator CB for max — dual-write doesn't replace the alias,
            // so the accumulator CB still has tiles from the last K-chunk's reduce.
            // (Sum doesn't need this because planting replaced the alias entirely.)
            sdpa_cb_pop_front_out_of_line(q_prev.max, Sq_chunk_t);
        }
        // On last ring_iter: normalized output already in cb_out from normalize_row_streaming
    }

    // Padded multicast slots: the reader pushes every K (and V) entry it receives so this core's CB
    // advances with the injector's. Drain them in reader order with the same skips as a real Q chunk;
    // padded_slot_flat_q decodes the chunk each slot stands for, exactly as the reader does.
    if constexpr (!has_sliding_window) {
        for (uint32_t slot = 0; slot < padded_q_slots; ++slot) {
            const uint32_t owned = global_q_end - global_q_start;
            const uint32_t q_flat = padded_slot_flat_q<rotated_q_split_enabled>(
                rotated_slots, global_q_start, owned, owned + slot, use_zigzag_balancing ? 2 : 1);
            const uint32_t q_chunk = remap_q_index(q_flat, num_q_chunks, use_zigzag_balancing) % num_q_chunks;
            if constexpr (is_balanced_sdpa) {
                if (skip_first_half_q && q_chunk < num_q_chunks / 2) {
                    continue;
                }
            }
            const uint32_t padded_k_chunks = packed_sources ? packed_kv.chunk_count() : num_kv_chunks;
            for (uint32_t k_chunk = 0; k_chunk < padded_k_chunks; ++k_chunk) {
                if (!packed_sources && try_skip_oob_kv(ring_id, k_chunk, k_chunk >= num_local_k_chunks)) {
                    continue;
                }
                CircularBuffer(cb_kt_in).wait_front(DHt * Sk_chunk_t);
                sdpa_cb_pop_front_out_of_line(cb_kt_in, DHt * Sk_chunk_t);
                if constexpr (!kt_inplace_v) {
                    CircularBuffer(cb_v_in).wait_front(Sk_chunk_t * v_cb_physical_width_t);
                    sdpa_cb_pop_front_out_of_line(cb_v_in, Sk_chunk_t * v_cb_physical_width_t);
                }
                KV_chunks_processed_in_iter++;
            }
        }
    }

    // Dummy KV traffic exists only to align chained readers. Sliding mode uses independent
    // per-core reads so each core can prune a different K range and needs no phase padding.
    if constexpr (!has_sliding_window) {
        for (uint32_t dummy_chunk = 0;
             dummy_chunk <
             dummy_kv_chunks_for_phase_alignment<v_shares_k_buffer, kt_inplace_v>(KV_chunks_processed_in_iter);
             ++dummy_chunk) {
            CircularBuffer(cb_kt_in).wait_front(DHt * Sk_chunk_t);
            sdpa_cb_pop_front_out_of_line(cb_kt_in, DHt * Sk_chunk_t);
            // In-place latent-V never pushes a V entry, so there is nothing extra to drain.
            if constexpr (!kt_inplace_v) {
                CircularBuffer(cb_v_in).wait_front(Sk_chunk_t * v_cb_physical_width_t);
                sdpa_cb_pop_front_out_of_line(cb_v_in, Sk_chunk_t * v_cb_physical_width_t);
            }
        }
    }
}

}  // namespace gemma4_sliding
