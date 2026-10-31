// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Gemma4 sliding ring SDPA's copy of sdpa_ring_v2 (compute_streaming.hpp). The rest of the streaming
// primitives stay shared; only the ring loop is forked so Gemma4-only changes do not touch the shared op.
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"

namespace gemma4_sliding {

/**
 * Streaming Ring SDPA (v2), sliding-window copy: one synthetic ring pass per Q chunk over the local
 * slab and the predecessor halo, in the order of the sliding work plan. Accumulates raw
 * (un-normalized) softmax state across the K chunks and normalizes on the last K chunk of the last
 * ring iteration. Under a K split each band leaves its unnormalized state in the caller's
 * accumulators for the reducer instead.
 *
 * @tparam Sq_chunk_t       Q chunk size in tiles
 * @tparam Sk_chunk_t       K chunk size in tiles
 * @tparam Skt              Not used for ring (pass 0)
 * @tparam DHt              Head dimension in tiles
 * @tparam vDHt             V head dimension in tiles (== DHt for ring)
 * @tparam scale_fp32       Attention scale factor as raw uint32_t bits
 * @tparam qkt_subblock_h   QK matmul subblock height
 * @tparam cb_normalized_out CB for normalized output rows (written by normalize_row_streaming)
 * @tparam cb_signal        CB the writer waits on before draining a Q chunk's output (multi Q-chunk)
 * @tparam local_padded_Nt   Per-device KV padded sequence length in tiles (N_local / TILE_H)
 * @tparam q_local_padded_Nt Per-device Q padded sequence length in tiles.
 * @tparam kv_region_Nt      Per-source block-cyclic KV region; Q slab divided by the stripe split.
 * @tparam chunk_size_t      Per-chunk Q/K extent in tiles (chunked-only)
 *
 * @param global_q_start     First global Q chunk index for this core
 * @param global_q_end       One-past-last global Q chunk index for this core
 * @param num_local_k_chunks Number of K chunks from the local (non-joint) sequence
 * @param logical_nt         Actual (unpadded) global sequence length in tiles
 * @param acc_state          Persistent accumulator state (prev/cur CB halves for ping-pong)
 * @param is_last_ring_iter  True on the final ring iteration — triggers normalization
 * @param q_per_core         Number of Q chunks per core (1 keeps Q and the accumulators in L1)
 * @param lw_mask            Lightweight mask context for partial-tile padding
 * @param chunked            Chunked-prefill runtime state (q_start_idx_t, ring_index)
 * @param is_first_active_iter Set on the first ring iter that does work
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
    uint32_t kv_region_Nt = q_local_padded_Nt,
    // Split-KV: GROUPED_KV_SOURCE_COUNT > 1. Keeps the per-source path free of packed branches.
    bool packed_kv_enabled = false,
    typename MaskCtx = LightweightMaskContext>
void sdpa_ring_v2(
    const uint32_t global_q_start,
    const uint32_t global_q_end,
    const uint32_t num_q_chunks,
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
    const bool use_zigzag_balancing = false,
    const ChunkedContext& chunked = {},
    const bool is_first_active_iter = true,
    // Sliding K split only: this core's band of the work plan (ring_joint_ksplit.hpp).
    const uint32_t sliding_split_idx = 0,
    const uint32_t sliding_split_count = 1,
    const uint32_t* packed_source_ids = nullptr,
    const uint32_t packed_source_count = 0,
    // Rows of newest slabs this pass appends (packed_kv_pass_rows).
    const uint32_t packed_newest_rows = 0) {
    init_sdpa_streaming_semaphores();

    constexpr bool has_sliding_window = sliding_window_size > 0;
    // The shared op's dense ring loop (per-source ring iterations, DRAM-staged accumulators, balanced
    // skips, multicast padding) is not carried in this copy.
    static_assert(has_sliding_window && chunked_enabled, "This ring loop serves chunked sliding prefill only");
    static_assert(!is_balanced_sdpa, "Sliding attention is unbalanced");
    // Sliding folds every source into one pass per Q, so a Q chunk never stages state in DRAM between
    // passes: the only non-final pass is a K-split band's, and K split runs one Q chunk per core.
    ASSERT(is_last_ring_iter || q_per_core == 1);
    ASSERT(is_first_active_iter);

    // reduce_trigger enables early reduce start via semaphore signaling from packer to unpacker.
    // All conditions are compile-time except the active_Sk == Sk_chunk_t guard (padded chunks).
    constexpr bool can_reduce_trigger = reduce_trigger_supported && (Sk_chunk_t % qkt_subblock_w == 0) &&
                                        (Sk_chunk_t / qkt_subblock_w > 1) && (Sk_chunk_t % 2 == 0);

    // Subblock width for the non-padded (common) case — compile-time constant.
    constexpr uint32_t full_sbw = qkt_subblock_w;

    // Pre-compute subblock widths for each mask type (hoisted out of per-chunk loop).
    uint32_t global_n_sbw = full_sbw;
    uint32_t local_n_sbw = full_sbw;
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

    for (uint32_t q = global_q_start; q < global_q_end; q++) {
        // Compute Q chunk index (with optional zigzag remapping for causal balancing).
        // num_q_chunks is total per-head chunks (local + joint), matching the divisor the
        // writer/reader use to flatten (batch, head, q_chunk) — see ring_joint_sdpa.cpp.
        uint32_t q_chunk = remap_q_index(q, num_q_chunks, use_zigzag_balancing) % num_q_chunks;

        // Absolute tile row of this Q chunk's first row.
        const uint32_t q_start_tile =
            logical_nt - ring_size * q_local_padded_Nt + chunked.ring_index * q_local_padded_Nt + q_chunk * Sq_chunk_t;

        constexpr bool circular_kv_cache = circular_kv_slab_count > 1;
        const ttnn::operations::transformer::sdpa::ring_joint::ChunkedQMapping q_mapping{
            chunked.kv_pad_rotation.q_pre_wrap_start_tile,
            chunked.kv_pad_rotation.q_pre_wrap_tile_count,
            chunked.kv_pad_rotation.q_post_wrap_start_tile,
            chunked.kv_pad_rotation.q_valid_tile_count};
        const ttnn::operations::transformer::sdpa::ring_joint::SlidingQWorkPlan<sliding_q_plan_max_source_ranges>
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
        // K chunks this core processes for this Q chunk: its band of the work plan.
        const auto sliding_band = ttnn::operations::transformer::sdpa::ring_joint::sliding_ksplit_range(
            sliding_q_plan.total_k_chunk_count, sliding_split_idx, sliding_split_count);
        const uint32_t per_q_valid_kv = sliding_band.end - sliding_band.begin;
        // Use persistent accumulator state from caller.
        AccumulatorHalf q_prev = acc_state.prev, q_cur = acc_state.cur;

        const bool is_first_kv_for_this_q = is_first_active_iter;

        uint32_t KV_chunks_processed = 0;

        for (uint32_t k_chunk = sliding_band.begin; k_chunk < sliding_band.end; ++k_chunk) {
            const auto sliding_k_chunk = sliding_q_plan.k_chunk_at(k_chunk);
            const uint32_t source_ring_id = sliding_k_chunk.source_ring_id;
            const uint32_t source_k_chunk = sliding_k_chunk.source_k_chunk;
            KV_chunks_processed++;

            const bool is_first = is_first_kv_for_this_q && (KV_chunks_processed == 1);
            const bool is_last_k = (KV_chunks_processed == per_q_valid_kv);

            // Last K chunk of last ring_iter triggers per-row normalization
            const bool is_last_k_of_last_ring_iter = is_last_ring_iter && is_last_k;

            // Signal writer that last K-chunk is starting (it drains the output row by row).
            if (is_last_k && q_per_core > 1) {
                CircularBuffer(cb_signal).reserve_back(1);
                sdpa_cb_push_back_out_of_line(cb_signal, 1);
            }

            // Determine if this K chunk needs masking (partial tile within a tile boundary)
            bool is_global_n_mask_chunk = false;
            bool is_local_n_mask_chunk = false;
            if constexpr (global_n_mask_enabled) {
                is_global_n_mask_chunk = ring_iter_needs_global_n_mask && source_k_chunk == global_n_mask_chunk_id;
            }
            if constexpr (local_n_mask_enabled) {
                is_local_n_mask_chunk = local_n_needs_masking && source_k_chunk == local_n_mask_chunk_id;
            }

            bool apply_mask = kv_pad_rotation_enabled;
            if constexpr (global_n_mask_enabled) {
                apply_mask = apply_mask || is_global_n_mask_chunk;
            }

            // Resolve lightweight mask params for partial tile masking
            uint32_t lw_partial_tile_idx = 0;
            if constexpr (lightweight_mask_enabled) {
                if (apply_mask) {
                    // Live column 0 = tile-aligned boundary this dispatch: no partial stamp.
                    if constexpr (global_n_mask_enabled) {
                        if (is_global_n_mask_chunk) {
                            lw_partial_tile_idx =
                                lw_mask.global_n_partial_col > 0 ? lw_mask.global_n_partial_tile_idx : 0u;
                        }
                    }
                }
            }

            // Tile-level matmul skip for global_n, local_n, or joint_n padding.
            // Runtime reduce narrows to active_Sk; matmul/sub_exp/V also need narrowing.
            // Also select pre-computed subblock width for this chunk's mask type.
            uint32_t active_Sk_param = Sk_chunk_t;
            uint32_t chunk_sbw = full_sbw;
            bool narrowed_by_mask = false;
            if constexpr (global_n_mask_enabled) {
                if (is_global_n_mask_chunk) {
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

            // K start tile fed to the diag stamp, in Q's global coord frame.
            const uint32_t step_k_start_tile =
                kv_global_tile_for_local<true, local_padded_Nt, chunk_size_t, kv_region_Nt>(
                    source_ring_id, source_k_chunk * Sk_chunk_t);
            const bool step_apply_causal =
                q_start_tile < step_k_start_tile + Sk_chunk_t && q_start_tile + Sq_chunk_t > step_k_start_tile;

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

            // Per-source traversal: the K chunk takes the ordinary (contiguous) mask path.
            ring_joint::PackedKVChunkMask packed_mask;

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
                false,  // use_provided_mask
                false,  // has_q_base_tiles: only head-serial passes offset Q
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
                INVALID_CB,  // save_out_cb: no DRAM staging between sliding passes
                INVALID_CB,  // save_max_cb
                step_apply_causal,
                kv_pad_rotation_enabled ? q_chunk * Sq_chunk_t : q_start_tile,
                step_k_start_tile,
                lw_mask.neginf_tile_idx,
                1,  // sliding edge-mask tiles follow neginf in cb_mask_in
                2,
                3,
                4,
                has_sliding_window,
                step_straddle_col,
                step_straddle_jump,
                step_kv_pad_rotation,
                0,  // q_base_tiles
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
            }
        }
        ASSERT(KV_chunks_processed == per_q_valid_kv);

        // Pop Q — not popped inside step since ring_mode gates the early Q pop.
        // When q_per_core == 1, Q is identical across ring iterations so we keep it
        // fronted in the CB and only pop on the last iteration to avoid redundant DRAM re-reads.
        if (q_per_core > 1 || is_last_ring_iter) {
            sdpa_cb_pop_front_out_of_line(cb_q_in, Sq_chunk_t * DHt);
        }

        // Single Q-chunk: persist the accumulators in L1 (a K-split band hands them to the reducer).
        // On the last ring_iter the normalized output is already in cb_out from normalize_row_streaming.
        if (q_per_core == 1) {
            acc_state.prev = q_prev;
            acc_state.cur = q_cur;
            acc_state.last_call_k_chunks = KV_chunks_processed;
        }
    }
}

}  // namespace gemma4_sliding
