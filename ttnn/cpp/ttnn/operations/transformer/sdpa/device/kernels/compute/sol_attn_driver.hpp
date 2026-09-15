// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// sol_attn's chunk driver, copied from compute_common.hpp's sdpa_inner_loop / sdpa_standard.
//
// A copy rather than a shared edit: block-sparse chunk skipping and the pooled-summary residual
// change this loop, and compute_common.hpp is included by sdpa, vsa_sdpa, sparse_sdpa and both ring
// ops. Divergence from the original is the point; do not re-sync it wholesale.
//
// The only change so far is `chunk_keep_every` and the skip it gates. The skip sits in the same
// position as the existing causal skip and drains the same CBs, because the reader streams every
// chunk regardless of selection and `KV_chunks_processed_in_iter` -- incremented before either skip
// -- feeds the accumulator alias parity after the loop.
#pragma once

#include "compute_common.hpp"
#include "compute_streaming.hpp"

template <
    SDPAType sdpa_type,
    uint32_t cb_qk_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_attention_sink,
    uint32_t cb_scale_in,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t NH,
    uint32_t DHt,
    uint32_t vDHt,
    bool use_attention_sink,
    bool use_provided_mask,
    bool use_padded_mask,
    bool use_joint_mask,
    bool is_chunked,
    uint32_t scale_fp32,
    uint32_t sliding_window_size,
    bool lightweight_mask_enabled = false,
    bool chunked_enabled = false,
    uint32_t chunked_q_local_padded_Nt = 0,
    uint32_t chunked_chunk_size_t = 0,
    bool use_windowed_narrowing = false,
    uint32_t cb_windowed_k_range = 0,
    // Keep every Nth KV chunk, skip the rest. 0 or 1 = dense, and the skip compiles out.
    uint32_t chunk_keep_every = 0>
void sol_attn_inner_loop(
    const uint32_t Skt,
    const uint32_t qk_in0_block_w,
    const uint32_t qk_subblock_w,
    const uint32_t qk_subblock_h,
    const uint32_t qk_in0_num_subblocks,
    const uint32_t qk_in1_num_subblocks,
    const uint32_t qk_num_blocks,
    const uint32_t out_in0_block_w,
    const uint32_t out_subblock_w,
    const uint32_t out_subblock_h,
    const uint32_t out_in0_num_subblocks,
    const uint32_t out_in1_num_subblocks,
    const uint32_t out_num_blocks,
    const uint32_t iter_q_start,
    const uint32_t iter_q_end,
    const uint32_t q_num_chunks,
    const uint32_t local_q_start,
    const uint32_t chunked_q_chunk_offset,
    const uint32_t iter_k_chunk_start,
    const uint32_t iter_k_chunk_end,
    const uint32_t q_chunk_tiles,
    const uint32_t k_chunk_tiles,
    const uint32_t v_chunk_tiles,
    const uint32_t qk_chunk_tiles,
    const uint32_t out_chunk_tiles,
    const uint32_t mask_chunk_0,
    const uint32_t mask_chunk_1,
    const uint32_t ring_iter,
    const uint32_t ring_id,
    const uint32_t num_local_k_chunks,
    const uint32_t local_padded_Nt,
    const uint32_t logical_nt,
    const bool ring_iter_needs_global_n_mask,
    const bool ring_iter_needs_joint_n_mask,
    const bool local_n_needs_masking,
    const uint32_t global_n_mask_chunk_id,
    const uint32_t local_n_mask_chunk_id,
    const uint32_t joint_n_mask_chunk_id,
    const uint32_t cb_q_in,
    const uint32_t cb_k_in,
    const uint32_t cb_v_in,
    const uint32_t cb_mask_in,
    const uint32_t cb_col_identity,
    const uint32_t cb_out_im_A,
    const uint32_t cb_out_im_B,
    const uint32_t cb_max_A,
    const uint32_t cb_max_B,
    const uint32_t cb_sum_A,
    const uint32_t cb_sum_B,
    const uint32_t cb_exp_max_diff,
    const uint32_t cb_lse_in,
    const uint32_t cb_lse_out,
    const uint32_t cb_prev_out,
    const uint32_t cb_out,
    const LightweightMaskContext& lw_mask = {},
    const bool is_causal = false,
    const bool is_balanced = false,
    const bool use_zigzag_balancing = false,
    const bool is_last_ring_iter = true,
    const ChunkedContext& chunked = {}) {
    // Parameter-stable CB locals. Aliases (cb_sum_A/B, cb_max_A/B, cb_out_im_A/B) are
    // std::swap-mutated below, so they use inline CircularBuffer(alias).method() at the call
    // sites instead. cb_out is constructed conditionally near its consumer (cur.out target).
    CircularBuffer cb_q_in_obj(cb_q_in);
    CircularBuffer cb_k_in_obj(cb_k_in);
    CircularBuffer cb_v_in_obj(cb_v_in);
    CircularBuffer cb_qk_im_obj(cb_qk_im);
    CircularBuffer cb_attention_sink_obj(cb_attention_sink);
    CircularBuffer cb_lse_in_obj(cb_lse_in);
    CircularBuffer cb_prev_out_obj(cb_prev_out);
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    uint32_t KV_chunks_processed_in_iter = 0;
    const uint32_t q_per_core = iter_q_end - iter_q_start;

    for (uint32_t q_iter = iter_q_start; q_iter < iter_q_end; ++q_iter) {
        uint32_t q_start_tile = 0;    // First tile of Q chunk (tile units, both STANDARD and RING)
        uint32_t q_high_tile = 0;     // STANDARD: upper tile bound for K iteration
        uint32_t causal_k_limit = 0;  // RING: K-chunk index beyond which all K is above the diagonal
        if constexpr (sdpa_type == STANDARD) {
            const uint32_t linear_q_chunk = local_q_start + (q_iter - iter_q_start);
            // Mod is a no-op when the input is per-head ([0, q_num_chunks)) and extracts the
            // per-head q_chunk when it's a flat global index (global Q scheduling spans heads).
            uint32_t q_chunk = remap_q_index(linear_q_chunk, q_num_chunks, use_zigzag_balancing) % q_num_chunks;
            // Get Q chunk
            if constexpr (is_chunked) {
                q_chunk = chunked_q_chunk_offset + q_chunk;
            }
            q_start_tile = q_chunk * Sq_chunk_t;
            if (is_causal) {
                // Clamp to total K-tile extent. Mirrors reader_interleaved's clamp; without
                // both, the reader and compute disagree on K-chunk count when Q-chunk extends
                // past total K (Sq_chunk_t > Skt) → CB deadlock.
                const uint32_t q_high_unclamped = q_start_tile + Sq_chunk_t;
                q_high_tile = q_high_unclamped < Skt ? q_high_unclamped : Skt;
            } else {
                q_high_tile = Skt;
            }
        } else if (sdpa_type == RING) {
            uint32_t q_chunk = remap_q_index(q_iter, q_num_chunks, use_zigzag_balancing) % q_num_chunks;

            if constexpr (chunked_enabled) {
                // Absolute Q tile row. Diag stamp masks K past Q's range; logical_n skip handles K past the cache.
                q_start_tile =
                    chunked.q_start_idx_t + chunked.ring_index * chunked_q_local_padded_Nt + q_chunk * Sq_chunk_t;
            } else if (is_causal) {
                q_start_tile = q_chunk * Sq_chunk_t;
                causal_k_limit = (q_start_tile + Sq_chunk_t + Sk_chunk_t - 1) / Sk_chunk_t;
            }
            if (is_balanced && (q_chunk < q_num_chunks / 2)) {
                continue;
            }
        }  // If ring attention

        // Set up ping pong buffers
        uint32_t alias_prev_sum = cb_sum_A;
        uint32_t alias_cur_sum = cb_sum_B;
        uint32_t alias_prev_max = cb_max_A;
        uint32_t alias_cur_max = cb_max_B;
        uint32_t alias_mm2_prev_out = cb_out_im_A;
        uint32_t alias_mm2_cur_out = cb_out_im_B;

        uint32_t k_chunk_end;
        if constexpr (sdpa_type == STANDARD) {
            // loop while k_low < q_high => (k_chunk * Sk_chunk_t) < q_high_tile.
            k_chunk_end = (q_high_tile + Sk_chunk_t - 1) / Sk_chunk_t;
        } else {  // RING or JOINT.
            k_chunk_end = iter_k_chunk_end;
        }

        // Windowed K-range narrowing: this Q chunk's [k_lo, k_hi) comes from the reader's ctrl CB —
        // read via the UNPACK mailbox so all three TRISCs agree — and overrides BOTH bounds. The
        // reader streams exactly this many K/V chunks and the writer produces exactly this many mask
        // chunks; any disagreement deadlocks the CBs.
        uint32_t k_chunk_start = iter_k_chunk_start;
        if constexpr (use_windowed_narrowing) {
            CircularBuffer cb_k_range_obj(cb_windowed_k_range);
            cb_k_range_obj.wait_front(1);
            k_chunk_start = ckernel::read_tile_value(cb_windowed_k_range, 0, 0);
            k_chunk_end = ckernel::read_tile_value(cb_windowed_k_range, 0, 1);
            cb_k_range_obj.pop_front(1);
        }

        uint32_t processed_k_chunks = 0;

        for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
            uint32_t kv_global_start_tile = 0;  // RING only: abs K-tile index of this k_chunk's start
            if constexpr (sdpa_type == RING) {
                const bool kv_chunk_is_joint = k_chunk >= num_local_k_chunks;
                // Global index into the padded KV tensor. Chunked: non-monotonic local→abs map.
                if constexpr (chunked_enabled) {
                    kv_global_start_tile =
                        kv_global_tile_for_local<true, 0, chunked_chunk_size_t, chunked_q_local_padded_Nt>(
                            ring_id, k_chunk * Sk_chunk_t);
                } else {
                    kv_global_start_tile = local_padded_Nt * ring_id + k_chunk * Sk_chunk_t;
                }
                if (!kv_chunk_is_joint && (kv_global_start_tile >= logical_nt)) {
                    // This is a KV chunk on spatial input beyond the logical N, and not joint KV. Skip it.
                    continue;
                }
            }

            KV_chunks_processed_in_iter++;

            // Chunked-prefill: never take this skip (local-frame causal_k_limit doesn't apply —
            // the diag stamp uses absolute coords every k_chunk instead).
            if (sdpa_type == RING && !chunked_enabled && k_chunk >= causal_k_limit && is_causal) {
                cb_k_in_obj.wait_front(k_chunk_tiles);
                cb_v_in_obj.wait_front(v_chunk_tiles);
                cb_k_in_obj.pop_front(k_chunk_tiles);
                cb_v_in_obj.pop_front(v_chunk_tiles);

                continue;
            }

            // Block-sparse skip. Chunk 0 is always kept: the normalization after this loop is
            // unconditional, so a q chunk that selected nothing would divide by a zero accumulator.
            // No drain: the reader applies the same stride and never pushes a skipped chunk.
            if constexpr (chunk_keep_every > 1) {
                if (k_chunk != 0 && (k_chunk % chunk_keep_every) != 0) {
                    continue;
                }
            }

            /**
             * QK = Q_CHUNK @ K_CHUNK
             *
             * matmul_blocks internally waits on both inputs
             */
            reconfig_data_format(cb_k_in, cb_q_in);
            pack_reconfig_data_format(cb_qk_im);
            matmul_blocks(
                cb_q_in,
                cb_k_in,
                cb_qk_im,
                Sq_chunk_t,
                Sk_chunk_t,
                DHt,
                qk_num_blocks,
                qk_in0_num_subblocks,
                qk_in1_num_subblocks,
                qk_in0_block_w,
                qk_subblock_h,
                qk_subblock_w,
                true /*transpose*/);

            /**
             * Note
             * Typically, scores is multiplied by a scalar here. We employed an optimization
             * where we fuse the scaling into exp both in exp(x - max) and exp(prev_max - cur_max).
             * This gives us scaling for free on the performance-critical exp(x - max) computation.
             */

            bool apply_mask = false;
            bool needs_padding_mask =
                (sdpa_type == RING) &&
                ((ring_iter_needs_global_n_mask && k_chunk == global_n_mask_chunk_id) ||
                 (local_n_needs_masking && k_chunk == local_n_mask_chunk_id) ||
                 (ring_iter_needs_joint_n_mask && (k_chunk - num_local_k_chunks) == joint_n_mask_chunk_id) ||
                 (lw_mask.straddle_num_padded_tiles > 0 && k_chunk == lw_mask.straddle_mask_chunk_id));
            if (sdpa_type == RING && !is_causal) {
                apply_mask = needs_padding_mask;
            } else if (is_causal || sliding_window_size > 0) {
                // Chunked-prefill (RING): use abs K so the causal-overlap test matches the diag stamp's frame.
                uint32_t k_low_idx;
                uint32_t k_high_idx;
                if constexpr (sdpa_type == RING && chunked_enabled) {
                    k_low_idx = kv_global_start_tile;
                    k_high_idx = k_low_idx + Sk_chunk_t;
                    // Straddle: chunk crosses a per-chunk slab boundary in the local cache. The
                    // tail tiles land in the next slab, so the actual high global K is shifted by
                    // chunk_size_t - q_local_padded_Nt.
                    if (chunked_q_local_padded_Nt > 0) {
                        const uint32_t local_start = k_chunk * Sk_chunk_t;
                        const uint32_t slab_end_local =
                            (local_start / chunked_q_local_padded_Nt + 1) * chunked_q_local_padded_Nt;
                        if (local_start + Sk_chunk_t > slab_end_local) {
                            k_high_idx += (chunked_chunk_size_t - chunked_q_local_padded_Nt);
                        }
                    }
                } else {
                    k_low_idx = k_chunk * Sk_chunk_t;
                    k_high_idx = k_low_idx + Sk_chunk_t;
                }
                // Apply mask if causal overlap, sliding window, or this K chunk has padding
                apply_mask = (q_start_tile < k_high_idx) || (sliding_window_size > 0) || needs_padding_mask;
            } else if constexpr (use_provided_mask) {
                apply_mask = true;
            } else if constexpr (use_padded_mask) {
                // Apply mask only on the last K chunk
                apply_mask = (k_chunk == iter_k_chunk_end - 1);
            } else if constexpr (use_joint_mask) {
                // Apply mask for specific chunk combinations
                apply_mask = (k_chunk == mask_chunk_0) || (k_chunk == mask_chunk_1);
            }

            if (apply_mask) {
                /* QK += MASK */
                reconfig_data_format(cb_qk_im, cb_mask_in);
                if constexpr (lightweight_mask_enabled) {
                    // Re-enter reserved state on cb_qk_im so the lightweight mask can be stamped in-place.
                    // matmul_blocks above already pushed the QK tiles, so tiles_received has been bumped;
                    // without the pop+push cycle below, reduce_c's wait-front would return immediately
                    // and unpack could start before the mask stamps land in L1.
                    // Safe because pop-front only moves rd_ptr (L1 data is untouched), and on a
                    // single-buffered CB of size qk_chunk_tiles the re-reserved wr_ptr wraps back to the
                    // same physical region holding the QK scores.
                    // Warning: this won't work if cb_qk_im is double-buffered -- the stamps would land
                    // in the other buffer, leaving the QK scores unmasked.
                    cb_qk_im_obj.wait_front(Sk_chunk_t * Sq_chunk_t);
                    cb_qk_im_obj.pop_front(Sk_chunk_t * Sq_chunk_t);
                    cb_qk_im_obj.reserve_back(Sk_chunk_t * Sq_chunk_t);
                    // Chunked-prefill: feed abs K (matches abs q_start_tile so diag stamp lines up).
                    uint32_t k_start_tile_for_mask;
                    uint32_t lw_straddle_col = 0;
                    uint32_t lw_straddle_jump = 0;
                    if constexpr (sdpa_type == RING && chunked_enabled) {
                        k_start_tile_for_mask = kv_global_start_tile;
                        // Chunked-prefill straddle: a K-chunk can begin in one per-chunk K
                        // region and end in the next when k_chunk_size does not divide
                        // q_local_padded_Nt. Global K is non-contiguous across the K-chunk in
                        // that case (jumps by chunk_size_t - q_local_padded_Nt between regions),
                        // so we signal the column boundary (straddle_col) and the jump
                        // (straddle_jump) to the diag stamp — see same comment block in
                        // compute_streaming.hpp:sol_attn_inner_loop_step for the full picture.
                        if (chunked_q_local_padded_Nt > 0) {
                            const uint32_t local_start = k_chunk * Sk_chunk_t;
                            const uint32_t slab_end_local =
                                (local_start / chunked_q_local_padded_Nt + 1) * chunked_q_local_padded_Nt;
                            if (local_start + Sk_chunk_t > slab_end_local) {
                                lw_straddle_col = slab_end_local - local_start;
                                lw_straddle_jump = chunked_chunk_size_t - chunked_q_local_padded_Nt;
                            }
                        }
                    } else {
                        k_start_tile_for_mask = k_chunk * Sk_chunk_t;
                    }
                    lw_mask.template apply<dst_size>(
                        cb_mask_in,
                        cb_qk_im,
                        Sk_chunk_t,
                        Sq_chunk_t,
                        k_chunk,
                        num_local_k_chunks,
                        ring_iter_needs_global_n_mask,
                        ring_iter_needs_joint_n_mask,
                        local_n_needs_masking,
                        global_n_mask_chunk_id,
                        local_n_mask_chunk_id,
                        joint_n_mask_chunk_id,
                        q_start_tile,
                        k_start_tile_for_mask,
                        lw_straddle_col,
                        lw_straddle_jump);
                    cb_qk_im_obj.push_back(Sk_chunk_t * Sq_chunk_t);
                } else {
                    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
                }
            }

            /**
             * reduce_c can perform both reduce_max and eltwise max with previous result.
             * if do_eltwise_max:
             *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
             * else:
             *  cur_max = max(qk, dim=-1)
             *
             * Use the reduce_c overload with cols as a runtime arg which uses standard
             * reduce_tile + binary_max_tile. The overload with cols as a template arg
             * is bf16-only but cb_qk_im could be fp32.
             */
            reconfig_data_format(cb_qk_im, cb_identity_scale_in);
            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t>(
                alias_cur_max, alias_prev_max, Sk_chunk_t, processed_k_chunks > 0);

            /**
             * sub_exp fuses a few operations.
             * In-place it performs `QK = exp((QK - cur_max) * scale)`
             *
             * It also partially performs reduce_sum on the output using L1 accumulation.
             * `cur_sum = sum_tiles(exp((QK - cur_max) * scale), dim=-1)`
             *
             * Partial reduce_sum is used to push the final row_reduction within a tile
             * outside of the loop over K chunks.
             */
            sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, scale_fp32, true>(
                alias_cur_max, alias_cur_sum, Sk_chunk_t);

            // Reconfigure unpackers: srcA (context 0) = cb_v_in, srcB (context 1) = cb_qk_im (operands are swapped in
            // matmul)
            reconfig_data_format(cb_v_in, cb_qk_im);
            pack_reconfig_data_format(alias_mm2_cur_out);

            /* OUT_IM = QK @ V_CHUNK */
            matmul_blocks(
                cb_qk_im,
                cb_v_in,
                alias_mm2_cur_out,
                Sq_chunk_t,
                vDHt,
                Sk_chunk_t,
                out_num_blocks,
                out_in0_num_subblocks,
                out_in1_num_subblocks,
                out_in0_block_w,
                out_subblock_h,
                out_subblock_w,
                false /*transpose*/);

            cb_qk_im_obj.pop_front(qk_chunk_tiles);
            reconfig_data_format(alias_prev_max, alias_cur_max);

            /* OUT_ACC += OUT_IM */
            if (processed_k_chunks > 0) {
                /**
                 * cb_exp_max_diff = torch.exp((cb_prev_max - cb_cur_max) * scale)
                 * Scale is fused into exp again since max is the max of unscaled scores.
                 */
                sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);
                CircularBuffer(alias_prev_max).pop_front(Sq_chunk_t);

                /**
                 * cb_prev_sum *= cb_exp_max_diff
                 * This is a bcast_cols since max_diff is a column vector and prev_sum is a partial
                 * reduction, containing the sum of tiles in dim=-1 of QK.
                 */
                mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);

                /* cb_cur_sum += cb_prev_sum */
                add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);

                /**
                 * alias_mm2_cur_out += alias_mm2_prev_out * cb_exp_max_diff
                 * This uses L1 accumulation to accumulate onto mm2_cur_out.
                 */
                mul_block_bcast_cols<Sq_chunk_t, vDHt, false, true>(
                    alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out);
            }

            // Swap CB handles to prepare for next iteration
            std::swap(alias_prev_sum, alias_cur_sum);
            std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
            std::swap(alias_prev_max, alias_cur_max);

            processed_k_chunks++;
        }

        /**
         * Performs final row-reduction on the partial sum.
         */
        matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);

        /**
         * Process attention sink as a virtual K chunk.
         * The attention sink provides additional logits that are included in the softmax
         * denominator but don't contribute to the output (no S @ V computation).
         * This effectively allows some attention probability to be "absorbed" by the sink,
         * reducing attention weights on actual tokens.
         *
         * Shape of attention_sink: [Sq_chunk_t, 1] tiles
         * Each head has one sink logit value that is broadcast to all query positions in the chunk.
         * The reader kernel replicates the per-head value across all Sq_chunk_t positions.
         */
        if constexpr (use_attention_sink) {
            // Treat attention_sink as scores (already scaled)
            // Shape: [Sq_chunk_t, 1] tiles - same per-head sink value broadcast to all query positions

            // 1. Update running max: cur_max = max(prev_max, attention_sink)
            //    This compares the previous max with the sink logit
            reconfig_data_format(cb_attention_sink, cb_identity_scale_in);

            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_attention_sink, cb_identity_scale_in, Sq_chunk_t, 1>(
                alias_cur_max, alias_prev_max, true);

            // 2. Compute exp((prev_max - cur_max) * scale) to rescale previous statistics
            sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);
            CircularBuffer(alias_prev_max).pop_front(Sq_chunk_t);

            // 3. Rescale previous sum: prev_sum *= exp(prev_max - cur_max)
            mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);
            // 4. Compute exp((attention_sink - cur_max) * scale) and accumulate in cur_sum
            //    This adds the attention sink's contribution to the softmax denominator
            sub_exp_block_bcast_cols_inplace<cb_attention_sink, Sq_chunk_t, scale_fp32, false>(
                alias_cur_max, alias_cur_sum, 1);

            // 5. Add rescaled previous sum to current sum: cur_sum += prev_sum
            add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);

            // 6. Update running statistics for final normalization
            std::swap(alias_prev_sum, alias_cur_sum);
            std::swap(alias_prev_max, alias_cur_max);

            // 7. Rescale accumulated output: mm2_prev_out *= exp(prev_max - cur_max)
            //    Note: We do NOT compute attention_sink @ V, so output only has real token contributions
            //    But we need to rescale it due to the updated max
            mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(
                alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out);
            std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
        }

        if constexpr (sdpa_type == RING) {
            log_block(alias_prev_sum, alias_cur_max, Sq_chunk_t);

            // Scale prev_max by scale_fp32
            mul_block_bcast_scalar_inplace<cb_scale_in, Sq_chunk_t>(alias_prev_max);
            add_block_inplace(alias_prev_max, alias_cur_max, Sq_chunk_t);

            /* cb_cur_sum = 1.0 / cb_cur_sum */
            recip_block_inplace(alias_prev_sum, Sq_chunk_t);
            /* cb_out_accumulate_im *= cb_cur_sum */
            mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(alias_mm2_prev_out, alias_prev_sum);

            if (ring_iter > 0) {
                // Update output according to previous and current LSE
                /**
                 * sig = torch.sigmoid(cur_lse - prev_lse)
                 * out = prev_out - sig * (prev_out - cur_out)
                 * lse = prev_lse - torch.logsigmoid(prev_lse - cur_lse)
                 */
                cb_lse_in_obj.wait_front(Sq_chunk_t);
                cb_prev_out_obj.wait_front(out_chunk_tiles);

                uint32_t alias_cur_lse = alias_prev_max;      // full
                uint32_t alias_sig = alias_cur_max;           // empty
                uint32_t alias_cur_out = alias_mm2_prev_out;  // full
                uint32_t alias_sub = alias_mm2_cur_out;       // empty

                // alias_sig = sigmoid(alias_cur_lse - cb_lse_in)
                sigmoid_sub(alias_cur_lse, cb_lse_in, alias_sig, Sq_chunk_t);

                // alias_sub = cb_prev_out - alias_cur_out
                reconfig_data_format(cb_prev_out, alias_cur_out);
                sub_block(cb_prev_out, alias_cur_out, alias_sub, out_chunk_tiles);
                // alias_sub *= alias_sig
                reconfig_data_format(alias_sub, alias_sig);
                mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(alias_sub, alias_sig);
                // cb_out = cb_prev_out - alias_sub
                reconfig_data_format(cb_prev_out, alias_sub);
                pack_reconfig_data_format(cb_out);
                sub_block(cb_prev_out, alias_sub, cb_out, out_chunk_tiles);
                cb_prev_out_obj.pop_front(out_chunk_tiles);
                CircularBuffer(alias_cur_out).pop_front(out_chunk_tiles);
                CircularBuffer(alias_sub).pop_front(out_chunk_tiles);

                // alias_sig = sigmoid(cb_lse_in - alias_cur_lse)
                // alias_cur_lse = log(alias_sig)
                // cb_lse_out = cb_lse_in - alias_cur_lse
                pack_reconfig_data_format(alias_sig);
                reconfig_data_format(cb_lse_in, alias_cur_lse);
                logsigmoid_sub(cb_lse_in, alias_cur_lse, alias_sig, Sq_chunk_t);
                sub_block(cb_lse_in, alias_sig, cb_lse_out, Sq_chunk_t);
                CircularBuffer(alias_sig).pop_front(Sq_chunk_t);
                CircularBuffer(alias_cur_lse).pop_front(Sq_chunk_t);
                cb_lse_in_obj.pop_front(Sq_chunk_t);
            } else {
                pack_reconfig_data_format(cb_out);
                copy_block(alias_mm2_prev_out, cb_out, out_chunk_tiles);

                pack_reconfig_data_format(cb_lse_out);
                copy_block(alias_prev_max, cb_lse_out, Sq_chunk_t);
            }
        } else {
            /* cb_cur_sum = 1.0 / cb_cur_sum */
            recip_block_inplace(alias_prev_sum, Sq_chunk_t);

            /* cb_out_accumulate_im *= cb_cur_sum */
            pack_reconfig_data_format(cb_out);
            mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(alias_mm2_prev_out, alias_prev_sum, cb_out);

            // free up cb_prev_max after K chunks
            CircularBuffer(alias_prev_max).pop_front(Sq_chunk_t);
        }

        // When q_per_core == 1, Q is identical across ring iterations so we keep it
        // fronted in the CB and only pop on the last iteration to avoid redundant DRAM re-reads.
        if (q_per_core > 1 || is_last_ring_iter) {
            cb_q_in_obj.pop_front(q_chunk_tiles);
        }

        // Under global Q scheduling the reader pushes one cb_attention_sink slot per Q iter,
        // so we must drain matching slots inside this loop. Pre-PR the push/pop pair was
        // 1:1 outside the loop; the new cadence is 1:1 per iter.
        if constexpr (use_attention_sink) {
            cb_attention_sink_obj.pop_front(Sq_chunk_t);
        }
    }

    if constexpr (sdpa_type == RING) {
        if (KV_chunks_processed_in_iter % 2 == 0) {
            cb_k_in_obj.wait_front(k_chunk_tiles);
            cb_v_in_obj.wait_front(v_chunk_tiles);
            cb_k_in_obj.pop_front(k_chunk_tiles);
            cb_v_in_obj.pop_front(v_chunk_tiles);
        }
    }
}

template <
    uint32_t cb_qk_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_attention_sink,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    uint32_t vDHt,
    bool use_attention_sink,
    bool is_causal,
    bool use_provided_mask,
    bool use_padded_mask,
    bool is_chunked,
    uint32_t scale_fp32,
    uint32_t sliding_window_size,
    bool lightweight_mask_enabled = false,
    bool use_windowed_narrowing = false,
    uint32_t cb_windowed_k_range = 0,
    uint32_t chunk_keep_every = 0,
    // Accepted for call-site uniformity with sol_attn_standard_v2; the legacy path doesn't act on it
    // (use_streaming_compute routes every tested block_mask shape to v2 -- see sol_attn.cpp).
    uint32_t cb_block_mask_row = 0>
void sol_attn_standard(
    const uint32_t Skt,
    const uint32_t qk_in0_block_w,
    const uint32_t qk_subblock_w,
    const uint32_t qk_subblock_h,
    const uint32_t qk_in0_num_subblocks,
    const uint32_t qk_in1_num_subblocks,
    const uint32_t qk_num_blocks,
    const uint32_t out_in0_block_w,
    const uint32_t out_subblock_w,
    const uint32_t out_subblock_h,
    const uint32_t out_in0_num_subblocks,
    const uint32_t out_in1_num_subblocks,
    const uint32_t out_num_blocks,
    const uint32_t iter_q_start,
    const uint32_t iter_q_end,
    const uint32_t q_num_chunks,
    const uint32_t local_q_start,
    const uint32_t chunked_q_chunk_offset,
    const uint32_t k_num_chunks,
    const uint32_t q_chunk_tiles,
    const uint32_t k_chunk_tiles,
    const uint32_t v_chunk_tiles,
    const uint32_t qk_chunk_tiles,
    const uint32_t out_chunk_tiles,
    const uint32_t cb_q_in,
    const uint32_t cb_k_in,
    const uint32_t cb_v_in,
    const uint32_t cb_mask_in,
    const uint32_t cb_col_identity,
    const uint32_t cb_out_im_A,
    const uint32_t cb_out_im_B,
    const uint32_t cb_max_A,
    const uint32_t cb_max_B,
    const uint32_t cb_sum_A,
    const uint32_t cb_sum_B,
    const uint32_t cb_exp_max_diff,
    const uint32_t cb_out,
    const LightweightMaskContext& lw_mask = {},
    const bool use_zigzag_balancing = false) {
    sol_attn_inner_loop<
        STANDARD,
        cb_qk_im,
        cb_identity_scale_in,
        cb_attention_sink,
        0,  // cb_scale_in (not used)
        Sq_chunk_t,
        Sk_chunk_t,
        0,  // NH (not used)
        DHt,
        vDHt,
        use_attention_sink,
        use_provided_mask,
        use_padded_mask,
        false,  // use_joint_mask (not used)
        is_chunked,
        scale_fp32,
        sliding_window_size,
        lightweight_mask_enabled,
        false,  // chunked_enabled (not used)
        0,      // chunked_q_local_padded_Nt (not used)
        0,      // chunked_chunk_size_t (not used)
        use_windowed_narrowing,
        cb_windowed_k_range,
        chunk_keep_every>(
        Skt,
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
        iter_q_start,
        iter_q_end,
        q_num_chunks,
        local_q_start,
        chunked_q_chunk_offset,
        0,             // iter_k_chunk_start
        k_num_chunks,  // iter_k_chunk_end
        q_chunk_tiles,
        k_chunk_tiles,
        v_chunk_tiles,
        qk_chunk_tiles,
        out_chunk_tiles,
        0,      // mask_chunk_0 (not used)
        0,      // mask_chunk_1 (not used)
        0,      // ring_iter (not used)
        0,      // ring_id (not used)
        0,      // num_local_k_chunks (not used)
        0,      // local_padded_Nt (not used)
        0,      // logical_nt (not used)
        false,  // ring_iter_needs_global_n_mask (not used)
        false,  // ring_iter_needs_joint_n_mask (not used)
        false,  // local_n_needs_masking (not used)
        0,      // global_n_mask_chunk_id (not used)
        0,      // local_n_mask_chunk_id (not used)
        0,      // joint_n_mask_chunk_id (not used)
        cb_q_in,
        cb_k_in,
        cb_v_in,
        cb_mask_in,
        cb_col_identity,
        cb_out_im_A,
        cb_out_im_B,
        cb_max_A,
        cb_max_B,
        cb_sum_A,
        cb_sum_B,
        cb_exp_max_diff,
        0,  // cb_lse_in (not used)
        0,  // cb_lse_out (not used)
        0,  // cb_prev_out (not used)
        cb_out,
        lw_mask,
        is_causal,
        false,  // is_balanced (not used)
        use_zigzag_balancing);
}

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
    bool use_padded_mask,
    uint32_t cb_q_in,
    uint32_t cb_kt_in,
    uint32_t cb_v_in,
    uint32_t cb_qkt_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_exp_max_diff,
    uint32_t cb_col_identity,
    uint32_t cb_recip_scratch,
    uint32_t cb_normalized_out,
    uint32_t cb_mask_in,
    uint32_t sliding_window_size = 0,
    bool is_causal_sdpa = false,
    bool use_attention_sink = false,
    uint32_t cb_attention_sink = INVALID_CB,
    bool use_provided_mask = false,
    bool use_windowed_narrowing = false,
    uint32_t cb_windowed_k_range = INVALID_CB,
    // Keep every Nth KV chunk, skip the rest. 0 or 1 = dense, and every branch below compiles out.
    uint32_t chunk_keep_every = 0,
    // Per-Q-chunk keep/skip row, one uint32_t (0 or nonzero) per k_chunk, written by the reader into
    // this CB before the K-chunk loop below. INVALID_CB when block_mask is absent, matching the
    // windowed ctrl CB's convention above.
    uint32_t cb_block_mask_row = INVALID_CB>
void sol_attn_standard_v2(
    const uint32_t q_chunks_per_core,
    const uint32_t k_num_chunks,
    const uint32_t cb_out_im_A,
    const uint32_t cb_out_im_B,
    const uint32_t cb_max_A,
    const uint32_t cb_max_B,
    const uint32_t cb_sum_A,
    const uint32_t cb_sum_B,
    const uint32_t local_q_start = 0,
    const uint32_t chunked_q_chunk_offset = 0,
    const LightweightMaskContext& lw_mask = {},
    const uint32_t q_num_chunks = 0,
    const bool use_zigzag_balancing = false) {
    init_sdpa_streaming_semaphores();

    // use_padded_mask + is_causal_sdpa is handled at the host level (mutually exclusive).
    static_assert(
        !(use_padded_mask && is_causal_sdpa), "use_padded_mask and is_causal_sdpa are mutually exclusive in v2");
    // K-loop bound geometry shared with the reader (see sliding_window_geometry.hpp).
    using window_geom = SlidingWindowLoopGeometry<sliding_window_size, is_causal_sdpa, TILE_HEIGHT>;
    constexpr bool has_sliding_window = window_geom::has_sliding_window;
    constexpr uint32_t left_window_tiles = window_geom::left_window_tiles;
    constexpr uint32_t right_window_tiles = window_geom::right_window_tiles;

    // v1 dense-mask scope: a user-provided mask supplies all masking itself (causal/sliding/padding
    // are baked into the tensor by the caller, and the reader neginf-fills padded positions), so it
    // is never combined with the structured stamp paths.
    static_assert(
        !(use_provided_mask && (is_causal_sdpa || has_sliding_window)),
        "use_provided_mask is mutually exclusive with causal/sliding stamping in v2");

    // Neginf tile is permanently fronted by the writer — wait once before any K-chunk loop.
    // Skipped for a user-provided mask: the writer generates no palette; the reader streams the
    // dense mask (with padded positions neginf-filled) per chunk instead.
    constexpr uint32_t padded_k_tiles_inner = (Sk_chunk_t - (Skt % Sk_chunk_t)) % Sk_chunk_t;
    if constexpr (use_padded_mask && padded_k_tiles_inner > 0 && !use_provided_mask) {
        CircularBuffer(cb_mask_in).wait_front(1);
    }

    constexpr uint32_t last_chunk_Sk = Sk_chunk_t - padded_k_tiles_inner;

    for (uint32_t q = 0; q < q_chunks_per_core; q++) {
        AccumulatorHalf prev = {cb_sum_A, cb_max_A, cb_out_im_A};
        AccumulatorHalf cur = {cb_sum_B, cb_max_B, cb_out_im_B};

        // reduce_trigger enables early reduce start via semaphore signaling from packer to unpacker.
        // The unpack MOP is split in half (block_ct_dim / 2), so active_Sk must be even,
        // and we need >1 subblock so the semaphore fires before the reduce's second half.
        constexpr bool can_reduce_trigger = reduce_trigger_supported && (Sk_chunk_t % qkt_subblock_w == 0) &&
                                            (Sk_chunk_t / qkt_subblock_w > 1) && (Sk_chunk_t % 2 == 0);

        // Pre-compute subblock width: compile-time for full chunks, hoisted for padded last chunk.
        constexpr uint32_t full_sbw = qkt_subblock_w;
        constexpr uint32_t padded_sbw = (last_chunk_Sk < Sk_chunk_t && last_chunk_Sk % qkt_subblock_w != 0)
                                            ? largest_factor_le(last_chunk_Sk, qkt_subblock_w)
                                            : qkt_subblock_w;

        // With largest_factor_le, padded chunks also have evenly-dividing subblocks,
        // so reduce_trigger can be enabled when the same constraints hold for last_chunk_Sk.
        constexpr bool can_reduce_trigger_padded = reduce_trigger_supported && (padded_k_tiles_inner > 0) &&
                                                   (last_chunk_Sk % padded_sbw == 0) &&
                                                   (last_chunk_Sk / padded_sbw > 1) && (last_chunk_Sk % 2 == 0);

        // Optional zigzag Q-chunk remap plus per-Q K-chunk bounds. Causal uses the
        // diagonal upper bound; sliding-window adds a lower bound and, for non-causal
        // centered windows, an upper bound around the Q chunk.
        uint32_t q_chunk_local = local_q_start + q;
        uint32_t q_start_tile = 0;
        uint32_t k_loop_start = 0;
        uint32_t k_loop_end = k_num_chunks;
        if constexpr (is_causal_sdpa || has_sliding_window) {
            // Reader and writer apply the same remap; compute must agree or causal
            // masks and output positions desync. The mod is a no-op when the input is per-head
            // ([0, q_num_chunks)) and extracts the per-head q_chunk when it's a flat global index
            // (global Q scheduling iterates across batches and heads).
            q_chunk_local = remap_q_index(q_chunk_local, q_num_chunks, use_zigzag_balancing) % q_num_chunks;
            // q_chunk_global is the absolute Q chunk index (used for the diagonal);
            // chunked-prefill shifts this via chunked_q_chunk_offset.
            const uint32_t q_chunk_global = q_chunk_local + chunked_q_chunk_offset;
            q_start_tile = q_chunk_global * Sq_chunk_t;
            if constexpr (is_causal_sdpa) {
                const uint32_t limit = (q_start_tile + Sq_chunk_t + Sk_chunk_t - 1) / Sk_chunk_t;
                k_loop_end = limit < k_num_chunks ? limit : k_num_chunks;
            }
            if constexpr (has_sliding_window) {
                if (q_start_tile > left_window_tiles) {
                    k_loop_start = (q_start_tile - left_window_tiles) / Sk_chunk_t;
                }
                if constexpr (!is_causal_sdpa) {
                    const uint32_t limit =
                        (q_start_tile + Sq_chunk_t + right_window_tiles + Sk_chunk_t - 1) / Sk_chunk_t;
                    k_loop_end = limit < k_num_chunks ? limit : k_num_chunks;
                }
            }
        }
        // Windowed K-range narrowing: this Q chunk's [k_lo, k_hi) comes from the reader's ctrl CB —
        // read via the UNPACK mailbox so all three TRISCs agree. The reader streams exactly this many
        // K/V chunks and the writer produces exactly this many mask chunks; disagreement deadlocks.
        if constexpr (use_windowed_narrowing) {
            CircularBuffer cb_k_range_obj(cb_windowed_k_range);
            cb_k_range_obj.wait_front(1);
            k_loop_start = ckernel::read_tile_value(cb_windowed_k_range, 0, 0);
            k_loop_end = ckernel::read_tile_value(cb_windowed_k_range, 0, 1);
            cb_k_range_obj.pop_front(1);
        }

        // Block mask: the reader wrote this Q chunk's keep/skip row (one uint32_t per k_chunk) before
        // pushing any K/V, so it is always available by the time compute would wait on it. Held for
        // the whole k_chunk loop (many reads, not a fixed pair like the windowed range above) and
        // popped once after the loop -- see the matching pop_front below.
        constexpr bool use_block_mask = (cb_block_mask_row != INVALID_CB);
        CircularBuffer cb_block_mask_row_obj(cb_block_mask_row);
        if constexpr (use_block_mask) {
            cb_block_mask_row_obj.wait_front(1);
        }

        auto call_step = [&](auto profiling_tag,
                             bool is_last,
                             bool is_first,
                             uint32_t active_Sk,
                             bool reduce_trigger,
                             uint32_t sbw,
                             bool apply_causal,
                             uint32_t k_start_tile,
                             bool apply_mask,
                             uint32_t lw_partial_tile_idx,
                             bool apply_sliding_window) {
            sdpa_inner_loop_step<
                decltype(profiling_tag)::value,
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
                use_padded_mask,
                false,  // ring_mode
                is_causal_sdpa,
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
                false,
                0,
                0,
                0,
                vDHt,
                false,
                sliding_window_size,
                use_attention_sink,
                cb_attention_sink,
                use_provided_mask>(
                prev,
                cur,
                is_last,
                is_first,
                apply_mask,
                lw_partial_tile_idx,
                active_Sk,
                reduce_trigger,
                sbw,
                INVALID_CB,  // save_out_cb
                INVALID_CB,  // save_max_cb
                apply_causal,
                q_start_tile,
                k_start_tile,
                lw_mask.neginf_tile_idx,
                lw_mask.primary_diag_tile_idx,
                lw_mask.sliding_leading_prev_tile_idx,
                lw_mask.sliding_leading_tile_idx,
                lw_mask.sliding_trailing_next_tile_idx,
                apply_sliding_window);
        };

        // Under a chunk stride, is_first / is_last must track the first and last KEPT chunks:
        // is_last drives the normalization inside sdpa_inner_loop_step, so pointing it at a chunk
        // the stride skips would leave the accumulator unnormalized. Chunk 0 is always kept, so a
        // q chunk can never select nothing.
        auto stride_keeps = [](uint32_t k) {
            if constexpr (chunk_keep_every > 1) {
                return k == 0 || (k % chunk_keep_every) == 0;
            }
            return true;
        };
        // block_mask is mutually exclusive with chunk_keep_every (validated at the op level), so at
        // most one of these two predicates is ever the real one; the other compiles out to `true`.
        auto mask_keeps = [&](uint32_t k) {
            if constexpr (use_block_mask) {
                return k == 0 || ckernel::read_tile_value(cb_block_mask_row, 0, k) != 0;
            }
            return true;
        };
        auto chunk_kept = [&](uint32_t k) { return stride_keeps(k) && mask_keeps(k); };
        uint32_t first_kept = k_loop_start;
        uint32_t last_kept = k_loop_end - 1;
        if constexpr (chunk_keep_every > 1 || use_block_mask) {
            bool seen = false;
            for (uint32_t k = k_loop_start; k < k_loop_end; k++) {
                if (chunk_kept(k)) {
                    if (!seen) {
                        first_kept = k;
                        seen = true;
                    }
                    last_kept = k;
                }
            }
        }

        for (uint32_t k_chunk = k_loop_start; k_chunk < k_loop_end; k_chunk++) {
            if constexpr (chunk_keep_every > 1 || use_block_mask) {
                // The reader applies the same predicate and never pushes a skipped chunk, so there is
                // nothing to drain -- a drain here would deadlock waiting on tiles that never come.
                if (!chunk_kept(k_chunk)) {
                    continue;
                }
            }
            bool is_first = (k_chunk == first_kept);
            bool is_last = (k_chunk == last_kept);

            // Padded path is non-causal only (use_padded_mask && is_causal_sdpa rejected by static_assert).
            // With sliding-window loop narrowing, the loop's last chunk may not be the tensor's final K chunk.
            // A user-provided mask processes the full Sk_chunk_t (padded cols are neginf in the dense
            // mask), so it never takes the partial-tile narrowing.
            const bool is_padded =
                !is_causal_sdpa && !use_provided_mask && (k_chunk == k_num_chunks - 1) && (padded_k_tiles_inner > 0);
            uint32_t chunk_active_Sk = is_padded ? last_chunk_Sk : Sk_chunk_t;
            bool chunk_reduce_trigger = is_padded ? can_reduce_trigger_padded : can_reduce_trigger;
            uint32_t chunk_sbw = is_padded ? padded_sbw : full_sbw;

            // Last-chunk narrowing: causal and non-causal partial-tile K are mutually exclusive
            // (static_assert at function entry), but both shrink chunk_active_Sk on is_last to
            // skip cols past the diag (causal) or past Sk's last partial tile (non-causal partial).
            //
            // Mask side:
            // - Causal: per-row diagonal/trailing-neginf stamp via apply_lightweight_mask_streaming
            //   (no-stamp / partial-stamp / full-neginf decided per row by diag_col).
            // - Non-causal partial: trailing fully-padded tiles → neginf via num_padded; partial
            //   boundary tile → vertical-bar mask tile via apply_mask + lw_partial_tile_idx.
            bool apply_partial_mask = false;
            bool apply_sliding_mask = false;
            bool apply_causal_mask = is_causal_sdpa;
            uint32_t target_active_Sk = chunk_active_Sk;
            if constexpr (use_padded_mask && !is_causal_sdpa && !use_provided_mask) {
                // Stamp the partial-tile vertical mask whenever the tensor's last K chunk carries a
                // partial tile (Sk % TILE != 0), independent of fully padded tiles. When there are no
                // fully padded tiles, global_n_padded_tiles is 0, so target_active_Sk remains
                // Sk_chunk_t and only the vertical mask stamp is added.
                if ((k_chunk == k_num_chunks - 1) && lw_mask.global_n_partial_col > 0) {
                    target_active_Sk = Sk_chunk_t - lw_mask.global_n_padded_tiles;
                    apply_partial_mask = true;
                }
            }
            if constexpr (has_sliding_window) {
                const uint32_t k_start_tile = k_chunk * Sk_chunk_t;
                const uint32_t trailing_range_end =
                    is_causal_sdpa ? (q_start_tile + Sq_chunk_t) : (q_start_tile + Sq_chunk_t + right_window_tiles);
                apply_sliding_mask = true;
                if constexpr (is_causal_sdpa) {
                    const uint32_t k_end_tile = k_start_tile + Sk_chunk_t;
                    apply_causal_mask = (q_start_tile < k_end_tile) && ((q_start_tile + Sq_chunk_t) > k_start_tile);
                }
                if (is_last && trailing_range_end > k_start_tile) {
                    const uint32_t window_active_Sk = trailing_range_end - k_start_tile;
                    target_active_Sk = target_active_Sk < window_active_Sk ? target_active_Sk : window_active_Sk;
                }
            } else if constexpr (is_causal_sdpa) {
                if (is_last) {
                    target_active_Sk = q_start_tile + Sq_chunk_t - k_chunk * Sk_chunk_t;
                }
            }
            if (target_active_Sk < chunk_active_Sk) {
                chunk_active_Sk = target_active_Sk;
                chunk_sbw = largest_factor_le(chunk_active_Sk, qkt_subblock_w);
                // reduce_trigger relies on active_Sk == Sk_chunk_t for the unpack MOP split.
                chunk_reduce_trigger = false;
            }

            call_step(
                std::false_type{},
                is_last,
                is_first,
                chunk_active_Sk,
                chunk_reduce_trigger,
                chunk_sbw,
                apply_causal_mask,
                k_chunk * Sk_chunk_t,
                apply_partial_mask,
                apply_partial_mask ? lw_mask.global_n_partial_tile_idx : 0u,
                apply_sliding_mask);

            // Post-iteration cleanup
            // prev.out and cb_exp_max_diff are already popped row-by-row inside salad_correct_row.
            if (!is_first) {
                sdpa_cb_pop_front_out_of_line(prev.max, Sq_chunk_t);
                sdpa_cb_pop_front_out_of_line(prev.sum, Sq_chunk_t);
            }

            if (is_last) {
                sdpa_cb_pop_front_out_of_line(cur.max, Sq_chunk_t);
            } else {
                std::swap(prev, cur);
            }
        }
        if constexpr (use_block_mask) {
            cb_block_mask_row_obj.pop_front(1);
        }
        // Q already popped inside sdpa_inner_loop_step after Phase 1 of the last K chunk.
    }
}
