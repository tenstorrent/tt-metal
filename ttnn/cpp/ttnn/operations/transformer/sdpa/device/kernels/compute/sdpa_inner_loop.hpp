// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_common.hpp"

enum {
    STANDARD = 0,
    JOINT = 1,
    RING = 2,
} SDPAType;

/**
 * For SDPA and windowed SDPA use q_start = 0, q_end = q_chunks_per_core.
 * For windowed SDPA use !BALANCED_Q_PARALLEL, !is_chunked, !is_causal.
 *
 * For SDPA and windowed SDPA use iter_k_chunk_start = 0.
 * For joint SDPA use iter_k_chunk_start = 0 and iter_k_chunk_end = k_num_chunks.
 * For joint SDPA set ring_id_equals_N_mask_ring_id = true and ring_id_equals_L_mask_ring_id = true.
 */
template <
    SDPAType sdpa_type,
    bool use_attention_sink,
    bool is_causal,
    bool use_provided_mask,
    bool use_padded_mask,
    bool use_joint_mask,
    bool is_chunked,
    bool scale_fp32>
void sdpa_inner_loop(
    const uint32_t Skt,
    const uint32_t DHt,
    const uint32_t vDHt,
    const uint32_t Sq_chunk_t,
    const uint32_t Sk_chunk_t,
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
    const uint32_t sliding_window_size,
    const uint32_t iter_q_start,
    const uint32_t iter_q_end,
    const uint32_t q_num_chunks,
    const uint32_t local_q_start,
    const uint32_t chunked_q_chunk_offset,
    const uint32_t iter_k_chunk_start,
    const uint32_t iter_k_chunk_end,
    const uint32_t q_chunk_tiles,
    const uint32_t k_chunk_tiles,
    const uint32_t qk_chunk_tiles,
    const uint32_t out_chunk_tiles,
    const uint32_t mask_chunk_0,
    const uint32_t mask_chunk_1,
    const bool ring_id_equals_N_mask_ring_id,
    const bool ring_id_equals_L_mask_ring_id,
    const uint32_t cb_q_in,
    const uint32_t cb_k_in,
    const uint32_t cb_v_in,
    const uint32_t cb_mask_in,
    const uint32_t cb_attention_sink,
    const uint32_t cb_identity_scale_in,
    const uint32_t cb_col_identity,
    const uint32_t cb_qk_im,
    const uint32_t cb_out_im_A,
    const uint32_t cb_out_im_B,
    const uint32_t cb_max_A,
    const uint32_t cb_max_B,
    const uint32_t cb_sum_A,
    const uint32_t cb_sum_B,
    const uint32_t cb_exp_max_diff,
    const uint32_t cb_out) {
    for (uint32_t q_iter = iter_q_start; q_iter < iter_q_end; ++q_chunk) {
        uint32_t q_low_idx;
        uint32_t q_high_idx;
        if constexpr (sdpa_type == STANDARD) {
            uint32_t q_chunk;
#if defined BALANCED_Q_PARALLEL
            uint32_t q_chunk_div_2 = q_end / 2;  // q_chunks_per_core / 2.
            if (q_iter < q_chunk_div_2) {        // bottom half
                q_chunk = local_q_start + q_iter;
            } else {
                uint32_t back_q_iter = q_iter - q_chunk_div_2;  // Back half should start at 0
                q_chunk = q_num_chunks - 1 - (local_q_start + back_q_iter);
            }
#else
            q_chunk = local_q_start + q_iter;
#endif
            // Get Q chunk
            if constexpr (is_chunked) {
                q_chunk = chunked_q_chunk_offset + q_chunk;
            }
            q_low_idx = q_chunk * Sq_chunk_t;  // This is the sequence index of the first tile of this chunk
            q_high_idx;
            if constexpr (is_causal) {
                q_high_idx = q_low_idx + Sq_chunk_t;
            } else {
                q_high_idx = Skt;
            }
        }

        // Set up ping pong buffers
        uint32_t alias_prev_sum = cb_sum_A;
        uint32_t alias_cur_sum = cb_sum_B;
        uint32_t alias_prev_max = cb_max_A;
        uint32_t alias_cur_max = cb_max_B;
        uint32_t alias_mm2_prev_out = cb_out_im_A;
        uint32_t alias_mm2_cur_out = cb_out_im_B;

        cb_wait_front(cb_q_in, q_chunk_tiles);

        int k_chunk_end;
        if (constexpr(sdpa_type == STANDARD)) {
            // loop while k_low < q_high => (k_chunk * Sk_chunk_t) < q_high_idx.
            k_chunk_end = (q_high_idx + Sk_chunk_t - 1) / Sk_chunk_t;
        } else {  // RING or JOINT.
            k_chunk_end = iter_k_chunk_end;
        }

        for (uint32_t k_chunk = iter_k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
            if (constexpr(sdpa_type == RING) && k_chunk >= global_logical_NK_chunks &&
                k_chunk < global_padded_NK_chunks) {
                // This is a KV chunk on spatial input beyond the chunk-padded length of the spatial input.
                // If k_chunk >= global_padded_NK_chunks, then this is a joint KV chunk.
                continue;
            }

            /* QK = Q_CHUNK @ K_CHUNK */
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

            // Finding the diagonal is harder now that q_chunk_size and k_chunk_size can differ
            // Q-range = [q_low, q_high)
            // K-range = [k_low, k_high)
            // does_overlap = not (q_low >= k_high or k_low >= q_high)
            // Due to loop bounds, we should never have k_low >= q_high. Can simplify this conditional check
            if constexpr (is_causal || sliding_window_size > 0) {
                const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
                const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;
                /* QK += MASK */
                if (!(q_low_idx >= k_high_idx) || sliding_window_size > 0) {
                    // If no sliding window - simple causal case - only apply along the diagonal
                    // Otherwise, apply mask for all chunks
                    reconfig_data_format(cb_qk_im, cb_mask_in);
                    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
                }
            } else if constexpr (use_provided_mask) {
                /* QK += MASK */
                reconfig_data_format(cb_qk_im, cb_mask_in);
                add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
            } else if constexpr (use_padded_mask) {
                // only uses mask on the last K chunk if it exists at all
                if (k_chunk == iter_k_chunk_end - 1) {
                    /* QK += MASK */
                    reconfig_data_format(cb_qk_im, cb_mask_in);
                    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
                }
            } else if constexpr (use_joint_mask) {
                if ((ring_id_equals_N_mask_ring_id && k_chunk == mask_chunk_0) ||
                    (ring_id_equals_L_mask_ring_id && k_chunk == mask_chunk_1)) {
                    /* QK += MASK */
                    reconfig_data_format(cb_qk_im, cb_mask_in);
                    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
                }
            }

            /**
             * reduce_c can perform both reduce_max and eltwise max with previous result.
             * if do_eltwise_max:
             *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
             * else:
             *  cur_max = max(qk, dim=-1)
             */
            reconfig_data_format(cb_qk_im, cb_identity_scale_in);
            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, Sk_chunk_t>(
                alias_cur_max, alias_prev_max, k_chunk > iter_k_chunk_start);
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
            sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, Sk_chunk_t, scale_fp32, true>(
                alias_cur_max, alias_cur_sum);

            cb_wait_front(cb_qk_im, qk_chunk_tiles);
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

            cb_pop_front(cb_qk_im, qk_chunk_tiles);
            reconfig_data_format(alias_prev_max, alias_cur_max);

            /* OUT_ACC += OUT_IM */
            if (k_chunk > iter_k_chunk_start) {
                /**
                 * cb_exp_max_diff = torch.exp((cb_prev_max - cb_cur_max) * scale)
                 * Scale is fused into exp again since max is the max of unscaled scores.
                 */

                sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);
                cb_pop_front(alias_prev_max, Sq_chunk_t);

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
                mul_block_bcast_cols<Sq_chunk_t, vDHt>(alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out, true);
            }

            // Swap CB handles to prepare for next iteration
            std::swap(alias_prev_sum, alias_cur_sum);
            std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
            std::swap(alias_prev_max, alias_cur_max);
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
            cb_pop_front(alias_prev_max, Sq_chunk_t);

            // 3. Rescale previous sum: prev_sum *= exp(prev_max - cur_max)
            mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);
            // 4. Compute exp((attention_sink - cur_max) * scale) and accumulate in cur_sum
            //    This adds the attention sink's contribution to the softmax denominator
            sub_exp_block_bcast_cols_inplace<cb_attention_sink, Sq_chunk_t, 1, scale_fp32, false>(
                alias_cur_max, alias_cur_sum);

            // 5. Add rescaled previous sum to current sum: cur_sum += prev_sum
            add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);

            // 6. Update running statistics for final normalization
            std::swap(alias_prev_sum, alias_cur_sum);
            std::swap(alias_prev_max, alias_cur_max);

            // 7. Rescale accumulated output: mm2_prev_out *= exp(prev_max - cur_max)
            //    Note: We do NOT compute attention_sink @ V, so output only has real token contributions
            //    But we need to rescale it due to the updated max
            mul_block_bcast_cols<Sq_chunk_t, vDHt>(alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out, false);
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
            mul_block_bcast_cols_inplace<Sq_chunk_t, DHt>(alias_mm2_prev_out, alias_prev_sum);
            if (ring_iter > 0) {
                // Update output according to previous and current LSE
                /**
                 * sig = torch.sigmoid(cur_lse - prev_lse)
                 * out = prev_out - sig * (prev_out - cur_out)
                 * lse = prev_lse - torch.logsigmoid(prev_lse - cur_lse)
                 */
                cb_wait_front(cb_lse_in, Sq_chunk_t);
                cb_wait_front(cb_prev_out, out_chunk_tiles);

                uint32_t alias_cur_lse = alias_prev_max;      // full
                uint32_t alias_sig = alias_cur_max;           // empty
                uint32_t alias_cur_out = alias_mm2_prev_out;  // full
                uint32_t alias_sub = alias_mm2_cur_out;       // empty

                // alias_sig = sigmoid(alias_cur_lse - cb_lse_in)
                sigmoid_sub(alias_cur_lse, cb_lse_in, alias_sig, Sq_chunk_t);

                // alias_sub = cb_prev_out - alias_cur_out
                sub_block(cb_prev_out, alias_cur_out, alias_sub, out_chunk_tiles);
                // alias_sub *= alias_sig
                mul_block_bcast_cols_inplace<Sq_chunk_t, DHt>(alias_sub, alias_sig);
                // cb_out = cb_prev_out - alias_sub
                sub_block(cb_prev_out, alias_sub, cb_out, out_chunk_tiles);
                cb_pop_front(cb_prev_out, out_chunk_tiles);
                cb_pop_front(alias_cur_out, out_chunk_tiles);
                cb_pop_front(alias_sub, out_chunk_tiles);

                // alias_sig = sigmoid(cb_lse_in - alias_cur_lse)
                // alias_cur_lse = log(alias_sig)
                // cb_lse_out = cb_lse_in - alias_cur_lse
                logsigmoid_sub(cb_lse_in, alias_cur_lse, alias_sig, Sq_chunk_t);
                sub_block(cb_lse_in, alias_sig, cb_lse_out, Sq_chunk_t);
                cb_pop_front(alias_sig, Sq_chunk_t);
                cb_pop_front(alias_cur_lse, Sq_chunk_t);
                cb_pop_front(cb_lse_in, Sq_chunk_t);
            } else {
                pack_reconfig_data_format(cb_out);
                copy_block(alias_mm2_prev_out, cb_out, out_chunk_tiles);

                copy_block(alias_prev_max, cb_lse_out, Sq_chunk_t);
            }
        } else {
            /* cb_cur_sum = 1.0 / cb_cur_sum */
            recip_block_inplace(alias_prev_sum, Sq_chunk_t);

            /* cb_out_accumulate_im *= cb_cur_sum */
            pack_reconfig_data_format(cb_out);
            mul_block_bcast_cols<Sq_chunk_t, vDHt>(alias_mm2_prev_out, alias_prev_sum, cb_out, false);

            // free up cb_prev_max after K chunks
            cb_pop_front(alias_prev_max, Sq_chunk_t);
        }

        cb_pop_front(cb_q_in, q_chunk_tiles);
    }

    if constexpr (use_attention_sink) {
        cb_pop_front(cb_attention_sink, Sq_chunk_t);
    }
}
