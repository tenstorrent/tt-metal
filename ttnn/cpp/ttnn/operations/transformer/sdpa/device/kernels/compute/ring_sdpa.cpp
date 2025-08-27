// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "compute_kernel_api.h"
#include "compute_common.hpp"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    DPRINT << "RING_COMPUTE: Starting MAIN function" << ENDL();
    // Compile-time arguments (same as regular SDPA plus ring parameters)
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Skt = get_compile_time_arg_val(3);
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t vDHt = get_compile_time_arg_val(5);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t local_q_num_chunks = get_compile_time_arg_val(7);  // Fixed: was q_num_chunks
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(9);

    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(10);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(11);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(12);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(13);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(14);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(15);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(16);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(17);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(18);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(19);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(20);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(21);

    constexpr uint32_t num_cores = get_compile_time_arg_val(22);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(23);

    // Ring distribution parameters
    constexpr uint32_t ring_size = get_compile_time_arg_val(24);
    constexpr uint32_t ring_id = get_compile_time_arg_val(25);
    constexpr uint32_t first_chunk_id = get_compile_time_arg_val(26);
    constexpr uint32_t second_chunk_id = get_compile_time_arg_val(27);

    // Runtime arguments
    const uint32_t core_id = get_arg_val<uint32_t>(0);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(1);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(2);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(3);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(4);
    const uint32_t local_q_start = get_arg_val<uint32_t>(5);
    const uint32_t local_q_end = get_arg_val<uint32_t>(6);

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;  // Should be 2 for ring

    DPRINT << "RING_COMPUTE: Ring params - ring_size=" << (uint32_t)ring_size << " ring_id=" << (uint32_t)ring_id
           << ENDL();
    DPRINT << "RING_COMPUTE: q_chunks_per_core=" << (uint32_t)q_chunks_per_core << ENDL();

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    // Circular buffer indices
    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;

    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_out_im_A = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_im_B = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_A = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_B = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_A = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_B = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    mm_init(cb_q_in, cb_k_in, cb_out);

    DPRINT << "RING_COMPUTE: Starting main processing loops" << ENDL();

    // Main processing loop: iterate over batches, heads, and Q chunks
    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                uint32_t q_chunk;

                // Ring Q chunk selection with mutual exclusion
#if defined RING_Q_DISTRIBUTION
                // Ring distribution provides global load balancing across devices
                uint32_t global_q_chunk = (q_iter == 0) ? first_chunk_id : second_chunk_id;
                q_chunk = global_q_chunk;

#elif defined BALANCED_Q_PARALLEL
                // Per-core load balancing (disabled when using ring distribution)
                uint32_t q_chunk_div_2 = q_chunks_per_core / 2;
                if (q_iter < q_chunk_div_2) {
                    q_chunk = local_q_start + q_iter;
                } else {
                    uint32_t back_q_iter = q_iter - q_chunk_div_2;
                    q_chunk = local_q_num_chunks - 1 - (local_q_start + back_q_iter);  // Fixed: was q_num_chunks
                }
                DPRINT << "RING_COMPUTE: Using BALANCED_Q_PARALLEL - q_chunk=" << (uint32_t)q_chunk << ENDL();
#else
                q_chunk = local_q_start + q_iter;  // Simple consecutive processing
                DPRINT << "RING_COMPUTE: Using DEFAULT - q_chunk=" << (uint32_t)q_chunk << ENDL();
#endif

                // Calculate Q range for this chunk (always causal for ring distribution)
                uint32_t q_low_idx = q_chunk * Sq_chunk_t;
                uint32_t q_high_idx = q_low_idx + Sq_chunk_t;

                DPRINT << "RING_COMPUTE: Q range calc - q_chunk=" << (uint32_t)q_chunk
                       << " Sq_chunk_t=" << (uint32_t)Sq_chunk_t << " q_low_idx=" << (uint32_t)q_low_idx
                       << " q_high_idx=" << (uint32_t)q_high_idx << ENDL();

                // Set up ping pong buffers
                uint32_t alias_prev_sum = cb_sum_A;
                uint32_t alias_cur_sum = cb_sum_B;
                uint32_t alias_prev_max = cb_max_A;
                uint32_t alias_cur_max = cb_max_B;
                uint32_t alias_mm2_prev_out = cb_out_im_A;
                uint32_t alias_mm2_cur_out = cb_out_im_B;

                cb_wait_front(cb_q_in, q_chunk_tiles);

                // Process K chunks for this Q chunk with early termination
                DPRINT << "RING_COMPUTE: Starting K chunk loop for q_chunk=" << (uint32_t)q_chunk
                       << " q_low_idx=" << (uint32_t)q_low_idx << " q_high_idx=" << (uint32_t)q_high_idx << ENDL();
                for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
                    const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
                    const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;

                    // Early computation skipping for causal constraint
                    // Ring distribution is always causal, so no need for is_causal conditional
                    bool should_compute = !(q_low_idx >= k_high_idx);

                    if (should_compute) {
                        DPRINT << "MATMUL k_chunk=" << (uint32_t)k_chunk << " tiles=" << (uint32_t)k_chunk_tiles
                               << ENDL();
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
                        DPRINT << "MATMUL done k=" << (uint32_t)k_chunk << ENDL();

                        // Apply causal mask (generated by writer kernel)
                        reconfig_data_format(cb_qk_im, cb_mask_in);
                        add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);

                        // Max reduction with optional element-wise max from previous K chunks
                        reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                        reduce_c<
                            PoolType::MAX,
                            ReduceDim::REDUCE_ROW,
                            cb_qk_im,
                            cb_identity_scale_in,
                            Sq_chunk_t,
                            Sk_chunk_t>(alias_cur_max, alias_prev_max, k_chunk > 0);

                        // Fused subtract max, scale, and exp with partial sum reduction
                        sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, Sk_chunk_t, scale_fp32>(
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

                        /* Online accumulation with previous results */
                        if (k_chunk > 0) {
                            // Compute exp(prev_max - cur_max) with fused scaling
                            sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);
                            cb_pop_front(alias_prev_max, Sq_chunk_t);

                            // Update previous sum: prev_sum *= exp(prev_max - cur_max)
                            mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);
                            // Add to current sum: cur_sum += prev_sum
                            add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);

                            // Update previous output: cur_out += prev_out * exp(prev_max - cur_max)
                            mul_block_bcast_cols<Sq_chunk_t, vDHt>(
                                alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out, true);
                        }

                        // Swap CB handles to prepare for next iteration
                        std::swap(alias_prev_sum, alias_cur_sum);
                        std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
                        std::swap(alias_prev_max, alias_cur_max);
                    }
                    // If !should_compute: skip ALL operations for this K chunk
                    // Skipped chunks contribute zero automatically via FlashAttention's online algorithm
                }

                // Final normalization: divide by sum to get proper attention weights
                matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);
                recip_block_inplace(alias_prev_sum, Sq_chunk_t);

                // Final output: multiply accumulated result by normalization factor
                pack_reconfig_data_format(cb_out);
                mul_block_bcast_cols<Sq_chunk_t, vDHt>(alias_mm2_prev_out, alias_prev_sum, cb_out, false);

                cb_pop_front(cb_q_in, q_chunk_tiles);
                cb_pop_front(alias_prev_max, Sq_chunk_t);
            }
        }
    }
}
}  // namespace NAMESPACE
