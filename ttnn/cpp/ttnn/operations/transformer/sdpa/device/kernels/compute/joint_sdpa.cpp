// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "compute_kernel_api.h"
#include "compute_common.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t Skt = get_compile_time_arg_val(2);
    constexpr uint32_t DHt = get_compile_time_arg_val(3);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(6);

    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(7);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(8);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(9);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(10);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(11);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(12);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(13);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(14);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(15);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(16);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(17);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(18);

    constexpr bool use_joint_mask = get_compile_time_arg_val(19) == 1;
    constexpr uint32_t mask_chunk_0 = get_compile_time_arg_val(20);
    constexpr uint32_t mask_chunk_1 = get_compile_time_arg_val(21);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(22);

    uint32_t argidx = 0;
    const uint32_t local_batch_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_end = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * DHt;

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

    mm_init(cb_q_in, cb_k_in, cb_qk_im);

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            for (uint32_t q_chunk = local_q_start; q_chunk < local_q_end; ++q_chunk) {
                // Set up ping pong buffers
                uint32_t alias_prev_sum = cb_sum_A;
                uint32_t alias_cur_sum = cb_sum_B;
                uint32_t alias_prev_max = cb_max_A;
                uint32_t alias_cur_max = cb_max_B;
                uint32_t alias_mm2_prev_out = cb_out_im_A;
                uint32_t alias_mm2_cur_out = cb_out_im_B;

                cb_wait_front(cb_q_in, q_chunk_tiles);

                for (uint32_t k_chunk = 0; k_chunk < k_num_chunks; ++k_chunk) {
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

                    if constexpr (use_joint_mask) {
                        if ((k_chunk == mask_chunk_0) || (k_chunk == mask_chunk_1)) {
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
                    reduce_c<
                        PoolType::MAX,
                        ReduceDim::REDUCE_ROW,
                        cb_qk_im,
                        cb_identity_scale_in,
                        Sq_chunk_t,
                        Sk_chunk_t>(alias_cur_max, alias_prev_max, k_chunk > 0);

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
                    sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, Sk_chunk_t, scale_fp32>(
                        alias_cur_max, alias_cur_sum);

                    cb_wait_front(cb_qk_im, qk_chunk_tiles);
                    /* OUT_IM = QK @ V_CHUNK */
                    matmul_blocks(
                        cb_qk_im,
                        cb_v_in,
                        alias_mm2_cur_out,
                        Sq_chunk_t,
                        DHt,
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
                    if (k_chunk > 0) {
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
                        mul_block_bcast_cols<Sq_chunk_t, DHt>(
                            alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out, true);
                    }

                    // Swap alias_prev_sum and alias_cur_sum
                    std::swap(alias_prev_sum, alias_cur_sum);
                    std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
                    std::swap(alias_prev_max, alias_cur_max);
                }

                /**
                 * Performs final row-reduction on the partial sum.
                 */
                matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);
                /* cb_cur_sum = 1.0 / cb_cur_sum */
                recip_block_inplace(alias_prev_sum, Sq_chunk_t);

                /* cb_out_accumulate_im *= cb_cur_sum */
                pack_reconfig_data_format(cb_out);
                mul_block_bcast_cols<Sq_chunk_t, DHt>(alias_mm2_prev_out, alias_prev_sum, cb_out, false);

                cb_pop_front(cb_q_in, q_chunk_tiles);
                // free up cb_prev_max after K chunks
                cb_pop_front(alias_prev_max, Sq_chunk_t);
            }
        }
    }
}
}  // namespace NAMESPACE
