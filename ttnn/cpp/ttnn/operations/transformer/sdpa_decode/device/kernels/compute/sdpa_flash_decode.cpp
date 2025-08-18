// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/untilize.h"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "compute_common.hpp"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/untilize.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t MAX_PACK_UNTILIZE_WIDTH = 8;

namespace NAMESPACE {

void MAIN {
    // Compile time arguments

    // Input dimensions in tiles
    constexpr uint32_t St = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);

    // Matmul configs
    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(5);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(6);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(8);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(9);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(10);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(11);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(12);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(13);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(14);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(15);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(16);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(19);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(20);

    // Attention-specific parameters
    constexpr bool is_causal = get_compile_time_arg_val(21) == 1;
    constexpr bool use_attention_mask = get_compile_time_arg_val(22) == 1;
    constexpr bool use_attention_sink = get_compile_time_arg_val(23) == 1;
    constexpr uint32_t max_dynamic_chunk_size = get_compile_time_arg_val(24);
    constexpr bool tilize_q = get_compile_time_arg_val(25) == 1;
    constexpr uint32_t q_heads_parallel_factor = get_compile_time_arg_val(26);
    constexpr bool use_half_tile = get_compile_time_arg_val(27);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(28);
    constexpr bool enable_split_reader = get_compile_time_arg_val(29);

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;
    constexpr bool untilize_output = tilize_q;
    constexpr bool use_pack_untilize = out_chunk_tiles <= MAX_PACK_UNTILIZE_WIDTH;

    // CB index definitions
    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    uint32_t cb_k_in_0 = tt::CBIndex::c_1;
    uint32_t cb_k_in_1 = tt::CBIndex::c_1;

    // OPTIMIZATION:
    if constexpr (enable_split_reader) {
        cb_k_in_0 = tt::CBIndex::c_13;
        cb_k_in_1 = tt::CBIndex::c_14;
    }

    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_attention_sink = tt::CBIndex::c_4;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_m_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_l_in = tt::CBIndex::c_7;
    constexpr uint32_t cb_q_rm = tt::CBIndex::c_10;
    constexpr uint32_t cb_zero_in = tt::CBIndex::c_12;

    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_out_im = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_accumulate_im = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_1 = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_2 = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_1 = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_2 = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;
    constexpr uint32_t cb_prev_sum_2 = tt::CBIndex::c_21;
    constexpr uint32_t cb_exp_max_diff_2 = tt::CBIndex::c_22;
    constexpr uint32_t cb_out_accumulate_im_2 = tt::CBIndex::c_23;

    constexpr uint32_t cb_out_o = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_m = tt::CBIndex::c_17;
    constexpr uint32_t cb_out_l = tt::CBIndex::c_18;
    constexpr uint32_t cb_out_final = tt::CBIndex::c_20;

    // Runtime arguments
    uint32_t arg_idx = 0;
    const bool do_reduce = get_arg_val<uint32_t>(arg_idx++) == 1;
    const bool apply_mask_at_last_chunk = do_reduce && is_causal;
    const bool do_output = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t cur_head = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_output = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_pos_arg = get_arg_val<uint32_t>(arg_idx++);

    // Idle core
    // get_arg_val<uint32_t>(0) can go from 0-63 for the core_num; for active cores 65 is out of range so 65 indicates
    // an idle_core
    if (get_arg_val<uint32_t>(0) == 65) {
        return;
    }

    // Get cur_pos
    constexpr uint32_t cur_pos_base = St * 32 - 1;
    uint32_t cur_pos = cur_pos_base;  // default to non-causal, which we do attention on the entire kv cache. In this
                                      // case we set cur_pos to the last position
    if constexpr (is_causal) {
        // using UINT32_MAX as a flag to indicate that cur_pos is not provided as a list
        if (cur_pos_arg != UINT32_MAX) {
            cur_pos = cur_pos_arg;
        } else {
            constexpr uint32_t cb_index_id = tt::CBIndex::c_8;
            cb_wait_front(cb_index_id, 1);
            volatile uint32_t* index_addr_ptr;
            cb_get_tile(cb_index_id, 0, &index_addr_ptr);
            uint32_t cb_get_tile_offset = 4;  // Using cb_get_tile, the first 4 elements do not have the data
            cur_pos = index_addr_ptr[cb_get_tile_offset + (cur_batch / q_heads_parallel_factor)];
            cb_release_tile(cb_index_id);
        }
        if (cur_pos == UINT32_MAX) {
            // cur_pos of -1 indicates that the user should be skipped
            return;
        }
    }

    // Get dynamic chunk size for K in tiles
    auto Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(cur_pos);
    auto k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;

    // Get the sequence length assignment
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end] =
        get_runtime_args(cur_pos, cur_batch, core_num_in_reduce, num_cores_per_head, k_chunk_size_dynamic);
    if (k_chunk_start == k_chunk_end) {
        return;  // early exit because no computes needs to be done
    }

    // Get number of worker cores to wait for
    uint32_t num_cores_to_wait = num_cores_per_head - 1;
    if (num_cores_per_head > k_num_chunks) {
        num_cores_to_wait = k_num_chunks - 1;
    }

    // We tilize input Q if it is in ROW MAJOR layout
    {
        if constexpr (tilize_q) {
            compute_kernel_hw_startup(cb_q_rm, cb_q_in);
            tilize_init(cb_q_rm, q_chunk_tiles, cb_q_in);
            cb_wait_front(cb_q_rm, q_chunk_tiles);
            cb_reserve_back(cb_q_in, q_chunk_tiles);
            tilize_block(cb_q_rm, q_chunk_tiles, cb_q_in);
            tilize_uninit(cb_q_rm, cb_q_in);
            cb_push_back(cb_q_in, q_chunk_tiles);
            cb_pop_front(cb_q_rm, q_chunk_tiles);
            mm_init_short(cb_q_in, cb_k_in_0);
        } else {
            mm_init(cb_q_in, cb_k_in_0, cb_qk_im);
        }
        cb_wait_front(cb_q_in, q_chunk_tiles);
    }
    // Define dynamic matmul configs
#ifdef DYNAMIC_CHUNK_SIZE
    const uint32_t qk_subblock_h_dynamic = 1;
    const uint32_t qk_subblock_w_dynamic = Sk_chunk_t_dynamic;  // Guaranteed < DST
    const uint32_t qk_in0_num_subblocks_dynamic = 1;
    const uint32_t qk_in1_num_subblocks_dynamic = 1;
    const uint32_t out_in0_block_w_dynamic = Sk_chunk_t_dynamic;
    const uint32_t out_num_blocks_dynamic = 1;

    const uint32_t qk_chunk_tiles_dynamic = Sq_chunk_t * Sk_chunk_t_dynamic;
#else
    constexpr uint32_t qk_subblock_h_dynamic = qk_subblock_h;
    constexpr uint32_t qk_subblock_w_dynamic = qk_subblock_w;
    constexpr uint32_t qk_in0_num_subblocks_dynamic = qk_in0_num_subblocks;
    constexpr uint32_t qk_in1_num_subblocks_dynamic = qk_in1_num_subblocks;
    constexpr uint32_t out_in0_block_w_dynamic = out_in0_block_w;
    constexpr uint32_t out_num_blocks_dynamic = out_num_blocks;

    constexpr uint32_t qk_chunk_tiles_dynamic = Sq_chunk_t * Sk_chunk_t;
#endif

    // TODO: Used for legacy sfpu functions
    // - VectorMode::RC is equivalent to 32x32 tiles
    // - VectorMode::R is equivalent to 16x32 tiles
    // NOTE: Using VectorMode::RC for 16x32 tiles will be correct accuracy, just slower due to unnecessary math
    constexpr int vector_mode = use_half_tile ? VectorMode::R : VectorMode::RC;

    // We set up Ping Pong intermediate buffers between loops
    uint32_t cb_cur_max = cb_max_1;
    uint32_t cb_prev_max = cb_max_2;
    uint32_t cb_cur_sum = cb_sum_1;
    uint32_t cb_prev_sum = cb_sum_2;

    // Loop through all heads assigned to core
    for (uint32_t cur_head_work = 0; cur_head_work < num_heads_per_core; ++cur_head_work) {

        /******************************************************************************
         *                           FLASH ATTENTION LOOP                             *
         ******************************************************************************/
        /**
         * Compute Parameters (most are compile time but some are dynamic):
         * @tparam St - Total sequence length in tiles
         * @tparam DHt - Head dimension in tiles
         * @tparam Sq_chunk_t - Query chunk size in tiles
         * @tparam Sk_chunk_t - Key chunk size in tiles (dynamic)
         * @tparam qk_in0_block_w - QK matmul block width
         * @tparam qk_subblock_w - QK matmul subblock width (dynamic)
         * @tparam qk_subblock_h - QK matmul subblock height (dynamic)
         * @tparam qk_in0_num_subblocks - QK input0 subblocks (dynamic)
         * @tparam qk_in1_num_subblocks - QK input1 subblocks (dynamic)
         * @tparam qk_num_blocks - QK number of blocks
         * @tparam out_in0_block_w - Output matmul block width (dynamic)
         * @tparam out_subblock_w - Output matmul subblock width
         * @tparam out_subblock_h - Output matmul subblock height
         * @tparam out_in0_num_subblocks - Output input0 subblocks
         * @tparam out_in1_num_subblocks - Output input1 subblocks
         * @tparam out_num_blocks - Output number of blocks (dynamic)
         * @tparam is_causal - Whether to use causal attention (if mask is applied)
         * @tparam use_attention_mask - Whether to use attention mask for non-causal attention
         *
         * Circular Buffer Parameters:
         * @tparam cb_q_in - Query input buffer
         * @tparam cb_k_in - Key input buffer
         * @tparam cb_v_in - Value input buffer
         * @tparam cb_mask_in - Mask input buffer
         * @tparam cb_scale_in - Scale input buffer
         * @tparam cb_identity_scale_in - Identity scale buffer
         * @tparam cb_qk_im - QK intermediate buffer
         * @tparam cb_out_im - Output intermediate buffer
         * @tparam cb_out_accumulate_im - Output accumulate buffer
         * @tparam cb_cur_max - Current max buffer
         * @tparam cb_prev_max - Previous max buffer
         * @tparam cb_cur_sum - Current sum buffer
         * @tparam cb_prev_sum - Previous sum buffer
         * @tparam cb_exp_max_diff - Exp max diff buffer
         * @tparam cb_out_o - Output O buffer
         * @tparam cb_out_m - Output M buffer
         * @tparam cb_out_l - Output L buffer
         *
         * Runtime Parameters:
         * @param k_chunk_start - Start index of key chunk
         * @param k_chunk_end - End index of key chunk
         * @param do_reduce - Whether to perform reduction
         * @param qk_chunk_tiles - Number of QK chunk tiles (dynamic)
         * @param out_chunk_tiles - Number of output chunk tiles
         */
        /* START OF FLASH ATTENTION LOOP */
        {
            uint32_t cb_out_mm = cb_out_accumulate_im;

            // Loop through all K chunks
            for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
                // Reconfig register DF
                reconfig_data_format(cb_q_in, cb_k_in_0);
                pack_reconfig_data_format(cb_qk_im);

                // OPTIMIZATION: Add the attention mask directly on top of DST if chunk sizes are dynamic
#ifdef DYNAMIC_CHUNK_SIZE
                bool add_mask_fusion =
                    is_causal && k_chunk == k_chunk_end - 1 && apply_mask_at_last_chunk || use_attention_mask;
#else
                bool add_mask_fusion = false;
#endif

                /* QK = Q_CHUNK @ K_CHUNK */
                {
                    cb_matmul_blocks(
                        cb_q_in,
                        cb_k_in_0,
                        cb_k_in_1,
                        cb_qk_im,
                        Sq_chunk_t,
                        Sk_chunk_t_dynamic,
                        DHt,
                        qk_num_blocks,
                        qk_in0_num_subblocks_dynamic,
                        qk_in1_num_subblocks_dynamic,
                        qk_in0_block_w,
                        qk_subblock_h_dynamic,
                        qk_subblock_w_dynamic,
                        true,
                        add_mask_fusion,
                        cb_mask_in,
                        cb_zero_in,
                        enable_split_reader);
                }

                /* QK += MASK */
                if (!add_mask_fusion) {
                    if constexpr (is_causal) {
                        // For decode, we only apply mask at the last chunk for causal mode
                        if (k_chunk == k_chunk_end - 1 && apply_mask_at_last_chunk) {
                            reconfig_data_format(cb_qk_im, cb_mask_in);
                            add_block_inplace<false>(cb_qk_im, cb_mask_in, qk_chunk_tiles_dynamic);
                        }
                    } else {
                        if constexpr (use_attention_mask) {
                            reconfig_data_format(cb_qk_im, cb_mask_in);
                            add_block_inplace<true>(cb_qk_im, cb_mask_in, qk_chunk_tiles_dynamic);
                        }
                    }
                }

                /**
                 * OPTIMIZATION
                 * Typically, scores are multiplied by a scalar here, but an optimization was employed
                 * where the scaling is fused into exp both in exp(x - max) and exp(prev_max - cur_max).
                 * This gives us scaling for free on the performance-critical exp(x - max) computation.
                 */

                reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                pack_reconfig_data_format(cb_cur_max);

                /**
                 * OPTIMIZATION
                 * reduce_c can perform both reduce_max and eltwise max with previous result.
                 * if do_eltwise_max:
                 *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
                 * else:
                 *  cur_max = max(qk, dim=-1)
                 */
                reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
                    cb_cur_max, cb_prev_max, Sk_chunk_t_dynamic, k_chunk > k_chunk_start);

                /* QK -= cb_cur_max */
                /* QK = exp(QK)*/
                reconfig_data_format(cb_qk_im, cb_cur_max);
                pack_reconfig_data_format(cb_qk_im);

                /**
                 * sub_exp performs `QK = exp((QK - cur_max) * scale)`
                 */
                sub_exp_block_bcast_cols_inplace_reduce<cb_qk_im, Sq_chunk_t, scale_fp32, vector_mode>(
                    cb_cur_max, cb_cur_sum, Sk_chunk_t_dynamic);
                cb_wait_front(cb_qk_im, qk_chunk_tiles_dynamic);

                // Reconfig register DF
                reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                pack_reconfig_data_format(cb_cur_sum);

                /* reduce_c performs CUR_SUM = sum(QK, dim = -1) */
                reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
                    cb_cur_sum, cb_cur_sum, Sk_chunk_t_dynamic, false);

                /* OUT_IM = QK @ V_CHUNK */
                reconfig_data_format(cb_qk_im, cb_v_in);  // DEBUG
                pack_reconfig_data_format(cb_out_im);

                {
                    cb_matmul_blocks(
                        cb_qk_im,
                        cb_v_in,
                        cb_v_in,
                        cb_out_mm,
                        Sq_chunk_t,
                        vDHt,
                        Sk_chunk_t_dynamic,
                        out_num_blocks_dynamic,
                        out_in0_num_subblocks,
                        out_in1_num_subblocks,
                        out_in0_block_w_dynamic,
                        out_subblock_h,
                        out_subblock_w,
                        false /*transpose*/,
                        false,
                        cb_mask_in,
                        cb_zero_in,
                        false);
                }
                // Reconfig register DF
                reconfig_data_format_srca(cb_out_im);
                cb_pop_front(cb_qk_im, qk_chunk_tiles_dynamic);

                /* OUT_ACC += OUT_IM */
                if (k_chunk == k_chunk_start) {
                    cb_out_mm = cb_out_im;
                } else {
                    // When there is more than 1 chunk, we perform Lazy Softmax

                    // Reconfig register DF
                    reconfig_data_format(cb_prev_max, cb_cur_max);
                    pack_reconfig_data_format(cb_exp_max_diff);

                    /* EXP_MAX_DIFF = exp(PREV_MAX - CUR_MAX) */
                    sub_exp_block<scale_fp32, vector_mode>(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
                    cb_pop_front(cb_prev_max, Sq_chunk_t);

                    /* PREV_SUM *= EXP_MAX_DIFF */
                    mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);

                    /* OUT_ACC *= EXP_MAX_DIFF */
                    reconfig_data_format(cb_out_accumulate_im, cb_exp_max_diff);
                    pack_reconfig_data_format(cb_out_accumulate_im);
                    mul_block_bcast_cols(cb_out_accumulate_im, cb_exp_max_diff, cb_out_accumulate_im, Sq_chunk_t, vDHt);

                    /* CUR_SUM += PREV_SUM */
                    reconfig_data_format(cb_cur_sum, cb_prev_sum);
                    pack_reconfig_data_format(cb_cur_sum);
                    add_block_inplace<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);

                    /* OUT_ACC += OUT_IM */
                    reconfig_data_format(cb_out_accumulate_im, cb_out_im);
                    pack_reconfig_data_format(cb_out_accumulate_im);
                    add_block_inplace<true>(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                }

                if (k_chunk < k_chunk_end - 1 || do_reduce) {
                    // Move intermediate sum and max values to appropriate ping pong buffers
                    reconfig_data_format(cb_cur_max, cb_cur_max);
                    pack_reconfig_data_format(cb_prev_max);

                    // PREV_MAX <- CUR_MAX
                    move_block<true>(cb_cur_max, cb_prev_max, Sq_chunk_t);

                    // PREV_SUM <- CUR_SUM
                    move_block<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
                } else {
                    // Write results OUT_ACC, CUR_MAX, CUR_SUM to designated
                    // Write o, m, l into cb_out
                    move_block<true>(cb_out_accumulate_im, cb_out_o, out_chunk_tiles);
                    move_block<true>(cb_cur_max, cb_out_m, Sq_chunk_t);
                    move_block<true>(cb_cur_sum, cb_out_l, Sq_chunk_t);
                }
            }
        }
        /* END OF FLASH ATTENTION LOOP */
        // Perform reduction across intermediates from other cores if this is the reduction core
        if (do_reduce) {
            // cb_out_accumulate_im should contain o_1 (output from FA of itself's core)
            // cb_prev_max and cb_prev_sum should contain m_1 and l_1 (max and sum of logits of itself's core)

            if (k_chunk_end - k_chunk_start < k_num_chunks) {
                // This indicates that there are computes done by other workers.
                // We need to wait for them and send to reducer's compute
                // Iterate through each worker

                for (uint32_t i = 0; i < num_cores_to_wait; i++) {
                    // OUT_ACC_2 <- WORKER_OUT
                    move_block<true>(cb_out_o, cb_out_accumulate_im_2, out_chunk_tiles);

                    // PREV_SUM_2 <- WORKER_SUM
                    move_block<true>(cb_l_in, cb_prev_sum_2, Sq_chunk_t);

                    // CUR_MAX = max(PREV_MAX, WORKER_MAX)
                    max_block<vector_mode>(cb_m_in, cb_prev_max, cb_cur_max, Sq_chunk_t);  // pushed, pushed, popped

                    // EXP_MAX_DIFF_2 = exp((WORKER_MAX - CUR_MAX)*scale)
                    // PREV_SUM_2 *= EXP_MAX_DIFF_2
                    sub_exp_block<scale_fp32, vector_mode>(cb_m_in, cb_cur_max, cb_exp_max_diff_2, Sq_chunk_t);
                    mul_block_inplace(cb_prev_sum_2, cb_exp_max_diff_2, Sq_chunk_t);

                    /// EXP_MAX_DIFF = exp((PREV_MAX - CUR_MAX)*scale)
                    // PREV_SUM *= EXP_MAX_DIFF
                    sub_exp_block<scale_fp32, vector_mode>(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
                    mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);

                    /// CUR_SUM = PREV_SUM_2 + PREV_SUM
                    add_block(cb_prev_sum_2, cb_prev_sum, cb_cur_sum, Sq_chunk_t);

                    // OUT_ACC_2 *= EXP_MAX_DIFF
                    // OUT_ACC *= EXP_MAX_DIFF_2
                    mul_block_bcast_cols_inplace(cb_out_accumulate_im, cb_exp_max_diff, Sq_chunk_t, vDHt);
                    mul_block_bcast_cols_inplace(cb_out_accumulate_im_2, cb_exp_max_diff_2, Sq_chunk_t, vDHt);

                    // OUT_ACC = OUT_ACC + OUT_ACC_2
                    add_block_inplace<true>(cb_out_accumulate_im, cb_out_accumulate_im_2, out_chunk_tiles);

                    // PREV_MAX <- CUR_MAX
                    // PREV_SUM <- CUR_SUM
                    cb_pop_front(cb_prev_max, Sq_chunk_t);
                    cb_pop_front(cb_m_in, Sq_chunk_t);
                    move_block<true>(cb_cur_max, cb_prev_max, Sq_chunk_t);
                    move_block<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
                }
            }

            /* CUR_SUM = 1.0 / CUR_SUM */
            cb_push_back(cb_cur_sum, Sq_chunk_t);
            reconfig_data_format(cb_cur_sum, cb_cur_sum);
            pack_reconfig_data_format(cb_cur_sum);

            // Handle attention sink here
            if constexpr (use_attention_sink) {
                // m_new
                max_block<vector_mode>(cb_attention_sink, cb_prev_max, cb_cur_max, Sq_chunk_t);

                // exp(m - m_new)
                sub_exp_block<scale_fp32, vector_mode>(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);

                // l -> l * exp(m - m_new)
                mul_block_inplace(cb_cur_sum, cb_exp_max_diff, Sq_chunk_t);

                // exp(sink - m_new)
                sub_exp_block<scale_fp32, vector_mode>(cb_attention_sink, cb_cur_max, cb_exp_max_diff_2, Sq_chunk_t);
                cb_pop_front(cb_cur_max, Sq_chunk_t);

                // l -> l + exp(sink - m_new)
                add_block_inplace<true>(cb_cur_sum, cb_exp_max_diff_2, Sq_chunk_t);

                // O -> O * exp(m - m_new)
                mul_block_bcast_cols_inplace(cb_out_accumulate_im, cb_exp_max_diff, Sq_chunk_t, vDHt);
            }

            reconfig_data_format(cb_cur_sum, cb_cur_sum);
            pack_reconfig_data_format(cb_cur_sum);
            recip_block_inplace<vector_mode>(cb_cur_sum, Sq_chunk_t);

            /* OUT_ACC *= CUR_SUM */
            reconfig_data_format(cb_out_accumulate_im, cb_cur_sum);
            pack_reconfig_data_format(cb_out_accumulate_im);

            mul_block_bcast_cols_inplace(cb_out_accumulate_im, cb_cur_sum, Sq_chunk_t, vDHt);
            pack_reconfig_data_format(cb_out_final);

            // Untilize output to ROW MAJOR if input Q was also ROW MAJOR
            if constexpr (untilize_output) {
                // Conditionally use pack_untilize or untilize
                if constexpr (use_pack_untilize) {
                    pack_untilize_init<out_chunk_tiles>(cb_out_accumulate_im, cb_out_final);
                } else {
                    untilize_init(cb_out_accumulate_im);
                }
                cb_wait_front(cb_out_accumulate_im, out_chunk_tiles);
                cb_reserve_back(cb_out_final, out_chunk_tiles);
                if constexpr (use_pack_untilize) {
                    pack_untilize_block<out_chunk_tiles>(cb_out_accumulate_im, 1, cb_out_final);
                } else {
                    untilize_block(cb_out_accumulate_im, out_chunk_tiles, cb_out_final);
                }
                if constexpr (use_pack_untilize) {
                    pack_untilize_uninit(cb_out_final);
                } else {
                    untilize_uninit(cb_out_final);
                }
                cb_pop_front(cb_out_accumulate_im, out_chunk_tiles);
                cb_push_back(cb_out_final, out_chunk_tiles);
            } else {
                // Move output to buffer for the writer
                move_block<true>(cb_out_accumulate_im, cb_out_final, out_chunk_tiles);
            }
            // Free up cb_prev_max after K chunks
            cb_pop_front(cb_prev_max, Sq_chunk_t);
            cb_pop_front(cb_prev_sum, Sq_chunk_t);
        }
    }

    // Free up cb_q_in after Q chunks
    cb_pop_front(cb_q_in, q_chunk_tiles);
}
}  // namespace NAMESPACE
