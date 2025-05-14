// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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

#include "cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "compute_common.hpp"

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t St = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(3);

    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(4);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(5);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(7);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(8);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(11);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(12);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(13);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(14);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(15);
    constexpr uint32_t num_cores_per_batch = get_compile_time_arg_val(16);
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(17);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(18);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(19);
    constexpr bool is_causal = get_compile_time_arg_val(20) == 1;
    constexpr bool use_attention_mask = get_compile_time_arg_val(21) == 1;
    constexpr uint32_t max_dynamic_chunk_size = get_compile_time_arg_val(22);

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * DHt;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;  // reuse it also for reduce input o
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_scale_in = tt::CBIndex::c_4;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_m_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_l_in = tt::CBIndex::c_7;

    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_out_im = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_accumulate_im = tt::CBIndex::c_26;
    constexpr uint32_t cb_cur_max = tt::CBIndex::c_27;
    constexpr uint32_t cb_prev_max = tt::CBIndex::c_28;
    constexpr uint32_t cb_cur_sum = tt::CBIndex::c_29;
    constexpr uint32_t cb_prev_sum = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;
    constexpr uint32_t cb_prev_sum_2 = tt::CBIndex::c_21;
    constexpr uint32_t cb_exp_max_diff_2 = tt::CBIndex::c_22;
    constexpr uint32_t cb_out_accumulate_im_2 = tt::CBIndex::c_23;

    constexpr uint32_t cb_out_o = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_m = tt::CBIndex::c_17;
    constexpr uint32_t cb_out_l = tt::CBIndex::c_18;
    constexpr uint32_t cb_out_final = tt::CBIndex::c_20;

    uint32_t arg_idx = 0;
    const bool do_reduce = get_arg_val<uint32_t>(arg_idx++) == 1;
    const bool apply_mask_at_last_chunk = do_reduce && is_causal;
    const bool do_output = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t cur_head = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_output = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_pos_arg = get_arg_val<uint32_t>(arg_idx++);

    // idle core
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
            cur_pos = index_addr_ptr[4 + cur_batch];
            cb_release_tile(cb_index_id);
        }

        if (cur_pos == UINT32_MAX) {
            // cur_pos of -1 indicates that the user should be skipped
            return;
        }
    }

    auto Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(cur_pos);
    auto k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;

    // Sequence length assignment
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end] =
        get_runtime_args(cur_pos, cur_batch, core_num_in_reduce, num_cores_per_head, k_chunk_size_dynamic);
    if (k_chunk_start == k_chunk_end) {
        return;  // early exit because no computes needs to be done
    }

    uint32_t num_cores_to_wait = num_cores_per_head - 1;
    if (num_cores_per_head > k_num_chunks) {
        num_cores_to_wait = k_num_chunks - 1;
    }

    mm_init(cb_q_in, cb_k_in, cb_out_final);
    cb_wait_front(cb_q_in, q_chunk_tiles);

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

    for (uint32_t cur_head_work = 0; cur_head_work < num_heads_per_core; ++cur_head_work) {
        flash_attention_loop<
            // Compile-time dimension parameters
            St,
            DHt,
            Sq_chunk_t,
            out_chunk_tiles,
            // QK matmul block parameters
            qk_in0_block_w,
            qk_num_blocks,
            // Output matmul block parameters
            out_subblock_w,
            out_subblock_h,
            out_in0_num_subblocks,
            out_in1_num_subblocks,
            // Attention parameters
            is_causal,
            use_attention_mask,
            // Circular buffer indices
            cb_q_in,
            cb_k_in,
            cb_v_in,
            cb_mask_in,
            cb_scale_in,
            cb_identity_scale_in,
            cb_qk_im,
            cb_out_im,
            cb_out_accumulate_im,
            cb_cur_max,
            cb_prev_max,
            cb_cur_sum,
            cb_prev_sum,
            cb_exp_max_diff,
            cb_out_o,
            cb_out_m,
            cb_out_l>(
            k_chunk_start,
            k_chunk_end,
            Sk_chunk_t_dynamic,
            qk_subblock_h_dynamic,
            qk_subblock_w_dynamic,
            qk_in0_num_subblocks_dynamic,
            qk_in1_num_subblocks_dynamic,
            out_in0_block_w_dynamic,
            out_num_blocks_dynamic,
            qk_chunk_tiles_dynamic,
            do_reduce,
            apply_mask_at_last_chunk);

        // do reduction across intermediates from other cores if this is the reduction core
        if (do_reduce) {
            // cb_out_accumulate_im should contain o_1
            // cb_prev_max and cb_prev_sum should contain m_1 and l_1

            if (k_chunk_end - k_chunk_start < k_num_chunks) {
                // This indicates that there are computes done by other workers. Needs to wait for them and send to
                // reducer's compute
                for (uint32_t i = 0; i < num_cores_to_wait; i++) {
                    // reconfig_data_format(cb_q_in, cb_q_in); // DEBUG
                    // pack_reconfig_data_format(cb_out_accumulate_im_2);
                    copy_block(cb_out_o, cb_out_accumulate_im_2, q_chunk_tiles);
                    copy_block(cb_l_in, cb_prev_sum_2, Sq_chunk_t);
                    max_block(cb_m_in, cb_prev_max, cb_cur_max, Sq_chunk_t);  // pushed, pushed, popped

                    // l = torch.exp(m_2 - m) * l_2 + torch.exp(m_1 - m) * l_1
                    /// l1 = torch.exp(m_2 - m) * l_2
                    // reconfig_data_format(cb_prev_max_2, cb_cur_max); // DEBUG
                    // pack_reconfig_data_format(cb_exp_max_diff_2);
                    sub_exp_block(cb_m_in, cb_cur_max, cb_exp_max_diff_2, Sq_chunk_t);
                    mul_block_inplace(cb_prev_sum_2, cb_exp_max_diff_2, Sq_chunk_t);
                    /// l2 = torch.exp(m_1 - m) * l_1
                    // reconfig_data_format(cb_prev_max, cb_cur_max); // DEBUG
                    // pack_reconfig_data_format(cb_exp_max_diff);
                    sub_exp_block(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
                    mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);
                    /// l = l1 + l2
                    // reconfig_data_format(cb_cur_sum, cb_prev_sum); // DEBUG
                    // pack_reconfig_data_format(cb_cur_sum);
                    add_block(cb_prev_sum_2, cb_prev_sum, cb_cur_sum, Sq_chunk_t);

                    // reconfig_data_format(cb_out_accumulate_im, cb_exp_max_diff); // DEBUG
                    // pack_reconfig_data_format(cb_out_accumulate_im);
                    mul_block_bcast_cols_inplace(cb_out_accumulate_im, cb_exp_max_diff, Sq_chunk_t, DHt);
                    mul_block_bcast_cols_inplace(cb_out_accumulate_im_2, cb_exp_max_diff_2, Sq_chunk_t, DHt);

                    // reconfig_data_format(cb_out_accumulate_im, cb_out_accumulate_im_2);
                    // pack_reconfig_data_format(cb_out_accumulate_im);
                    add_block_inplace<true>(cb_out_accumulate_im, cb_out_accumulate_im_2, q_chunk_tiles);

                    // copy tiles
                    // reconfig_data_format(cb_cur_max, cb_cur_max); // DEBUG
                    // pack_reconfig_data_format(cb_prev_max);
                    cb_pop_front(cb_prev_max, Sq_chunk_t);
                    cb_pop_front(cb_m_in, Sq_chunk_t);
                    copy_block(cb_cur_max, cb_prev_max, Sq_chunk_t);
                    copy_block(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
                }
            }

            /* cb_cur_sum = 1.0 / cb_cur_sum */
            cb_push_back(cb_cur_sum, Sq_chunk_t);

            reconfig_data_format(cb_cur_sum, cb_cur_sum);  // DEBUG
            pack_reconfig_data_format(cb_cur_sum);
            recip_block_inplace(cb_cur_sum, Sq_chunk_t);

            /* cb_out_accumulate_im *= cb_cur_sum */
            reconfig_data_format(cb_out_accumulate_im, cb_cur_sum);  // DEBUG
            pack_reconfig_data_format(cb_out_accumulate_im);
            mul_block_bcast_cols_inplace(cb_out_accumulate_im, cb_cur_sum, Sq_chunk_t, DHt);
            pack_reconfig_data_format(cb_out_final);
            copy_block(cb_out_accumulate_im, cb_out_final, out_chunk_tiles);

            // free up cb_prev_max after K chunks
            cb_pop_front(cb_prev_max, Sq_chunk_t);
            cb_pop_front(cb_prev_sum, Sq_chunk_t);
        }
    }
    cb_pop_front(cb_q_in, q_chunk_tiles);
}
}  // namespace NAMESPACE
