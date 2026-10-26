// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Length-balanced decode compute (see lean_schedule.hpp): runs the flash-decode loop once per piece of
// this core's slice, combines the partials of a root piece's children, and either finalizes the rows
// (whole segment or root) or hands the partial (o, m, l) to the writer (child).

#define LLK_ZEROFLAG_OUTLINE 1

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/lean_schedule.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"

__attribute__((noinline, noclone)) void move_block_ool(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    move_block<true>(in_cb, out_cb, num_tiles);
}

void kernel_main() {
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(5);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(6);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(8);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(9);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(11);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(12);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(13);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(14);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(24);
    constexpr uint32_t B = LEAN_B;
    constexpr uint32_t NKV = LEAN_NKV;
    constexpr uint32_t num_cores = LEAN_NUM_CORES;

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr VectorMode vector_mode = VectorMode::RC;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_m_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_l_in = tt::CBIndex::c_7;
    constexpr uint32_t cb_zero_in = tt::CBIndex::c_12;
    constexpr uint32_t cb_cur_pos = tt::CBIndex::c_15;
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
    constexpr uint32_t cb_child_o = tt::CBIndex::c_13;
    constexpr uint32_t cb_out_m = tt::CBIndex::c_17;
    constexpr uint32_t cb_out_l = tt::CBIndex::c_18;
    constexpr uint32_t cb_out_final = tt::CBIndex::c_20;

    const uint32_t core_idx = get_arg_val<uint32_t>(0);

    CircularBuffer(cb_cur_pos).wait_front(1);
    uint32_t pos[B];
    for (uint32_t b = 0; b < B; ++b) {
        pos[b] = read_tile_value(cb_cur_pos, 0, b);
    }
    CircularBuffer(cb_cur_pos).pop_front(1);

    constexpr uint32_t k_chunk_size = Sk_chunk_t * tt::constants::TILE_HEIGHT;
    auto get_n = [&](uint32_t b) -> uint32_t { return lean::chunks_for_pos(pos[b], k_chunk_size); };
    const lean::Schedule sched = lean::make_schedule<B, NKV>(num_cores, get_n);
    if (!sched.U || !sched.nonempty(core_idx)) {
        return;
    }

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, cb_qk_im);
    matmul_init(cb_q_in, cb_k_in);

    uint32_t q_batch = 0xFFFFFFFFu;
    lean::PieceIter<B, NKV> it(core_idx, sched);
    lean::Piece p;
    while (it.next(get_n, p)) {
        if (p.b != q_batch) {
            if (q_batch != 0xFFFFFFFFu) {
                CircularBuffer(cb_q_in).pop_front(q_chunk_tiles);
            }
            CircularBuffer(cb_q_in).wait_front(q_chunk_tiles);
            q_batch = p.b;
        }
        uint32_t cb_cur_max = cb_max_1;
        uint32_t cb_prev_max = cb_max_2;
        uint32_t cb_cur_sum = cb_sum_1;
        uint32_t cb_prev_sum = cb_sum_2;
        uint32_t cb_out_mm = cb_out_accumulate_im;
        const bool masked_last = p.j1 == p.n;

        for (uint32_t k_chunk = p.j0; k_chunk < p.j1; ++k_chunk) {
            reconfig_data_format(cb_k_in, cb_q_in);
            pack_reconfig_data_format(cb_qk_im);
            matmul_blocks(
                cb_q_in,
                cb_k_in,
                cb_qk_im,
                Sq_chunk_t,
                Sk_chunk_t,
                DHt,
                qk_in0_num_subblocks,
                qk_in1_num_subblocks,
                qk_in0_block_w,
                qk_subblock_h,
                qk_subblock_w,
                true,
                false,
                cb_mask_in,
                cb_zero_in);
            if (masked_last && k_chunk == p.j1 - 1) {
                reconfig_data_format(cb_qk_im, cb_mask_in);
                add_block_inplace<true>(cb_qk_im, cb_mask_in, qk_chunk_tiles);
            }
            reconfig_data_format(cb_qk_im, cb_identity_scale_in);
            pack_reconfig_data_format(cb_cur_max);
            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
                cb_cur_max, cb_prev_max, Sk_chunk_t, k_chunk > p.j0);
            reconfig_data_format(cb_qk_im, cb_cur_max);
            pack_reconfig_data_format(cb_qk_im);
            sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, scale_fp32, true, false, vector_mode>(
                cb_cur_max, cb_cur_sum, Sk_chunk_t);
            CircularBuffer(cb_qk_im).wait_front(qk_chunk_tiles);
            reconfig_data_format(cb_qk_im, cb_identity_scale_in);
            pack_reconfig_data_format(cb_cur_sum);
            reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
                cb_cur_sum, cb_cur_sum, Sk_chunk_t, false);
            reconfig_data_format(cb_v_in, cb_qk_im);
            pack_reconfig_data_format(cb_out_im);
            matmul_blocks(
                cb_qk_im,
                cb_v_in,
                cb_out_mm,
                Sq_chunk_t,
                vDHt,
                Sk_chunk_t,
                out_in0_num_subblocks,
                out_in1_num_subblocks,
                out_in0_block_w,
                out_subblock_h,
                out_subblock_w,
                false,
                false,
                cb_mask_in,
                cb_zero_in);
            reconfig_data_format_srca(cb_out_im);
            CircularBuffer(cb_qk_im).pop_front(qk_chunk_tiles);
            if (k_chunk == p.j0) {
                cb_out_mm = cb_out_im;
            } else {
                reconfig_data_format(cb_prev_max, cb_cur_max);
                pack_reconfig_data_format(cb_exp_max_diff);
                sub_exp_block<scale_fp32>(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
                CircularBuffer(cb_prev_max).pop_front(Sq_chunk_t);
                mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);
                reconfig_data_format(cb_out_accumulate_im, cb_exp_max_diff);
                pack_reconfig_data_format(cb_out_accumulate_im);
                mul_block_bcast_cols<Sq_chunk_t, vDHt, true, false>(
                    cb_out_accumulate_im, cb_exp_max_diff, cb_out_accumulate_im);
                reconfig_data_format(cb_cur_sum, cb_prev_sum);
                pack_reconfig_data_format(cb_cur_sum);
                add_block_inplace<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
                reconfig_data_format(cb_out_accumulate_im, cb_out_im);
                pack_reconfig_data_format(cb_out_accumulate_im);
                add_block_inplace<true>(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
            }
            reconfig_data_format(cb_cur_max, cb_cur_max);
            pack_reconfig_data_format(cb_prev_max);
            move_block<true>(cb_cur_max, cb_prev_max, Sq_chunk_t);
            move_block<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
        }

        if (p.is_root) {
            for (uint32_t child = 0; child < p.num_children; ++child) {
                move_block_ool(cb_l_in, cb_prev_sum_2, Sq_chunk_t);
                correction_block<scale_fp32, vector_mode>(
                    cb_m_in,
                    cb_prev_sum_2,
                    cb_cur_max,
                    cb_prev_max,
                    cb_cur_sum,
                    cb_prev_sum,
                    cb_exp_max_diff,
                    cb_exp_max_diff_2,
                    Sq_chunk_t);
                move_block_ool(cb_child_o, cb_out_accumulate_im_2, out_chunk_tiles);
                mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_exp_max_diff);
                mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im_2, cb_exp_max_diff_2);
                add_block_inplace<true>(cb_out_accumulate_im, cb_out_accumulate_im_2, out_chunk_tiles);
                CircularBuffer(cb_prev_max).pop_front(Sq_chunk_t);
                CircularBuffer(cb_m_in).pop_front(Sq_chunk_t);
                move_block_ool(cb_cur_max, cb_prev_max, Sq_chunk_t);
                move_block_ool(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
            }
        }

        if (p.is_child) {
            move_block_ool(cb_out_accumulate_im, cb_out_o, out_chunk_tiles);
            move_block_ool(cb_prev_max, cb_out_m, Sq_chunk_t);
            move_block_ool(cb_prev_sum, cb_out_l, Sq_chunk_t);
        } else {
            reconfig_data_format(cb_prev_sum, cb_prev_sum);
            pack_reconfig_data_format(cb_prev_sum);
            recip_block_inplace(cb_prev_sum, Sq_chunk_t);
            reconfig_data_format(cb_out_accumulate_im, cb_prev_sum);
            pack_reconfig_data_format(cb_out_accumulate_im);
            mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_prev_sum);
            pack_reconfig_data_format(cb_out_final);
            CircularBuffer(cb_prev_max).pop_front(Sq_chunk_t);
            move_block_ool(cb_out_accumulate_im, cb_out_final, out_chunk_tiles);
        }
    }
    if (q_batch != 0xFFFFFFFFu) {
        CircularBuffer(cb_q_in).pop_front(q_chunk_tiles);
    }
}
