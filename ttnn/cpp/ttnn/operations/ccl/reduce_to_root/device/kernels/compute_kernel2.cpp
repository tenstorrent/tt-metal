// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// this kernel receives l, m, s tensors from the reader and perform the following computations
// - inputs: l1, s1, m1 and l2, s2, m2; output: l, s, m
//----> m = max(m1, m2)
//- P1 = exp((m1 - m) * scale) (called exp_max_diff)
//- P2 = exp((m2 - m) * scale)
//----> s = s1 * P1 + s2 * P2
//----> l = l1 * P1 + l2 * P2
// writes the tensors l, s, m to the writer buffers

// for last round of device 1 add extra compute:
// out = v / s

// shoud do something similar to sdpa_flash_decode kernel (where out is the l)

// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#define EXP_APPROX_MODE false

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/compute_common.hpp"

inline void mul_block_bcast_cols2(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t rows, uint32_t cols) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: out_cb has rows*cols produced
    // Postcondition: in1_cb has rows consumed

    // This style of populating out_cb supports in0_cb being used as out_cb (ie. in-place)
    uint32_t num_tiles = rows * cols;
    mul_bcast_cols_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, rows);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            DPRINT << "mul col j=" << j << "\n";
            acquire_dst();
            mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
            cb_pop_front(in0_cb, 1);
            cb_reserve_back(out_cb, 1);
            pack_tile(0, out_cb);
            cb_push_back(out_cb, 1);
            DPRINT << "BEFORE release dst\n";
            release_dst();
            DPRINT << "after release dst\n";
        }
    }
}

namespace NAMESPACE {

constexpr uint32_t cb_out_o = get_compile_time_arg_val(0);                // l (output)
constexpr uint32_t cb_out_accumulate_im_2 = get_compile_time_arg_val(1);  // l2 (input)
constexpr uint32_t cb_out_accumulate_im = get_compile_time_arg_val(2);    // l1 (input)
constexpr uint32_t cb_prev_sum_2 = get_compile_time_arg_val(3);           // s2 (input)
constexpr uint32_t cb_m_in = get_compile_time_arg_val(4);                 // m1 (input)
constexpr uint32_t cb_prev_max = get_compile_time_arg_val(5);             // m2 (input)
constexpr uint32_t cb_cur_max = get_compile_time_arg_val(6);              // m (output)
constexpr uint32_t cb_exp_max_diff_2 = get_compile_time_arg_val(7);       // exp((m1-m)*scale) (P1)
constexpr uint32_t cb_prev_sum = get_compile_time_arg_val(8);             // s1 (input)
constexpr uint32_t cb_exp_max_diff = get_compile_time_arg_val(9);         // exp((m2-m)*scale) (P2)
constexpr uint32_t cb_cur_sum = get_compile_time_arg_val(10);             // s (output)
constexpr uint32_t cb_m_temp = get_compile_time_arg_val(11);              // temp for m
constexpr uint32_t cb_s_temp = get_compile_time_arg_val(12);              // temp for s
constexpr uint32_t cb_s1_temp = get_compile_time_arg_val(13);             // temp for s1
constexpr uint32_t cb_s2_temp = get_compile_time_arg_val(14);             // temp for s2
constexpr uint32_t cb_l1_temp = get_compile_time_arg_val(15);             // temp for l1
constexpr uint32_t cb_l2_temp = get_compile_time_arg_val(16);             // temp for l2
constexpr uint32_t scale_fp32 = get_compile_time_arg_val(17);
constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(18);
constexpr uint32_t vDHt = get_compile_time_arg_val(19);
constexpr uint32_t loop_size = get_compile_time_arg_val(20);

void MAIN {
    DPRINT << "reduce to root compute kernel started\n";
    DPRINT << "reading from cbs: "
           << " cb_out_accumulate_im_2: " << cb_out_accumulate_im_2 << " cb_out_accumulate_im: " << cb_out_accumulate_im
           << " cb_prev_sum _2: " << cb_prev_sum_2 << " cb_m_in: " << cb_m_in << " cb_prev_max: " << cb_prev_max
           << " cb_prev_sum: " << cb_prev_sum << "\n";
    const bool use_half_tile = true;
    constexpr int vector_mode = use_half_tile ? VectorMode::R : VectorMode::RC;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    mm_init(cb_out_accumulate_im, cb_out_accumulate_im, cb_out_accumulate_im);
    for (uint32_t loop_idx = 0; loop_idx < loop_size; ++loop_idx) {
        DPRINT << "starting loop idx: " << loop_idx << "\n";

        // Wait for all inputs to be available
        cb_wait_front(cb_m_in, Sq_chunk_t);
        cb_wait_front(cb_prev_max, Sq_chunk_t);
        cb_wait_front(cb_prev_sum, Sq_chunk_t);
        cb_wait_front(cb_prev_sum_2, Sq_chunk_t);
        cb_wait_front(cb_out_accumulate_im, out_chunk_tiles);
        cb_wait_front(cb_out_accumulate_im_2, out_chunk_tiles);
        DPRINT << "all inputs available\n";

        // move sum and max to temp cbs
        move_block<false>(cb_prev_sum_2, cb_s1_temp, Sq_chunk_t);
        DPRINT << "after moving s2\n";
        move_block<false>(cb_prev_sum, cb_s2_temp, Sq_chunk_t);
        DPRINT << "after moving s1\n";

        // Compute max(m1, m2) directly from source CBs
        DPRINT << "reserving and waiting before max block\n";
        max_block<vector_mode>(cb_m_in, cb_prev_max, cb_m_temp, Sq_chunk_t);
        DPRINT << "after max block\n";

        // P1 = exp((m1 - m_new) * scale)
        sub_exp_block<scale_fp32, vector_mode>(cb_m_in, cb_m_temp, cb_exp_max_diff_2, Sq_chunk_t);
        DPRINT << "after sub_exp_block1 (P1)\n";

        // s2 *= P1
        mul_block_inplace(cb_s1_temp, cb_exp_max_diff_2, Sq_chunk_t);
        DPRINT << "after s2 *= P1\n";

        // P2 = exp((m2 - m_new) * scale)
        sub_exp_block<scale_fp32, vector_mode>(cb_prev_max, cb_m_temp, cb_exp_max_diff, Sq_chunk_t);
        DPRINT << "after sub_exp_block2 (P2)\n";

        // s1 *= P2
        mul_block_inplace(cb_s2_temp, cb_exp_max_diff, Sq_chunk_t);
        DPRINT << "after s1 *= P2\n";

        // s_new = s2 * P1 + s1 * P2
        add_block_inplace<true>(cb_s1_temp, cb_s2_temp, Sq_chunk_t);
        DPRINT << "after cur sum\n";

        DPRINT << "START OF MUL L1\n";
        // l1 * P2 -> cb_l1_temp
        mul_block_bcast_cols(cb_out_accumulate_im, cb_exp_max_diff, cb_l1_temp, Sq_chunk_t, vDHt);
        DPRINT << "after l1 * P2\n";

        DPRINT << "START OF MUL L2\n";
        // l2 * P1 -> cb_l2_temp
        mul_block_bcast_cols(cb_out_accumulate_im_2, cb_exp_max_diff_2, cb_l2_temp, Sq_chunk_t, vDHt);
        DPRINT << "after l2 * P1\n";

        DPRINT << "START OF ADD\n";
        // l_new = l1 * P2 + l2 * P1
        add_block_inplace<true>(cb_l1_temp, cb_l2_temp, out_chunk_tiles);
        DPRINT << "after l add\n";

        if (loop_idx == 1) {
            move_block<false>(cb_s1_temp, cb_s_temp, Sq_chunk_t);
            recip_block_inplace<vector_mode>(cb_s_temp, Sq_chunk_t);
            mul_block_bcast_cols_inplace(cb_l1_temp, cb_s_temp, Sq_chunk_t, vDHt);
            DPRINT << "after final recip and mul\n";
            // cb_push_back(cb_s1_temp, Sq_chunk_t);
        }

        DPRINT << "after final div\n";

        // Output results
        move_block<true>(cb_l1_temp, cb_out_o, out_chunk_tiles);
        DPRINT << "after moving output l to output cb\n";
        move_block<true>(cb_m_temp, cb_cur_max, Sq_chunk_t);
        DPRINT << "after moving output m to output cb\n";
        move_block<true>(cb_s1_temp, cb_cur_sum, Sq_chunk_t);
        DPRINT << "after moving outputs to output cbs\n";

        // pop front all the cbs
        cb_pop_front(cb_m_in, Sq_chunk_t);
        cb_pop_front(cb_prev_max, Sq_chunk_t);
        cb_pop_front(cb_prev_sum, Sq_chunk_t);
        cb_pop_front(cb_prev_sum_2, Sq_chunk_t);
        // cb_pop_front(cb_out_accumulate_im, out_chunk_tiles);
        // cb_pop_front(cb_out_accumulate_im_2, out_chunk_tiles);

        // cb_push_back(cb_m_temp, Sq_chunk_t);
        // cb_push_back(cb_s_temp, Sq_chunk_t);
        // cb_push_back(cb_l1_temp, Sq_chunk_t);

        DPRINT << "reserved back temp cbs\n";

        // push back output cbs
        DPRINT << "end of loop\n";
    }
}

}  // namespace NAMESPACE
