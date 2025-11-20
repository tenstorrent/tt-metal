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

constexpr uint32_t MAX_PACK_UNTILIZE_WIDTH = 8;

namespace NAMESPACE {

constexpr uint32_t cb_out_o = get_compile_time_arg_val(0);                // l2 (input)
constexpr uint32_t cb_out_accumulate_im_2 = get_compile_time_arg_val(1);  // l (output)
constexpr uint32_t cb_l_in = get_compile_time_arg_val(2);                 // l1 (input)
constexpr uint32_t cb_prev_sum_2 = get_compile_time_arg_val(3);           // s2 (input)
constexpr uint32_t cb_m_in = get_compile_time_arg_val(4);                 // m1 (input)
constexpr uint32_t cb_prev_max = get_compile_time_arg_val(5);             // m2 (input)
constexpr uint32_t cb_cur_max = get_compile_time_arg_val(6);              // m (output)
constexpr uint32_t cb_exp_max_diff_2 = get_compile_time_arg_val(7);       // exp((m1-m)*scale) (P1)
constexpr uint32_t cb_prev_sum = get_compile_time_arg_val(8);             // s1 (input)
constexpr uint32_t cb_exp_max_diff = get_compile_time_arg_val(9);         // exp((m2-m)*scale) (P2)
constexpr uint32_t cb_cur_sum = get_compile_time_arg_val(10);             // s (output)
constexpr uint32_t cb_out_accumulate_im = get_compile_time_arg_val(11);   // l (output)
constexpr uint32_t scale_fp32 = get_compile_time_arg_val(12);
constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(13);
constexpr uint32_t vDHt = get_compile_time_arg_val(14);
constexpr uint32_t loop_size = get_compile_time_arg_val(15);

void MAIN {
    const bool use_half_tile = false;
    constexpr int vector_mode = use_half_tile ? VectorMode::R : VectorMode::RC;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    for (uint32_t loop_idx = 0; loop_idx < loop_size; ++loop_idx) {
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

        cb_pop_front(cb_prev_max, Sq_chunk_t);
        cb_pop_front(cb_m_in, Sq_chunk_t);

        // if do_final_division at the end, update OUT_ACC to be OUT_ACC / CUR_SUM
        if (loop_idx == 1) {
            // RECIP_CUR_SUM = 1 / CUR_SUM
            recip_block_inplace<vector_mode>(cb_cur_sum, Sq_chunk_t);
            // OUT_ACC = OUT_ACC * RECIP_CUR_SUM
            mul_block_bcast_cols_inplace(cb_out_accumulate_im, cb_cur_sum, Sq_chunk_t, vDHt);
        }

        // PREV_MAX <- CUR_MAX
        // PREV_SUM <- CUR_SUM
        move_block<true>(cb_cur_max, cb_prev_max, Sq_chunk_t);
        move_block<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
    }
}
}  // namespace NAMESPACE
