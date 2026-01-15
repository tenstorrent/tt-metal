// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

struct OutputCBs {
    uint32_t l_cb;
    uint32_t s_cb;
    uint32_t m_cb;
};

template <uint32_t scale_fp32>
inline OutputCBs reduce_fct(
    uint32_t cb_out_o,
    uint32_t cb_out_accumulate_im_2,
    uint32_t cb_out_accumulate_im,
    uint32_t cb_prev_sum_2,
    uint32_t cb_m_in,
    uint32_t cb_prev_max,
    uint32_t cb_cur_max,
    uint32_t cb_exp_max_diff_2,
    uint32_t cb_prev_sum,
    uint32_t cb_exp_max_diff,
    uint32_t cb_cur_sum,
    uint32_t cb_m_temp,
    uint32_t cb_s_temp,
    uint32_t cb_s1_temp,
    uint32_t cb_s2_temp,
    uint32_t cb_l1_temp,
    uint32_t cb_l2_temp,
    uint32_t Sq_chunk_t,
    uint32_t vDHt,
    uint32_t loop_size) {
    constexpr int mode = VectorMode::R;
    const uint32_t out_tiles = Sq_chunk_t * vDHt;

    // Wait for all inputs to be available
    cb_wait_front(cb_m_in, Sq_chunk_t);
    cb_wait_front(cb_prev_max, Sq_chunk_t);
    cb_wait_front(cb_prev_sum, Sq_chunk_t);
    cb_wait_front(cb_prev_sum_2, Sq_chunk_t);
    cb_wait_front(cb_out_accumulate_im, out_tiles);
    cb_wait_front(cb_out_accumulate_im_2, out_tiles);

    // move sum to temp cbs
    move_block<false>(cb_prev_sum, cb_s1_temp, Sq_chunk_t);
    move_block<false>(cb_prev_sum_2, cb_s2_temp, Sq_chunk_t);

    // Compute max(m1, m2) directly from source CBs
    max_block<mode>(cb_m_in, cb_prev_max, cb_m_temp, Sq_chunk_t);

    // P1 = exp((m1 - m_new) * scale) - store in cb_exp_max_diff_2
    sub_exp_block<scale_fp32, mode>(cb_m_in, cb_m_temp, cb_exp_max_diff_2, Sq_chunk_t);

    // P2 = exp((m2 - m_new) * scale) - store in cb_exp_max_diff
    sub_exp_block<scale_fp32, mode>(cb_prev_max, cb_m_temp, cb_exp_max_diff, Sq_chunk_t);

    // s1 * P1 (element-wise)
    mul_block_inplace(cb_s1_temp, cb_exp_max_diff_2, Sq_chunk_t);

    // s2 * P2 (element-wise)
    mul_block_inplace(cb_s2_temp, cb_exp_max_diff, Sq_chunk_t);

    // s_new = s1 * P1 + s2 * P2
    add_block_inplace<true>(cb_s1_temp, cb_s2_temp, Sq_chunk_t);

    //  l1 * P1 -> cb_l1_temp (broadcast column 0 of P1)
    mul_block_bcast_cols(cb_out_accumulate_im, cb_exp_max_diff_2, cb_l1_temp, Sq_chunk_t, vDHt);

    //  l2 * P2 -> cb_l2_temp (broadcast column 0 of P2)
    mul_block_bcast_cols(cb_out_accumulate_im_2, cb_exp_max_diff, cb_l2_temp, Sq_chunk_t, vDHt);

    //  l_new = l1 * P1 + l2 * P2
    add_block_inplace<true>(cb_l1_temp, cb_l2_temp, out_tiles);

    return {cb_l1_temp, cb_s1_temp, cb_m_temp};
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
constexpr uint32_t int_l_cb = get_compile_time_arg_val(21);
constexpr uint32_t int_s_cb = get_compile_time_arg_val(22);
constexpr uint32_t int_m_cb = get_compile_time_arg_val(23);

void MAIN {
    // this kernel receives l, m, s tensors from the reader and perform the following computations
    // - inputs: l1, s1, m1 and l2, s2, m2; output: l, s, m
    //----> m = max(m1, m2)
    //- P1 = exp((m1 - m) * scale)
    //- P2 = exp((m2 - m) * scale)
    //----> s = s1 * P1 + s2 * P2
    //----> l = l1 * P1 + l2 * P2
    // writes the tensors l, s, m to the writer buffers

    // for last round of device 1 add extra compute:
    // out = l / s
    const bool use_half_tile = true;
    constexpr int vector_mode = use_half_tile ? VectorMode::R : VectorMode::RC;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    mm_init(cb_out_accumulate_im, cb_out_accumulate_im, cb_out_accumulate_im);

    OutputCBs output_cbs = reduce_fct<scale_fp32>(
        cb_out_o,
        cb_out_accumulate_im_2,
        cb_out_accumulate_im,
        cb_prev_sum_2,
        cb_m_in,
        cb_prev_max,
        cb_cur_max,
        cb_exp_max_diff_2,
        cb_prev_sum,
        cb_exp_max_diff,
        cb_cur_sum,
        cb_m_temp,
        cb_s_temp,
        cb_s1_temp,
        cb_s2_temp,
        cb_l1_temp,
        cb_l2_temp,
        Sq_chunk_t,
        vDHt,
        loop_size);

    if (loop_size == 1) {
        // Move results to output cbs
        move_block<true>(output_cbs.l_cb, cb_out_o, out_chunk_tiles);
        move_block<true>(output_cbs.m_cb, cb_cur_max, Sq_chunk_t);
        move_block<true>(output_cbs.s_cb, cb_cur_sum, Sq_chunk_t);

        // pop front all the input cbs from first reduction
        cb_pop_front(cb_m_in, Sq_chunk_t);
        cb_pop_front(cb_prev_max, Sq_chunk_t);
        cb_pop_front(cb_prev_sum, Sq_chunk_t);
        cb_pop_front(cb_prev_sum_2, Sq_chunk_t);
    } else {
        // Move results to intermediate CBs
        move_block<true>(output_cbs.l_cb, int_l_cb, out_chunk_tiles);
        move_block<true>(output_cbs.m_cb, int_m_cb, Sq_chunk_t);
        move_block<true>(output_cbs.s_cb, int_s_cb, Sq_chunk_t);

        cb_pop_front(cb_m_in, Sq_chunk_t);
        cb_pop_front(cb_prev_max, Sq_chunk_t);
        cb_pop_front(cb_prev_sum, Sq_chunk_t);
        cb_pop_front(cb_prev_sum_2, Sq_chunk_t);

        // do final reduction
        OutputCBs output_cbs2 = reduce_fct<scale_fp32>(
            cb_out_o,
            int_l_cb,
            cb_out_accumulate_im,
            int_s_cb,
            cb_m_in,
            int_m_cb,
            cb_cur_max,
            cb_exp_max_diff_2,
            cb_prev_sum,
            cb_exp_max_diff,
            cb_cur_sum,
            cb_m_temp,
            cb_s_temp,
            cb_s1_temp,
            cb_s2_temp,
            cb_l1_temp,
            cb_l2_temp,
            Sq_chunk_t,
            vDHt,
            loop_size);

        // final division
        move_block<false>(output_cbs2.s_cb, cb_s_temp, Sq_chunk_t);
        recip_block_inplace<vector_mode>(cb_s_temp, Sq_chunk_t);
        mul_block_bcast_cols_inplace(output_cbs2.l_cb, cb_s_temp, Sq_chunk_t, vDHt);

        move_block<true>(output_cbs2.l_cb, cb_out_o, out_chunk_tiles);
        move_block<true>(output_cbs2.m_cb, cb_cur_max, Sq_chunk_t);
        move_block<true>(output_cbs2.s_cb, cb_cur_sum, Sq_chunk_t);

        // pop front all the input cbs from second reduction
        cb_pop_front(cb_m_in, Sq_chunk_t);
        cb_pop_front(int_m_cb, Sq_chunk_t);
        cb_pop_front(cb_prev_sum, Sq_chunk_t);
        cb_pop_front(int_s_cb, Sq_chunk_t);
    }
}

}  // namespace NAMESPACE
