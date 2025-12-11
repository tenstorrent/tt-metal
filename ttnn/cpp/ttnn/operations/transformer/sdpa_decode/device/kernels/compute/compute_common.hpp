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

/******************************************************************************
 *                                                                             *
 *                   Common Functions for Compute Kernels                      *
 *                                                                             *
 ******************************************************************************/

/******************************************************************************
 *                   Generic Compute Functions                                 *
 ******************************************************************************/

/**
 * out_cb = eltwise_max(in0, in1)
 */
template <int vector_mode = (int)VectorMode::RC>
void max_block(uint32_t in0, uint32_t in1, uint32_t out_cb, uint32_t num_tiles) {
    // inputs come in full, outputs go out full
    copy_tile_to_dst_init_short(in0);
    max_tile_init();

    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    cb_wait_front(in0, num_tiles);
    cb_wait_front(in1, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(in0, i, dst_reg_0);
        copy_tile(in1, i, dst_reg_1);
        max_tile(dst_reg_0, dst_reg_1, static_cast<int>(vector_mode));
        pack_tile(dst_reg_0, out_cb, i);
        release_dst();
    }
    cb_push_back(out_cb, num_tiles);
}

/**
 * out_cb = reduce[MAX,SUM](in0_cb * scale_cb)
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_cb,
    uint32_t scale_cb,
    uint32_t rows,
    int vector_mode = (int)VectorMode::RC>
void reduce_c(uint32_t out_cb, uint32_t prev_cb, uint32_t cols, bool do_eltwise_max = false) {
    // Precondition: in0_cb has rows*cols produced (row–major).
    // Precondition: scale_cb has 1 produced.
    // If do_eltwise_max == true, prev_cb has `rows` produced.
    // Postcondition: out_cb has `rows` produced.

    const uint32_t num_tiles = rows * cols;
    cb_wait_front(scale_cb, 1);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, rows);

    max_tile_init();
    constexpr uint32_t reduce_dst_idx = 0;
    constexpr uint32_t prev_max_dst_idx = 1;

    for (uint32_t r = 0; r < rows; ++r) {
        acquire_dst();
        reduce_init<pool_type, reduce_dim>(in0_cb, scale_cb, out_cb);
        for (uint32_t c = 0; c < cols; ++c) {
            reduce_tile<pool_type, reduce_dim>(in0_cb, scale_cb, r * cols + c, 0, reduce_dst_idx);
        }
        reduce_uninit();
        if (do_eltwise_max) {
            copy_tile_to_dst_init_short(prev_cb);
            copy_tile(prev_cb, r, prev_max_dst_idx);
            max_tile(reduce_dst_idx, prev_max_dst_idx, static_cast<int>(vector_mode));
        }
        pack_tile(reduce_dst_idx, out_cb);
        release_dst();
    }
    cb_push_back(out_cb, rows);
}

#ifdef TRISC_MATH
/**
 * recip_tile on only the columns 0:8 of a face
 */
template <bool legacy_compat = true>
void calculate_recip_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;
    if constexpr (legacy_compat) {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            sfpi::vFloat out = ckernel::sfpu::_reciprocal_compat_<APPROX ? 2 : 3>(in);
            // Note: negate check removed since in always >= 0.0
            // v_if (in < 0.0)
            // {
            //     out = -out;
            // }
            // v_endif;
            if constexpr (DST_ACCUM_MODE || APPROX) {
                sfpi::dst_reg[0] = out;
            } else {
                sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(out, 0));
            }
            sfpi::dst_reg += 2;
        }
    } else {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];

            if constexpr (APPROX) {
                sfpi::dst_reg[0] = ckernel::sfpu::_sfpu_reciprocal_<0>(in);
            } else {
                if constexpr (DST_ACCUM_MODE) {
                    sfpi::dst_reg[0] = ckernel::sfpu::_sfpu_reciprocal_<2>(in);
                } else {
                    sfpi::vFloat out = ckernel::sfpu::_sfpu_reciprocal_<1>(in);
                    sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(out, 0));
                }
            }

            sfpi::dst_reg += 2;
        }
    }
}

template <bool legacy_compat = true>
void recip_tile_first_column(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX /*APPROXIMATE*/>(
        calculate_recip_first_column<legacy_compat>, idst, (int)VectorMode::C);
}
#endif

/**
 * in_cb = 1 / in_cb
 */
template <int vector_mode = (int)VectorMode::RC>
void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Postcondition: in_cb has num_tiles produced
    copy_tile_to_dst_init_short(in_cb);
    recip_tile_init();
    cb_wait_front(in_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(in_cb, i, 0);
        MATH((recip_tile_first_column(0)));
        pack_tile(0, in_cb);
        release_dst();
    }
    cb_pop_front(in_cb, num_tiles);
    cb_reserve_back(in_cb, num_tiles);
    cb_push_back(in_cb, num_tiles);
}

/**
 * in0_cb = exp((in0_cb - in1_cb) * scale_fp32)
 */
template <uint32_t in0_cb, uint32_t rows, uint32_t scale_fp32, int vector_mode = (int)VectorMode::RC, uint32_t scale_cb>
void sub_exp_block_bcast_cols_inplace_reduce(uint32_t in1_cb, uint32_t reduce_cb, uint32_t cols) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows produced
    sub_bcast_cols_init_short(in0_cb, in1_cb);

    exp_tile_init<true, true, scale_fp32>();
    cb_wait_front(in0_cb, rows * cols);
    cb_wait_front(in1_cb, rows);
    cb_reserve_back(reduce_cb, rows);

#ifdef SUB_EXP_GRANULARITY
    uint32_t dst_tiles = SUB_EXP_GRANULARITY;
    uint32_t granularity = cols >> LOG2_SUB_EXP_GRANULARITY;
#else
    uint32_t dst_tiles = cols;
    uint32_t granularity = 1;
#endif
    uint32_t in0_index = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; u++) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);
                exp_tile<true, true>(j, vector_mode);
                in0_index++;
            }
            tile_regs_commit();
            cb_pop_front(in0_cb, dst_tiles);
            cb_reserve_back(in0_cb, dst_tiles);
            tile_regs_wait();

            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile(j, in0_cb);
            }
            cb_push_back(in0_cb, dst_tiles);
            tile_regs_release();
        }
    }
}

/**
 * in0_cb *= in1_cb
 */
void mul_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t rows, uint32_t cols) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows consumed

    uint32_t num_tiles = rows * cols;
    mul_bcast_cols_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, rows);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            acquire_dst();
            mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
            cb_pop_front(in0_cb, 1);
            cb_reserve_back(in0_cb, 1);
            pack_tile(0, in0_cb);
            cb_push_back(in0_cb, 1);
            release_dst();
        }
    }
    cb_pop_front(in1_cb, rows);
}

/**
 * out_cb = in0_cb * in1_cb
 */
void mul_block_bcast_cols(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t rows, uint32_t cols) {
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
            acquire_dst();
            mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
            cb_pop_front(in0_cb, 1);
            cb_reserve_back(out_cb, 1);
            pack_tile(0, out_cb);
            cb_push_back(out_cb, 1);
            release_dst();
        }
    }
    cb_pop_front(in1_cb, rows);
}

/**
 * in0_cb += in1_cb
 */
template <bool pop_in1>
void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    add_tiles_init(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        add_tiles(in0_cb, in1_cb, 0, i, 0);
        cb_pop_front(in0_cb, 1);
        cb_reserve_back(in0_cb, 1);
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, 1);
        release_dst();
    }
    if (pop_in1) {
        cb_pop_front(in1_cb, num_tiles);
    }
}

/**
 * out_cb = in0_cb + in1_cb
 */
void add_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    add_tiles_init(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        add_tiles(in0_cb, in1_cb, i, i, 0);
        pack_tile(0, out_cb, i);
        release_dst();
    }
    cb_push_back(out_cb, num_tiles);

    cb_pop_front(in0_cb, num_tiles);
    cb_pop_front(in1_cb, num_tiles);
}

/**
 * in0_cb *= in1_cb
 */
void mul_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles produced

    mul_tiles_init(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        invalidate_l1_cache();
        acquire_dst();
        mul_tiles(in0_cb, in1_cb, 0, i, 0);
        cb_pop_front(in0_cb, 1);
        cb_reserve_back(in0_cb, 1);
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, 1);
        release_dst();
    }
}

#ifdef TRISC_MATH
/**
 * exp_tile on only the columns 0:8 of a face
 */
template <bool SDPA_EXP_APPROX_MODE>
void calculate_exponential_first_column(int scale_bf16) {
    constexpr int ITERATIONS_HALF_FACE = 4;
    if constexpr (SDPA_EXP_APPROX_MODE) {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::vFloat result = ckernel::sfpu::
                _calculate_exponential_piecewise_<EXP_APPROX_MODE, true /*SCALE_EN*/, true /*SKIP_POSITIVE_CHECK*/>(
                    val, scale_bf16);
            sfpi::dst_reg[0] = result;

            // Stride by 2 to skip columns 8:16 of the face
            sfpi::dst_reg += 2;
        }
    } else {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            val = val * sfpi::s2vFloat16b(scale_bf16);
            sfpi::vFloat result = ckernel::sfpu::_sfpu_exp_improved_<DST_ACCUM_MODE>(val);
            sfpi::dst_reg[0] = result;

            // Stride by 2 to skip columns 8:16 of the face
            sfpi::dst_reg += 2;
        }
    }
}

template <bool SDPA_EXP_APPROX_MODE>
void exp_tile_first_column(uint32_t idst, int scale_bf16) {
    _llk_math_eltwise_unary_sfpu_params_<false /*APPROXIMATE*/>(
        calculate_exponential_first_column<SDPA_EXP_APPROX_MODE>, idst, (int)VectorMode::C, scale_bf16);
}
#endif

/**
 * out_cb = exp((in0_cb - in1_cb) * scale_fp32)
 */
template <uint32_t scale_fp32, int vector_mode = (int)VectorMode::RC>
void sub_exp_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: out_cb has num_tiles produced
    // Postcondition: in0_cb and in1_cb has num_tiles produced
    sub_tiles_init(in0_cb, in1_cb);
    exp_tile_init<EXP_APPROX_MODE, false>();
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

    // convert scale from fp32 to bf16
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    for (uint32_t i = 0; i < num_tiles; i++) {
        invalidate_l1_cache();
        acquire_dst();
        sub_tiles(in0_cb, in1_cb, i, i, 0);
        MATH((exp_tile_first_column<EXP_APPROX_MODE>(0, scale_bf16)));
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        release_dst();
    }
}

#ifdef TRISC_MATH
/**
 * The custom SFPI LLK function computes the following operation:
 * cur_max = max(prev_max, worker_max)
 * cur_sum = exp((worker_max - cur_max) * scale) * worker_sum + exp((prev_max - cur_max) * scale) * prev_sum
 * There are 4 results produced:
 * 1. exp_max_diff = exp((worker_max - cur_max) * scale), produced in dst_reg[prev_max_base_idx]
 * 2. exp_max_diff_2 = exp((prev_max - cur_max) * scale), produced in dst_reg[worker_max_base_idx]
 * 3. cur_sum produced in dst_reg[prev_sum_base_idx]
 * 4. cur_max produced in dst_reg[cur_max_base_idx]
 * fused_max_sub_exp_add_tile
 */
template <bool SDPA_EXP_APPROX_MODE>
void calculate_fused_max_sub_exp_add_tile(int scale_bf16) {
    constexpr int ITERATIONS_HALF_FACE = 4;
    constexpr uint32_t prev_max_base_idx = 0;      // dst_reg_0 (Tile 0)
    constexpr uint32_t worker_max_base_idx = 32;   // dst_reg_1 (Tile 1)
    constexpr uint32_t cur_max_base_idx = 64;      // dst_reg_2 (Tile 2)
    constexpr uint32_t prev_sum_base_idx = 96;     // dst_reg_3 (Tile 3)
    constexpr uint32_t worker_sum_base_idx = 128;  // dst_reg_4 (Tile 4)

    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        // Load inputs for this vector-slot into temporaries to avoid aliasing on dst_reg
        sfpi::vFloat prev_max_vec = sfpi::dst_reg[prev_max_base_idx];
        sfpi::vFloat worker_max_vec = sfpi::dst_reg[worker_max_base_idx];
        sfpi::vFloat prev_sum_vec = sfpi::dst_reg[prev_sum_base_idx];
        sfpi::vFloat worker_sum_vec = sfpi::dst_reg[worker_sum_base_idx];
        v_if(prev_max_vec < worker_max_vec) { sfpi::dst_reg[cur_max_base_idx] = worker_max_vec; }
        v_else { sfpi::dst_reg[cur_max_base_idx] = prev_max_vec; }
        v_endif;
        sfpi::vFloat cur_max = sfpi::dst_reg[cur_max_base_idx];

        // Compute differences
        sfpi::vFloat diff_prev = prev_max_vec - cur_max;
        sfpi::vFloat diff_worker = worker_max_vec - cur_max;

        // Exponentials of differences
        sfpi::vFloat exp_prev = ckernel::sfpu::
            _calculate_exponential_piecewise_<EXP_APPROX_MODE, true /*SCALE_EN*/, true /*SKIP_POSITIVE_CHECK*/>(
                diff_prev, scale_bf16);
        sfpi::vFloat exp_worker = ckernel::sfpu::
            _calculate_exponential_piecewise_<EXP_APPROX_MODE, true /*SCALE_EN*/, true /*SKIP_POSITIVE_CHECK*/>(
                diff_worker, scale_bf16);

        // Store exponentials for optional debug/pack-out
        sfpi::dst_reg[prev_max_base_idx] = exp_prev;
        sfpi::dst_reg[worker_max_base_idx] = exp_worker;

        // cur_sum = exp(worker_max - cur_max) * worker_sum + exp(prev_max - cur_max) * prev_sum
        sfpi::dst_reg[worker_sum_base_idx] = exp_worker * worker_sum_vec;
        sfpi::dst_reg[prev_sum_base_idx] = exp_prev * prev_sum_vec;
        sfpi::vFloat corr_worker_sum = sfpi::dst_reg[worker_sum_base_idx];
        sfpi::vFloat corr_prev_sum = sfpi::dst_reg[prev_sum_base_idx];
        sfpi::vFloat corr_sum = corr_worker_sum + corr_prev_sum;
        sfpi::dst_reg[prev_sum_base_idx] = corr_sum;
        sfpi::dst_reg += 2;
    }
}

template <bool SDPA_EXP_APPROX_MODE, int vector_mode = (int)VectorMode::C>
void fused_max_sub_exp_add_tile(uint32_t idst, int scale_bf16) {
    _llk_math_eltwise_unary_sfpu_params_<false /*APPROXIMATE*/>(
        calculate_fused_max_sub_exp_add_tile<SDPA_EXP_APPROX_MODE>, idst, vector_mode, scale_bf16);
}
#endif

template <uint32_t scale_fp32, int vector_mode = (int)VectorMode::C>
void correction_block(
    uint32_t cb_worker_max,
    uint32_t cb_worker_sum,
    uint32_t cb_cur_max,
    uint32_t cb_prev_max,
    uint32_t cb_cur_sum,
    uint32_t cb_prev_sum,
    uint32_t cb_exp_max_diff,
    uint32_t cb_exp_max_diff_2,
    uint32_t num_head_tiles) {
    cb_wait_front(cb_worker_max, num_head_tiles);
    cb_wait_front(cb_worker_sum, num_head_tiles);
    cb_wait_front(cb_prev_max, num_head_tiles);
    cb_wait_front(cb_prev_sum, num_head_tiles);

    cb_reserve_back(cb_cur_max, num_head_tiles);
    cb_reserve_back(cb_cur_sum, num_head_tiles);
    cb_reserve_back(cb_exp_max_diff, num_head_tiles);
    cb_reserve_back(cb_exp_max_diff_2, num_head_tiles);

    constexpr uint32_t dst_reg_0 = 0;  // dst_reg_0 is used for prev_max
    constexpr uint32_t dst_reg_1 = 1;  // dst_reg_1 is used for worker_max
    constexpr uint32_t dst_reg_2 = 2;  // dst_reg_2 is used for cur_max
    constexpr uint32_t dst_reg_3 = 3;  // dst_reg_3 is used for prev_sum, returns cur_sum
    constexpr uint32_t dst_reg_4 = 4;  // dst_reg_4 is used for worker_sum

    // convert scale from fp32 to bf16
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    for (uint32_t i = 0; i < num_head_tiles; i++) {
        acquire_dst();
        copy_tile_to_dst_init_short(cb_worker_max);
        exp_tile_init<EXP_APPROX_MODE, false>();
        max_tile_init();
        copy_tile(cb_prev_max, i, dst_reg_0);
        copy_tile(cb_worker_max, i, dst_reg_1);
        copy_tile(cb_prev_sum, i, dst_reg_3);
        copy_tile(cb_worker_sum, i, dst_reg_4);
        MATH((fused_max_sub_exp_add_tile<EXP_APPROX_MODE, vector_mode>(0, scale_bf16)));
        pack_tile(dst_reg_0, cb_exp_max_diff);
        pack_tile(dst_reg_1, cb_exp_max_diff_2);
        pack_tile(dst_reg_2, cb_cur_max);
        pack_tile(dst_reg_3, cb_cur_sum);
        cb_push_back(cb_cur_max, 1);
        cb_push_back(cb_cur_sum, 1);
        cb_push_back(cb_exp_max_diff, 1);
        cb_push_back(cb_exp_max_diff_2, 1);
        release_dst();
    }
    cb_pop_front(cb_prev_sum, num_head_tiles);
    cb_pop_front(cb_worker_sum, num_head_tiles);
}

/**
 * in_cb -> out_cb
 */
template <bool pop_in_cb>
void move_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Precondition: out_cb has num_tiles free
    // Postcondition: in_cb has num_tiles consumed
    // Postcondition: out_cb has num_tiles produced

    copy_tile_to_dst_init_short(in_cb);

    cb_wait_front(in_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

#pragma GCC unroll 0
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        copy_tile(in_cb, i, 0 /*dst*/);
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        release_dst();
    }
    if (pop_in_cb) {
        cb_pop_front(in_cb, num_tiles);
    }
}

/**
 * out_cb = in0_cb @ in1_cb
 */
ALWI void cb_matmul_blocks(
    const uint32_t& in0_cb,
    const uint32_t& in1_cb,
    const uint32_t& out_cb,
    const uint32_t& M,
    const uint32_t& N,
    const uint32_t& K,
    const uint32_t& num_blocks,
    const uint32_t& in0_num_subblocks,
    const uint32_t& in1_num_subblocks,
    const uint32_t& in0_block_w,
    const uint32_t& subblock_h,
    const uint32_t& subblock_w,
    const bool& transpose,
    const bool& add_mask,
    const uint32_t& mask_cb,
    const uint32_t& zero_cb) {
    // precondition: in0_cb has M*K produced
    // preconditino: in1_cb has K*N produced
    // postcondition: in0_cb is full, in1_cb is empty
    // postcondition: out_cb has M*N produced

    mm_block_init_short(
        in0_cb, in1_cb, transpose /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    reconfig_data_format(in1_cb, in0_cb);
    cb_wait_front(in1_cb, K * N);

    uint32_t output_num_tiles = M * N;
    cb_reserve_back(out_cb, output_num_tiles);
    uint32_t out_subblock_num_tiles = subblock_h * subblock_w;
    uint32_t in0_index_offset = 0;

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        uint32_t in1_index_offset = 0;

        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            tile_regs_acquire();
            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                matmul_block(
                    in0_cb, in1_cb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, in0_block_w);
                in0_index++;
                in1_index += N;
            }
            if (add_mask) {
                cb_wait_front(mask_cb, out_subblock_num_tiles);
                cb_wait_front(zero_cb, 1);
                add_tiles_init(zero_cb, mask_cb, true);
                for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                    add_tiles(zero_cb, mask_cb, 0, i, i);
                }
                cb_pop_front(mask_cb, out_subblock_num_tiles);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(out_cb, out_subblock_num_tiles);
            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                pack_tile(i, out_cb);
            }
            cb_push_back(out_cb, out_subblock_num_tiles);
            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
    }
    cb_pop_front(in1_cb, K * N);
}
