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
void max_block_inplace(uint32_t in0, uint32_t in1, uint32_t num_tiles) {
    // inputs come in full, outputs go out full
    copy_tile_to_dst_init_short(in0);
    max_tile_init();

    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    cb_wait_front(in0, num_tiles);
    cb_wait_front(in1, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(in0, 0, dst_reg_0);
        copy_tile(in1, i, dst_reg_1);
        cb_pop_front(in0, 1);
        cb_reserve_back(in0, 1);
        max_tile(dst_reg_0, dst_reg_1, static_cast<int>(VectorMode::C));
        pack_tile(dst_reg_0, in0);
        cb_push_back(in0, 1);
        release_dst();
    }
}

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
        max_tile(dst_reg_0, dst_reg_1, static_cast<int>(VectorMode::C));
        pack_tile(dst_reg_0, out_cb, i);
        release_dst();
    }
    cb_push_back(out_cb, num_tiles);
}

template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_cb,
    uint32_t scale_cb,
    uint32_t rows,
    uint32_t out_cb,
    uint32_t prev_cb>
void reduce_c(uint32_t cols, bool do_eltwise_max = false) {
    // Precondition: in0_cb has rows*cols produced. in0_cb has tiles in row-major order
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has rows free
    // Postcondition: in0_cb has rows*cols produced
    // Precondition: scale_cb has 1 produced
    // Postcondition: out_cb has rows produced

    reduce_init<pool_type, reduce_dim>(in0_cb, scale_cb, out_cb);

    max_tile_init();
    const uint32_t num_tiles = rows * cols;
    cb_wait_front(scale_cb, 1);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, rows);

    constexpr uint32_t reduce_dst_idx = 0;
    constexpr uint32_t prev_max_dst_idx = 1;
    for (uint32_t i = 0; i < rows; i++) {
        acquire_dst();
        for (uint32_t j = 0; j < cols; j++) {
            reduce_tile<pool_type, reduce_dim>(in0_cb, scale_cb, i * cols + j, 0, reduce_dst_idx);
        }
        if (do_eltwise_max) {
            copy_tile_to_dst_init_short(prev_cb);
            copy_tile(prev_cb, i, prev_max_dst_idx);
            max_tile(reduce_dst_idx, prev_max_dst_idx, static_cast<int>(VectorMode::C));
        }
        pack_tile(reduce_dst_idx, out_cb);
        release_dst();
    }
    cb_push_back(out_cb, rows);
    reduce_uninit();
}

void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Postcondition: in_cb has num_tiles produced
    copy_tile_to_dst_init_short(in_cb);
    recip_tile_init();
    cb_wait_front(in_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(in_cb, i, 0);
        recip_tile(0, static_cast<int>(VectorMode::C));
        pack_tile(0, in_cb);
        release_dst();
    }
    cb_pop_front(in_cb, num_tiles);
    cb_reserve_back(in_cb, num_tiles);
    cb_push_back(in_cb, num_tiles);
}

void sub_exp_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t rows, uint32_t cols) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows produced

    sub_bcast_cols_init_short(in0_cb, in1_cb);
    exp_tile_init<true>();
    cb_wait_front(in0_cb, rows * cols);
    cb_wait_front(in1_cb, rows);

#ifdef SUB_EXP_GRANULARITY
    constexpr uint32_t dst_tiles = SUB_EXP_GRANULARITY;
    uint32_t granularity = cols >> LOG2_SUB_EXP_GRANULARITY;
#else
    uint32_t dst_tiles = cols;
    uint32_t granularity = 1;
#endif
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; u++) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);
                exp_tile<true>(j);
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

template <uint32_t in0_cb, uint32_t rows, uint32_t scale_fp32>
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
    constexpr uint32_t dst_tiles = SUB_EXP_GRANULARITY;
    constexpr uint32_t granularity = cols >> LOG2_SUB_EXP_GRANULARITY;
#else
    uint32_t dst_tiles = cols;
    uint32_t granularity = 1;
#endif
    uint32_t in0_index = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; u++) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sub_tiles_bcast_cols(in0_cb, in1_cb, in0_index, i, j);
                exp_tile<true, true>(j);
                in0_index++;
            }
            tile_regs_commit();
            tile_regs_wait();

            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile(j, in0_cb);
            }

            // While we have results in DST, take advantage of L1 accumulation
            // to reduce row x cols tiles to rows x 1 tiles.
            if (u > 0) {
                // If on the same row, keep accumulating
                PACK((llk_pack_reconfig_l1_acc(1)));
            }
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile<true>(j, reduce_cb, i);

                if (u == 0 && j == 0) {
                    // If this was the first tile of a row, start accumulating
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }
            tile_regs_release();
            PACK((llk_pack_reconfig_l1_acc(0)));
        }
    }
    cb_pop_front(in0_cb, rows * cols);
    cb_reserve_back(in0_cb, rows * cols);
    cb_push_back(in0_cb, rows * cols);
    cb_push_back(reduce_cb, rows);
}

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

template <uint32_t rows, uint32_t cols>
void mul_block_bcast_cols(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, bool pack_accumulate = false) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Precondition: out_cb has rows*cols produced
    // Postcondition: in0_cb empty
    // Postcondition: in1_cb empty
    // Postcondition: out_cb has rows*cols produced

    constexpr uint32_t num_tiles = rows * cols;
#ifdef DHT_GRANULARITY
    constexpr uint32_t dst_tiles = DHT_GRANULARITY;
    constexpr uint32_t granularity = cols >> LOG2_DHT_GRANULARITY;
#else
    constexpr uint32_t dst_tiles = cols;
    constexpr uint32_t granularity = 1;
#endif
    mul_bcast_cols_init_short(in0_cb, in1_cb);
    PACK((llk_pack_reconfig_l1_acc(pack_accumulate)));
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, rows);
    if (!pack_accumulate) {
        cb_reserve_back(out_cb, num_tiles);
    }
    uint32_t in0_index = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; ++u) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                mul_tiles_bcast_cols(in0_cb, in1_cb, in0_index, i, j);
                in0_index++;
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile(j, out_cb);
            }
            tile_regs_release();
        }
    }
    PACK((llk_pack_reconfig_l1_acc(false)));
    cb_pop_front(in1_cb, rows);
    cb_pop_front(in0_cb, num_tiles);
    if (!pack_accumulate) {
        cb_push_back(out_cb, num_tiles);
    } else {
        cb_pop_front(out_cb, num_tiles);
        cb_reserve_back(out_cb, num_tiles);
        cb_push_back(out_cb, num_tiles);
    }
}

void mul_block_bcast_scalar_inplace(uint32_t in0_cb, uint32_t in1_scalar_cb, uint32_t num_tiles) {
    // Precondition: in0_cb has num_tiles produced
    // Precondition: in1_scalar_cb has 1 produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_scalar_cb has 1 produced

#ifdef MUL_BCAST_GRANULARITY
    constexpr uint32_t dst_tiles = MUL_BCAST_GRANULARITY;
    uint32_t granularity = num_tiles >> LOG2_MUL_BCAST_GRANULARITY;
#else
    uint32_t dst_tiles = num_tiles;
    uint32_t granularity = 1;
#endif

    reconfig_data_format(in0_cb, in1_scalar_cb);
    mul_tiles_bcast_scalar_init_short(in0_cb, in1_scalar_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_scalar_cb, 1);
    for (uint32_t g = 0; g < granularity; ++g) {
        acquire_dst();
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            mul_tiles_bcast_scalar(in0_cb, in1_scalar_cb, i, 0, i);
        }
        cb_pop_front(in0_cb, dst_tiles);
        cb_reserve_back(in0_cb, dst_tiles);
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            pack_tile(i, in0_cb);
        }
        cb_push_back(in0_cb, dst_tiles);
        release_dst();
    }
}

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

template <uint32_t M>
void matmul_reduce(uint32_t in1_cb, const uint32_t& out_cb) {
    // precondition: in0_cb has M*K produced
    // precondition: in1_cb has K*1 produced
    // postcondition: out_cb has M*1 produced

    constexpr uint32_t N = 1;  // Result of reduce is Mx1
    constexpr uint32_t in0_block_w = N;

    // Set dimensions directly
    mm_block_init_short(
        out_cb,
        in1_cb,
        0 /* transpose */,
        N /* ct_dim = 1 col */,
        M /* rt_dim = all rows */,
        in0_block_w /* kt_dim = 1 element from in0 per dot product */);

    reconfig_data_format(in1_cb, out_cb);
    cb_wait_front(in1_cb, N);
    cb_wait_front(out_cb, M);

    tile_regs_acquire();

    uint32_t dst_index = 0;
    uint32_t in0_index = 0;
    uint32_t in1_index = 0;

    matmul_block(
        out_cb,
        in1_cb,
        in0_index,
        in1_index,
        dst_index,
        0, /* col offset */
        N, /* width */
        M, /* height */
        in0_block_w);

    tile_regs_commit();
    cb_pop_front(out_cb, M);

    tile_regs_wait();
    for (uint32_t i = 0; i < M; ++i) {
        pack_tile(i, out_cb);
    }

    tile_regs_release();
    cb_push_back(out_cb, M);
}

template <uint32_t scale_fp32>
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
        exp_tile<EXP_APPROX_MODE, false, true, true>(0, static_cast<int>(VectorMode::C), scale_bf16);
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        release_dst();
    }
}

void copy_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
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
    cb_pop_front(in_cb, num_tiles);
}

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
    const bool& transpose) {
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
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                pack_tile(i, out_cb);
            }
            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
    }
    cb_pop_front(in1_cb, K * N);
    cb_push_back(out_cb, output_num_tiles);
}

/******************************************************************************
 *                   Flash Decode Functions                                    *
 ******************************************************************************/

/**
 * Flash attention computation loop
 *
 * Template Parameters:
 * @tparam St - Total sequence length in tiles
 * @tparam DHt - Head dimension in tiles
 * @tparam Sq_chunk_t - Query chunk size in tiles
 * @tparam Sk_chunk_t - Key chunk size in tiles
 * @tparam qk_in0_block_w - QK matmul block width
 * @tparam qk_subblock_w - QK matmul subblock width
 * @tparam qk_subblock_h - QK matmul subblock height
 * @tparam qk_in0_num_subblocks - QK input0 subblocks
 * @tparam qk_in1_num_subblocks - QK input1 subblocks
 * @tparam qk_num_blocks - QK number of blocks
 * @tparam out_in0_block_w - Output matmul block width
 * @tparam out_subblock_w - Output matmul subblock width
 * @tparam out_subblock_h - Output matmul subblock height
 * @tparam out_in0_num_subblocks - Output input0 subblocks
 * @tparam out_in1_num_subblocks - Output input1 subblocks
 * @tparam out_num_blocks - Output number of blocks
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
 * @param qk_chunk_tiles - Number of QK chunk tiles
 * @param out_chunk_tiles - Number of output chunk tiles
 */
template <
    // Compile-time dimension parameters
    uint32_t St,
    uint32_t DHt,
    uint32_t Sq_chunk_t,
    uint32_t out_chunk_tiles,
    // QK matmul block parameters
    uint32_t qk_in0_block_w,
    uint32_t qk_num_blocks,
    // Output matmul block parameters
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t out_in0_num_subblocks,
    uint32_t out_in1_num_subblocks,
    // Attention parameters
    bool is_causal,
    bool use_attention_mask,
    // Circular buffer indices
    uint32_t cb_q_in,
    uint32_t cb_k_in,
    uint32_t cb_v_in,
    uint32_t cb_mask_in,
    uint32_t cb_scale_in,
    uint32_t cb_identity_scale_in,
    uint32_t cb_qk_im,
    uint32_t cb_out_im,
    uint32_t cb_out_accumulate_im,
    uint32_t cb_cur_max,
    uint32_t cb_prev_max,
    uint32_t cb_cur_sum,
    uint32_t cb_prev_sum,
    uint32_t cb_exp_max_diff,
    uint32_t cb_out_o,
    uint32_t cb_out_m,
    uint32_t cb_out_l,
    uint32_t scale_fp32>
void flash_attention_loop(
    // Runtime parameters
    uint32_t k_chunk_start,
    uint32_t k_chunk_end,
    uint32_t Sk_chunk_t,
    uint32_t qk_subblock_h,
    uint32_t qk_subblock_w,
    uint32_t qk_in0_num_subblocks,
    uint32_t qk_in1_num_subblocks,
    uint32_t out_in0_block_w,
    uint32_t out_num_blocks,
    uint32_t qk_chunk_tiles,
    bool do_reduce,
    bool apply_mask_at_last_chunk  // for causal mode, optionally apply mask at the last chunk
) {
    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
        /* QK = Q_CHUNK @ K_CHUNK */
        cb_matmul_blocks(
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
         * Typically, scores are multiplied by a scalar here, but an optimization was employed
         * where the scaling is fused into exp both in exp(x - max) and exp(prev_max - cur_max).
         * This gives us scaling for free on the performance-critical exp(x - max) computation.
         */

        if constexpr (is_causal) {
            // For decode, we only apply mask at the last chunk for causal mode
            if (k_chunk == k_chunk_end - 1 && apply_mask_at_last_chunk) {
                /* QK += MASK */
                reconfig_data_format(cb_qk_im, cb_mask_in);
                add_block_inplace<false>(cb_qk_im, cb_mask_in, qk_chunk_tiles);
            }
        } else {
            if constexpr (use_attention_mask) {
                reconfig_data_format(cb_qk_im, cb_mask_in);
                add_block_inplace<true>(cb_qk_im, cb_mask_in, qk_chunk_tiles);
            }
        }

        /**
         * reduce_c can perform both reduce_max and eltwise max with previous result.
         * if do_eltwise_max:
         *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
         * else:
         *  cur_max = max(qk, dim=-1)
         */
        reduce_c<
            PoolType::MAX,
            ReduceDim::REDUCE_ROW,
            cb_qk_im,
            cb_identity_scale_in,
            Sq_chunk_t,
            cb_cur_max,
            cb_prev_max>(Sk_chunk_t, k_chunk > k_chunk_start);

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
        sub_exp_block_bcast_cols_inplace_reduce<cb_qk_im, Sq_chunk_t, scale_fp32>(cb_cur_max, cb_cur_sum, Sk_chunk_t);
        cb_wait_front(cb_qk_im, qk_chunk_tiles);

        /* OUT_IM = QK @ V_CHUNK */
        cb_matmul_blocks(
            cb_qk_im,
            cb_v_in,
            cb_out_im,
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
        reconfig_data_format_srca(cb_out_im);
        cb_pop_front(cb_qk_im, qk_chunk_tiles);

        /* OUT_ACC += OUT_IM */
        if (k_chunk == k_chunk_start) {
            copy_block(cb_out_im, cb_out_accumulate_im, out_chunk_tiles);
        } else {
            /* cb_exp_max_diff = torch.exp(cb_prev_max - cb_cur_max) */
            sub_exp_block<scale_fp32>(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
            cb_pop_front(cb_prev_max, Sq_chunk_t);

            /* cb_out_accumulate_im *= cb_exp_max_diff */
            mul_block_bcast_cols<Sq_chunk_t, DHt>(cb_out_accumulate_im, cb_exp_max_diff, cb_out_im, true);

            /* cb_cur_sum += cb_prev_sum */
            add_block_inplace<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
        }

        if (k_chunk < k_chunk_end - 1 || do_reduce) {
            // Set cb_prev_sum and cb_prev_max
            copy_block(cb_cur_max, cb_prev_max, Sq_chunk_t);
            copy_block(cb_cur_sum, cb_prev_sum, Sq_chunk_t);

        } else {
            // Write o, m, l into cb_out
            copy_block(cb_out_accumulate_im, cb_out_o, out_chunk_tiles);
            copy_block(cb_cur_max, cb_out_m, Sq_chunk_t);
            copy_block(cb_cur_sum, cb_out_l, Sq_chunk_t);
        }
    }
}
