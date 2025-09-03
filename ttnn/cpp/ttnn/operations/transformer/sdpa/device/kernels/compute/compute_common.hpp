// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/softplus.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"

template <uint32_t num_tiles>
void max_block_inplace(uint32_t in0, uint32_t in1) {
    // inputs come in full, outputs go out full
    copy_tile_to_dst_init_short(in0);
    max_tile_init();

    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    cb_wait_front(in0, num_tiles);
    cb_wait_front(in1, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(in0, i, dst_reg_0);
        copy_tile(in1, i, dst_reg_1);
        max_tile(dst_reg_0, dst_reg_1, static_cast<int>(VectorMode::C));
        pack_tile(dst_reg_0, in0);
        release_dst();
    }
    cb_pop_front(in0, num_tiles);
    cb_reserve_back(in0, num_tiles);
    cb_push_back(in0, num_tiles);
}

template <PoolType pool_type, ReduceDim reduce_dim, uint32_t in0_cb, uint32_t scale_cb, uint32_t rows, uint32_t cols>
void reduce_c(uint32_t out_cb, uint32_t prev_cb, bool do_eltwise_max = false) {
    // Precondition: in0_cb has rows*cols produced. in0_cb has tiles in row-major order
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has rows free
    // Postcondition: in0_cb has rows*cols produced
    // Precondition: scale_cb has 1 produced
    // Postcondition: out_cb has rows produced

    constexpr uint32_t num_tiles = rows * cols;
    cb_wait_front(scale_cb, 1);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, rows);

    max_tile_init();
    constexpr uint32_t reduce_dst_idx = 0;
    constexpr uint32_t prev_max_dst_idx = 1;

    for (uint32_t i = 0; i < rows; i++) {
        acquire_dst();
        reduce_init<pool_type, reduce_dim>(in0_cb, scale_cb, out_cb);
        for (uint32_t j = 0; j < cols; j++) {
            reduce_tile<pool_type, reduce_dim>(in0_cb, scale_cb, i * cols + j, 0, reduce_dst_idx);
        }
        reduce_uninit();
        if (do_eltwise_max) {
            copy_tile_to_dst_init_short(prev_cb);
            copy_tile(prev_cb, i, prev_max_dst_idx);
            max_tile(reduce_dst_idx, prev_max_dst_idx, static_cast<int>(VectorMode::C));
        }

        pack_tile(reduce_dst_idx, out_cb);
        release_dst();
    }

    cb_push_back(out_cb, rows);
}

void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Postcondition: in_cb has num_tiles produced
    copy_tile_to_dst_init_short(in_cb);
    recip_tile_init();
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(in_cb);

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

template <uint32_t in0_cb, uint32_t rows, uint32_t cols, uint32_t scale_fp32>
void sub_exp_block_bcast_cols_inplace(uint32_t in1_cb, uint32_t reduce_cb) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows produced
    sub_bcast_cols_init_short(in0_cb, in1_cb);

    exp_tile_init<true, true, scale_fp32>();
    cb_wait_front(in0_cb, rows * cols);
    cb_wait_front(in1_cb, rows);
    cb_reserve_back(reduce_cb, rows);

    constexpr uint32_t dst_tiles = SUB_EXP_GRANULARITY;
    constexpr uint32_t granularity = cols >> LOG2_SUB_EXP_GRANULARITY;
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

template <uint32_t rows, uint32_t cols>
void mul_block_bcast_cols(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, bool pack_accumulate = false) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Precondition: out_cb has rows*cols produced
    // Postcondition: in0_cb empty
    // Postcondition: in1_cb empty
    // Postcondition: out_cb has rows*cols produced

    constexpr uint32_t num_tiles = rows * cols;
    constexpr uint32_t dst_tiles = DHT_GRANULARITY;
    constexpr uint32_t granularity = cols >> LOG2_DHT_GRANULARITY;
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

template <uint32_t rows, uint32_t cols>
void mul_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows consumed

    constexpr uint32_t num_tiles = rows * cols;
    constexpr uint32_t dst_tiles = DHT_GRANULARITY;
    constexpr uint32_t granularity = cols >> LOG2_DHT_GRANULARITY;
    mul_bcast_cols_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, rows);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; ++u) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                mul_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);
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
    cb_pop_front(in1_cb, rows);
}

template <uint32_t in1_scalar_cb, uint32_t num_tiles>
void mul_block_bcast_scalar_inplace(uint32_t in0_cb) {
    // Precondition: in0_cb has num_tiles produced
    // Precondition: in1_scalar_cb has 1 produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_scalar_cb has 1 produced

    constexpr uint32_t dst_tiles = STATS_GRANULARITY;
    constexpr uint32_t granularity = num_tiles >> LOG2_STATS_GRANULARITY;
    reconfig_data_format(in0_cb, in1_scalar_cb);
    mul_tiles_bcast_scalar_init_short(in0_cb, in1_scalar_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_scalar_cb, 1);
    uint32_t in0_index = 0;
    for (uint32_t g = 0; g < granularity; ++g) {
        acquire_dst();
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            mul_tiles_bcast_scalar(in0_cb, in1_scalar_cb, in0_index, 0, i);
            in0_index++;
        }
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            pack_tile(i, in0_cb);
        }
        release_dst();
    }
    cb_pop_front(in0_cb, num_tiles);
    cb_reserve_back(in0_cb, num_tiles);
    cb_push_back(in0_cb, num_tiles);
}

void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    add_tiles_init(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        add_tiles(in0_cb, in1_cb, i, i, 0);
        pack_tile(0, in0_cb);
        release_dst();
    }

    cb_pop_front(in1_cb, num_tiles);
    cb_pop_front(in0_cb, num_tiles);
    cb_reserve_back(in0_cb, num_tiles);
    cb_push_back(in0_cb, num_tiles);
}

void mul_tiles_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    /**
     * Given in0_cb and in1_cb, multiply each tile of in0_cb by the corresponding tile of in1_cb
     * and bcast cols of in1_cb.
     */
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles produced

    mul_bcast_cols_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
        cb_pop_front(in0_cb, 1);
        cb_reserve_back(in0_cb, 1);
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, 1);
        release_dst();
    }
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

    // Convert scale_fp32 to bf16 scale
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    for (uint32_t i = 0; i < num_tiles; i++) {
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

void log_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    copy_tile_to_dst_init_short(in_cb);
    log_tile_init();
    cb_wait_front(in_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        copy_tile(in_cb, i, 0 /*dst*/);
        log_tile(0);
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        release_dst();
    }
}

void sigmoid_sub(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    // out_cb = sigmoid(in0_cb - in1_cb)

    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    sub_tiles_init(in0_cb, in1_cb);
    sigmoid_tile_init();

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        sub_tiles(in0_cb, in1_cb, i, i, 0);
        sigmoid_tile(0);
        pack_tile(0, out_cb);
        release_dst();
    }
    cb_push_back(out_cb, num_tiles);
}

void logsigmoid_sub(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    // out_cb = logsigmoid(in0_cb - in1_cb)

    // Implemented as softplus. logsigmoid(x) = -softplus(-x)

    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    sub_tiles_init(in0_cb, in1_cb);

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        // Remove negate by swapping inputs
        sub_tiles(in1_cb, in0_cb, i, i, 0);
        // negative_tile_init();
        // negative_tile(0);
        softplus_tile_init();
        softplus_tile(0, 0x3F800000, 0x3F800000, 0x41A00000);  // beta, beta_reciprocal, threshold
        negative_tile_init();
        negative_tile(0);
        // sigmoid_tile_init();
        // sigmoid_tile(0);
        // log_tile_init();
        // log_tile(0);
        pack_tile(0, out_cb);
        release_dst();
    }
    cb_push_back(out_cb, num_tiles);
}
void sub_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    // out_cb = in0_cb - in1_cb

    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    sub_tiles_init(in0_cb, in1_cb);

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        sub_tiles(in0_cb, in1_cb, i, i, 0);
        pack_tile(0, out_cb);
        release_dst();
    }
    cb_push_back(out_cb, num_tiles);
}

void matmul_blocks(
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

    uint32_t output_num_tiles = M * N;
    uint32_t out_subblock_num_tiles = subblock_h * subblock_w;
    uint32_t in0_index_offset = 0;

    reconfig_data_format(in1_cb, in0_cb);
    cb_wait_front(in1_cb, K * N);
    cb_reserve_back(out_cb, output_num_tiles);

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
            uint32_t dst_idx = 0;
            uint32_t out_col_offset = in1_subblock * subblock_w;
            for (uint32_t r = 0; r < subblock_h; r++) {
                uint32_t out_row_offset = (r + subblock_h * in0_subblock) * N;
                for (uint32_t c = 0; c < subblock_w; c++) {
                    pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset + c);
                    dst_idx++;
                }
            }
            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
    }
    cb_pop_front(in1_cb, K * N);
    cb_push_back(out_cb, output_num_tiles);
}

template <uint32_t M>
void matmul_reduce(uint32_t in1_cb, const uint32_t& out_cb) {
    // precondition: in0_cb has M*K produced
    // preconditino: in1_cb has K*N produced
    // postcondition: in0_cb is full, in1_cb is empty
    // postcondition: out_cb has M*N produced

    constexpr uint32_t N = 1;  // Result of reduce is 1 column
    constexpr uint32_t in0_block_w = N;
    constexpr uint32_t subblock_w = N;
    // Reuse the Sq_chunk_t granularity chosen for sub_exp_block
    constexpr uint32_t subblock_h = STATS_GRANULARITY;
    constexpr uint32_t in0_num_subblocks = M >> LOG2_STATS_GRANULARITY;

    /**
     * Use matmul on Mx1 input to reduce rows within tile to produce Mx1 output.
     */

    mm_block_init_short(
        out_cb, in1_cb, 0 /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    constexpr uint32_t output_num_tiles = M * N;
    constexpr uint32_t out_subblock_num_tiles = subblock_h * subblock_w;

    reconfig_data_format(in1_cb, out_cb);
    cb_wait_front(in1_cb, N);
    cb_wait_front(out_cb, M);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        tile_regs_acquire();

        uint32_t dst_index = 0;
        uint32_t in0_index = 0;
        uint32_t in1_index = 0;

        matmul_block(out_cb, in1_cb, in0_index, in1_index, dst_index, 0, subblock_w, subblock_h, in0_block_w);

        tile_regs_commit();
        cb_pop_front(out_cb, subblock_h);

        tile_regs_wait();
        for (uint32_t i = 0; i < subblock_h; i++) {
            pack_tile(i, out_cb);
        }
        tile_regs_release();
        cb_push_back(out_cb, subblock_h);
    }
}
