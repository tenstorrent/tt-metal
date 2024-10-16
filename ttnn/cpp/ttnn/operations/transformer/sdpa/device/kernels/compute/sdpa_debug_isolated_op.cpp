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

#include "debug/assert.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

namespace NAMESPACE {
template<uint32_t in0, uint32_t in1, uint32_t num_tiles>
void max_block_inplace() {
    // inputs come in full, outputs go out full
    copy_tile_to_dst_init_short(in0);
    max_tile_init();
    unpack_reconfig_data_format(in0, in1);

    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    cb_wait_front(in0, num_tiles);
    cb_wait_front(in1, num_tiles);
    unpack_reconfig_data_format(in0, in1);
    pack_reconfig_data_format(in0);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst(tt::DstMode::Half);
        copy_tile(in0, 0, dst_reg_0);
        copy_tile(in1, i, dst_reg_1);
        cb_pop_front(in0, 1);
        cb_reserve_back(in0, 1);
        max_tile(dst_reg_0, dst_reg_1);
        pack_tile(dst_reg_0, in0);
        cb_push_back(in0, 1);
        release_dst(tt::DstMode::Half);
    }
}

template<PoolType pool_type, ReduceDim reduce_dim, uint32_t in0_cb, uint32_t scale_cb, uint32_t out_cb, uint32_t rows, uint32_t cols>
void reduce_c() {
    // Precondition: in0_cb has rows*cols produced. in0_cb has tiles in row-major order
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has rows free
    // Postcondition: in0_cb has rows*cols produced
    // Precondition: scale_cb has 1 produced
    // Postcondition: out_cb has rows produced
    unpack_reconfig_data_format(in0_cb, scale_cb);

    reduce_init_delta<false, pool_type, reduce_dim>(in0_cb, scale_cb, out_cb);

    const uint32_t num_tiles = rows * cols;
    cb_wait_front(scale_cb, 1);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, rows);

    unpack_reconfig_data_format(in0_cb, scale_cb);
    pack_reconfig_data_format(out_cb);

    constexpr uint32_t reduce_dst_idx = 0;

    for (uint32_t i = 0; i < rows; i++) {
        acquire_dst(tt::DstMode::Half);
        for (uint32_t j = 0; j < cols; j++) {
            reduce_tile<pool_type, reduce_dim>(in0_cb, scale_cb, i*cols+j, 0, reduce_dst_idx);
        }

        cb_reserve_back(out_cb, 1);
        pack_tile(reduce_dst_idx, out_cb);
        cb_push_back(out_cb, 1);
        release_dst(tt::DstMode::Half);
    }

   reduce_revert_delta<reduce_dim>(out_cb);
}

template<bool reconfig_data_format=true>
void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Postcondition: in_cb has num_tiles produced

    copy_tile_to_dst_init_short(in_cb);
    recip_tile_init();

    if constexpr(reconfig_data_format) {
        unpack_reconfig_data_format_srca(in_cb);
        pack_reconfig_data_format(in_cb);
    }

    cb_wait_front(in_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst(tt::DstMode::Half);
        copy_tile(in_cb, 0, 0);
        cb_pop_front(in_cb, 1);
        recip_tile(0);
        cb_reserve_back(in_cb, 1);
        pack_tile(0, in_cb);
        cb_push_back(in_cb, 1);
        release_dst(tt::DstMode::Half);
    }
}

template<uint32_t in0_cb, uint32_t in1_cb, uint32_t rows, uint32_t cols>
void sub_exp_block_bcast_cols_inplace() {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows produced
    unpack_reconfig_data_format(in0_cb, in1_cb);

    sub_bcast_cols_init_short(in0_cb, in1_cb);
    exp_tile_init<true>();
    cb_wait_front(in0_cb, rows*cols);
    cb_wait_front(in1_cb, rows);

    unpack_reconfig_data_format(in0_cb, in1_cb);
    pack_reconfig_data_format(in0_cb);

    constexpr uint32_t dst_tiles = SUB_EXP_GRANULARITY;
    constexpr uint32_t granularity = cols >> LOG2_SUB_EXP_GRANULARITY;
    for (uint32_t i = 0; i < rows; ++i) {
        for(uint32_t u = 0; u < granularity; u++) {
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

template<bool reconfig_data_format=true>
void mul_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t rows, uint32_t cols) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows consumed

    uint32_t num_tiles = rows * cols;
    mul_bcast_cols_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, rows);

    if constexpr(reconfig_data_format) {
        unpack_reconfig_data_format(in0_cb, in1_cb);
        pack_reconfig_data_format(in0_cb);
    }

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            acquire_dst(tt::DstMode::Half);
            mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
            cb_pop_front(in0_cb, 1);
            cb_reserve_back(in0_cb, 1);
            pack_tile(0, in0_cb);
            cb_push_back(in0_cb, 1);
            release_dst(tt::DstMode::Half);
        }
    }
    cb_pop_front(in1_cb, rows);
}

template<uint32_t in0_cb, uint32_t in1_scalar_cb, uint32_t num_tiles>
void mul_block_bcast_scalar_inplace() {
    // Precondition: in0_cb has num_tiles produced
    // Precondition: in1_scalar_cb has 1 produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_scalar_cb has 1 produced

    constexpr uint32_t dst_tiles = MUL_BCAST_GRANULARITY;
    constexpr uint32_t granularity = num_tiles >> LOG2_MUL_BCAST_GRANULARITY;
    unpack_reconfig_data_format(in0_cb, in1_scalar_cb);
    mul_tiles_bcast_scalar_init_short();
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_scalar_cb, 1);

    unpack_reconfig_data_format(in0_cb, in1_scalar_cb);
    pack_reconfig_data_format(in0_cb);

    for (uint32_t g = 0; g < granularity; ++g) {
        acquire_dst(tt::DstMode::Half);
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            mul_tiles_bcast_scalar(in0_cb, in1_scalar_cb, i, 0, i);
        }
        cb_pop_front(in0_cb, dst_tiles);
        cb_reserve_back(in0_cb, dst_tiles);
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            pack_tile(i, in0_cb);
        }
        cb_push_back(in0_cb, dst_tiles);
        release_dst(tt::DstMode::Half);
    }
}

template<bool reconfig_data_format=true>
void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    add_tiles_init();
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);

    if constexpr(reconfig_data_format) {
        unpack_reconfig_data_format(in0_cb, in1_cb);
        pack_reconfig_data_format(in0_cb);
    }

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst(tt::DstMode::Half);
        add_tiles(in0_cb, in1_cb, 0, i, 0);
        cb_pop_front(in0_cb, 1);
        cb_reserve_back(in0_cb, 1);
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, 1);
        release_dst(tt::DstMode::Half);
    }

    cb_pop_front(in1_cb, num_tiles);
}

template<bool reconfig_data_format=true>
void mul_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles produced

    mul_tiles_init();
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);

    if constexpr(reconfig_data_format) {
        unpack_reconfig_data_format(in0_cb, in1_cb);
        pack_reconfig_data_format(in0_cb);
    }

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst(tt::DstMode::Half);
        mul_tiles(in0_cb, in1_cb, 0, i, 0);
        cb_pop_front(in0_cb, 1);
        cb_reserve_back(in0_cb, 1);
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, 1);
        release_dst(tt::DstMode::Half);
    }
}

void mul_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    mul_tiles_init();
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst(tt::DstMode::Half);
        mul_tiles(in0_cb, in1_cb, i, i, 0);
        pack_tile(0, out_cb, i);
        release_dst(tt::DstMode::Half);
    }
    cb_push_back(out_cb, num_tiles);

    cb_pop_front(in0_cb, num_tiles);
}

void sub_exp_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: out_cb has num_tiles produced
    // Postcondition: in0_cb and in1_cb has num_tiles produced
    unpack_reconfig_data_format(in0_cb, in1_cb);

    sub_tiles_init();
    exp_tile_init<true>();
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

    unpack_reconfig_data_format(in0_cb, in1_cb);
    pack_reconfig_data_format(out_cb);

    for (uint32_t i = 0; i < num_tiles; i++) {

        acquire_dst(tt::DstMode::Half);

        sub_tiles(in0_cb, in1_cb, i, i, 0);

        exp_tile<true>(0);

        pack_tile(0, out_cb);

        cb_push_back(out_cb, 1);
        release_dst(tt::DstMode::Half);
    }
}

template<bool reconfig_data_format=true, bool pop_in_cb=true>
void copy_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Precondition: out_cb has num_tiles free
    // Postcondition: in_cb has num_tiles consumed
    // Postcondition: out_cb has num_tiles produced

    copy_tile_to_dst_init_short(in_cb);

    cb_wait_front(in_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

    if constexpr(reconfig_data_format) {
        unpack_reconfig_data_format_srca(in_cb);
        pack_reconfig_data_format(out_cb);
    }

    #pragma GCC unroll 0
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst(tt::DstMode::Half);
        copy_tile(in_cb, i, 0/*dst*/);
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        release_dst(tt::DstMode::Half);
    }

    if constexpr(pop_in_cb) {
        cb_pop_front(in_cb, num_tiles);
    }
}

void matmul_blocks(const uint32_t& in0_cb, const uint32_t& in1_cb, const uint32_t& out_cb, const uint32_t& M, const uint32_t& N, const uint32_t& K, const uint32_t& num_blocks, const uint32_t& in0_num_subblocks, const uint32_t& in1_num_subblocks,
                    const uint32_t& in0_block_w, const uint32_t& subblock_h, const uint32_t& subblock_w, const bool& transpose) {
    // precondition: in0_cb has M*K produced
    // preconditino: in1_cb has K*N produced
    // postcondition: in0_cb is full, in1_cb is empty
    // postcondition: out_cb has M*N produced
    // unpack_reconfig_data_format(in0_cb, in1_cb);

    mm_block_init_short(in0_cb, in1_cb, transpose /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    cb_wait_front(in1_cb, K * N);

    uint32_t output_num_tiles = M * N;
    uint32_t out_subblock_num_tiles = subblock_h * subblock_w;
    uint32_t in0_index_offset = 0;

    unpack_reconfig_data_format(in1_cb, in0_cb);
    pack_reconfig_data_format(out_cb);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        uint32_t in1_index_offset = 0;
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                matmul_block(in0_cb, in1_cb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, in0_block_w);
                in0_index++;
                in1_index += N;

            }
            tile_regs_commit();

            cb_reserve_back(out_cb, out_subblock_num_tiles);
            tile_regs_wait();
            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                pack_tile(i, out_cb);
            }
            tile_regs_release();
            cb_push_back(out_cb, out_subblock_num_tiles);
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
    }
    cb_pop_front(in1_cb, K * N);
}

template<bool reconfig_data_format=true>
void copy_block_fill(uint32_t in_cb, uint32_t out_cb, uint32_t in_num_tiles, uint32_t out_num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Precondition: out_cb has num_tiles free
    // Postcondition: in_cb has num_tiles consumed
    // Postcondition: out_cb has num_tiles produced

    ASSERT(out_num_tiles%in_num_tiles == 0);
    uint32_t num_copies = out_num_tiles/in_num_tiles;

    copy_tile_to_dst_init_short(in_cb);

    cb_wait_front(in_cb, in_num_tiles);
    cb_reserve_back(out_cb, out_num_tiles);

    if constexpr(reconfig_data_format) {
        unpack_reconfig_data_format_srca(in_cb);
        pack_reconfig_data_format(out_cb);
    }

    #pragma GCC unroll 0
    for (uint32_t i = 0; i < in_num_tiles; i++) {
        for (uint32_t row = 0; row < num_copies; row++) {
            acquire_dst(tt::DstMode::Half);
            copy_tile(in_cb, i, 0/*dst*/);
            pack_tile(0, out_cb);
            cb_push_back(out_cb, 1);
            release_dst(tt::DstMode::Half);
        }
    }
    cb_pop_front(in_cb, in_num_tiles);
}

void MAIN {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t St = get_compile_time_arg_val(3);
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(7);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(8);

    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(9);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(10);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(11);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(12);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(13);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(14);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(15);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(16);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(17);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(18);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(19);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(20);

    constexpr uint32_t num_cores = get_compile_time_arg_val(21);

    const uint32_t core_id    = get_arg_val<uint32_t>(0);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(1);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(2);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(3);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(4);
    const uint32_t local_q_start = get_arg_val<uint32_t>(5);
    const uint32_t local_q_end = get_arg_val<uint32_t>(6);


    const uint32_t q_chunks_per_core = local_q_end - local_q_start;


    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * DHt;

    constexpr uint32_t cb_q_in = tt::CB::c_in0;
    constexpr uint32_t cb_k_in = tt::CB::c_in1;
    constexpr uint32_t cb_v_in = tt::CB::c_in2;
    constexpr uint32_t cb_mask_in = tt::CB::c_in3;
    constexpr uint32_t cb_scale_in = tt::CB::c_in4;
    constexpr uint32_t cb_identity_scale_in = tt::CB::c_in5;

    constexpr uint32_t cb_debug = tt::CB::c_out6;
    constexpr uint32_t cb_debug2 = tt::CB::c_out7;
    constexpr uint32_t cb_debug3 = tt::CB::c_out5;

    constexpr uint32_t cb_qk_im = tt::CB::c_intermed0;
    constexpr uint32_t cb_out_im = tt::CB::c_intermed1;
    constexpr uint32_t cb_out_accumulate_im = tt::CB::c_intermed2;
    constexpr uint32_t cb_cur_max = tt::CB::c_intermed3;
    constexpr uint32_t cb_prev_max = tt::CB::c_intermed4;
    constexpr uint32_t cb_cur_sum = tt::CB::c_intermed5;
    constexpr uint32_t cb_prev_sum = tt::CB::c_intermed6;
    constexpr uint32_t cb_exp_max_diff = tt::CB::c_intermed7;

    constexpr uint32_t cb_out = tt::CB::c_out0;

    ASSERT((k_chunk_tiles == q_chunk_tiles) && (k_chunk_tiles == out_chunk_tiles));

    DPRINT_MATH(DPRINT << "k_chunk_tiles: " << k_chunk_tiles << ENDL());
    DPRINT_MATH(DPRINT << "q_chunk_tiles: " << q_chunk_tiles << ENDL());
    DPRINT_MATH(DPRINT << "out_chunk_tiles: " << out_chunk_tiles << ENDL());

    mm_init();

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                uint32_t q_chunk;
                #if defined BALANCED_Q_PARALLEL
                uint32_t q_chunk_div_2 = q_chunks_per_core / 2;
                if (q_iter < q_chunk_div_2) { // bottom half
                    q_chunk = local_q_start + q_iter;
                } else {
                    uint32_t back_q_iter = q_iter - q_chunk_div_2; // Back half should start at 0
                    q_chunk = q_num_chunks - 1 - (local_q_start + back_q_iter);
                }
                #else
                q_chunk = local_q_start + q_iter;
                #endif

                // Get Q chunk
                const uint32_t q_low_idx = q_chunk * Sq_chunk_t; // This is the sequence index of the first tile of this chunk
                const uint32_t q_high_idx = q_low_idx + Sq_chunk_t;
                // DPRINT_MATH(DPRINT << "chkpt 1" << ENDL());

                cb_wait_front(cb_q_in, q_chunk_tiles);
                // copy_block(cb_q_in, cb_out_accumulate_im, q_chunk_tiles);

                // loop while k_low < q_high
                for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
                    const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
                    const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;
                    // DPRINT_MATH(DPRINT << "chkpt 1.1" << ENDL());

                    { // test mul block, add block, and max block in place
                        /* Q_CHUNK = Q_CHUNK * K_CHUNK */
                        // copy_block(cb_v_in, cb_out_im, k_chunk_tiles);
                        // max_block_inplace<cb_out_accumulate_im, cb_out_im, out_chunk_tiles>();
                        // cb_pop_front(cb_out_im, out_chunk_tiles);
                        { // do mul_block_inplace with cb_debug2 in fp32, replacing cb_out_im
                        // unpack_reconfig_data_format_srca(cb_v_in);
                        // pack_reconfig_data_format(cb_out_accumulate_im);
                        // copy_block<false>(cb_v_in, cb_debug2, k_chunk_tiles);
                        // unpack_reconfig_data_format(cb_out_accumulate_im, cb_out_accumulate_im);
                        // pack_reconfig_data_format(cb_out_accumulate_im);
                        // add_block_inplace<false>(cb_out_accumulate_im, cb_debug2, out_chunk_tiles);
                        }
                        // // not used
                        // cb_pop_front(cb_k_in, k_chunk_tiles);
                        // cb_pop_front(cb_mask_in, qk_chunk_tiles);
                    }

                    { // test reduce c sum
                        // // DPRINT_MATH(DPRINT << "chkpt 1.2" << ENDL());
                        // copy_block(cb_v_in, cb_out_im, k_chunk_tiles);
                        // reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_out_im, cb_identity_scale_in, cb_cur_sum, Sq_chunk_t, DHt>();
                        // // DPRINT_MATH(DPRINT << "chkpt 1.3" << ENDL());
                        // cb_pop_front(cb_out_im, out_chunk_tiles);
                        // copy_block_fill(cb_cur_sum, cb_out_im, Sq_chunk_t, out_chunk_tiles);
                        // add_block_inplace(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                        // // DPRINT_MATH(DPRINT << "chkpt 1.4" << ENDL());

                        // // not used
                        // cb_pop_front(cb_k_in, k_chunk_tiles);
                        // cb_pop_front(cb_mask_in, qk_chunk_tiles);
                    }

                    { // test reduce c sum
                        // // DPRINT_MATH(DPRINT << "chkpt 1.2" << ENDL());
                        // copy_block(cb_v_in, cb_out_im, k_chunk_tiles);
                        // reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_out_im, cb_identity_scale_in, cb_cur_sum, Sq_chunk_t, DHt>();
                        // // DPRINT_MATH(DPRINT << "chkpt 1.3" << ENDL());
                        // cb_pop_front(cb_out_im, out_chunk_tiles);
                        // copy_block_fill(cb_cur_sum, cb_out_im, Sq_chunk_t, out_chunk_tiles);
                        // add_block_inplace(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                        // // DPRINT_MATH(DPRINT << "chkpt 1.4" << ENDL());

                        // // not used
                        // cb_pop_front(cb_k_in, k_chunk_tiles);
                        // cb_pop_front(cb_mask_in, qk_chunk_tiles);
                    }

                    { // test matmul blocks
                        // // DPRINT_MATH(DPRINT << "chkpt 1.2" << ENDL());
                        // matmul_blocks(cb_q_in, cb_k_in, cb_qk_im, Sq_chunk_t, Sk_chunk_t, DHt, qk_num_blocks, qk_in0_num_subblocks, qk_in1_num_subblocks, qk_in0_block_w, qk_subblock_h, qk_subblock_w, true /*transpose*/);
                        // matmul_blocks(cb_qk_im, cb_v_in, cb_out_im, Sq_chunk_t, DHt, Sk_chunk_t, out_num_blocks, out_in0_num_subblocks, out_in1_num_subblocks, out_in0_block_w, out_subblock_h, out_subblock_w, false /*transpose*/);
                        // cb_pop_front(cb_qk_im, qk_chunk_tiles);

                        // if (k_chunk == 0){
                        //     copy_block(cb_out_im, cb_out_accumulate_im, out_chunk_tiles);
                        // } else {
                        //     add_block_inplace(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                        // }

                        // // not used
                        // cb_pop_front(cb_mask_in, qk_chunk_tiles);
                    }

                    { // test sub_exp_block
                        // sub_exp_block(cb_q_in, cb_v_in, cb_out_im, out_chunk_tiles);
                        // cb_pop_front(cb_v_in, out_chunk_tiles);

                        // if (k_chunk == 0){
                        //     copy_block(cb_out_im, cb_out_accumulate_im, out_chunk_tiles);
                        // } else {
                        //     add_block_inplace(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                        // }

                        // // not used
                        // cb_pop_front(cb_k_in, k_chunk_tiles);
                        // cb_pop_front(cb_mask_in, qk_chunk_tiles);
                    }

                    { // test recip_block
                        // recip_block_inplace(cb_out_accumulate_im, out_chunk_tiles);

                        // // not used
                        // cb_pop_front(cb_k_in, k_chunk_tiles);
                        // cb_pop_front(cb_v_in, k_chunk_tiles);
                        // cb_pop_front(cb_mask_in, qk_chunk_tiles);
                    }

                    { // test mul_block_bcast_scalar_inplace
                        copy_block<true, false>(cb_q_in, cb_out_im, k_chunk_tiles);
                        mul_block_bcast_scalar_inplace<cb_out_im, cb_scale_in, out_chunk_tiles>();

                        if (k_chunk == 0){
                            copy_block(cb_out_im, cb_out_accumulate_im, out_chunk_tiles);
                        } else {
                            add_block_inplace(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                        }

                        // not used
                        cb_pop_front(cb_k_in, k_chunk_tiles);
                        cb_pop_front(cb_v_in, k_chunk_tiles);
                        cb_pop_front(cb_mask_in, qk_chunk_tiles);
                    }

                    { // test sub_exp_block_bcast_cols_inplace
                        // copy_block<true, false>(cb_q_in, cb_out_im, k_chunk_tiles);
                        // copy_block<true, false>(cb_v_in, cb_cur_sum, Sq_chunk_t);
                        // cb_pop_front(cb_v_in, k_chunk_tiles);
                        // sub_exp_block_bcast_cols_inplace<cb_out_im, cb_cur_sum, Sq_chunk_t, DHt>();
                        // cb_pop_front(cb_cur_sum, Sq_chunk_t);

                        // if (k_chunk == 0){
                        //     copy_block(cb_out_im, cb_out_accumulate_im, out_chunk_tiles);
                        // } else {
                        //     add_block_inplace(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                        // }

                        // // not used
                        // cb_pop_front(cb_k_in, k_chunk_tiles);
                        // cb_pop_front(cb_mask_in, qk_chunk_tiles);
                    }

                    { // test mul_block_bcast_cols_inplace
                        // copy_block<true, false>(cb_q_in, cb_out_im, k_chunk_tiles);
                        // copy_block<true, false>(cb_v_in, cb_cur_sum, Sq_chunk_t);
                        // cb_pop_front(cb_v_in, k_chunk_tiles);
                        // mul_block_bcast_cols_inplace(cb_out_im, cb_cur_sum, Sq_chunk_t, DHt);

                        // if (k_chunk == 0){
                        //     copy_block(cb_out_im, cb_out_accumulate_im, out_chunk_tiles);
                        // } else {
                        //     add_block_inplace(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                        // }

                        // // not used
                        // cb_pop_front(cb_k_in, k_chunk_tiles);
                        // cb_pop_front(cb_mask_in, qk_chunk_tiles);
                    }
                }
                cb_pop_front(cb_q_in, q_chunk_tiles);
                // pack_reconfig_data_format(cb_out);
                copy_block(cb_out_accumulate_im, cb_out, out_chunk_tiles);
            }
        }
    }
    // DPRINT_MATH(DPRINT << "C done" << ENDL());
}
}
