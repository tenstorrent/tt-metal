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


namespace NAMESPACE {
template<uint32_t in0, uint32_t in1, uint32_t num_tiles>
void max_block_inplace() {
    // inputs come in full, outputs go out full
    copy_tile_to_dst_init_short(in0);
    max_tile_init();

    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    cb_wait_front(in0, num_tiles);
    cb_wait_front(in1, num_tiles);
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

    reduce_init_delta<false, pool_type, reduce_dim>(in0_cb, scale_cb, out_cb);

    const uint32_t num_tiles = rows * cols;
    cb_wait_front(scale_cb, 1);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, rows);

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
   UNPACK(tensix_sync()); // Workaround for issue #9370
}

void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Postcondition: in_cb has num_tiles produced
    copy_tile_to_dst_init_short(in_cb);
    recip_tile_init();

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

    sub_bcast_cols_init_short(in0_cb, in1_cb);
    exp_tile_init<true>();
    cb_wait_front(in0_cb, rows*cols);
    cb_wait_front(in1_cb, rows);


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

void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    add_tiles_init();
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
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

void mul_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles produced

    mul_tiles_init();
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
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

void sub_exp_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: out_cb has num_tiles produced
    // Postcondition: in0_cb and in1_cb has num_tiles produced

    sub_tiles_init();
    exp_tile_init<true>();
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

    for (uint32_t i = 0; i < num_tiles; i++) {

        acquire_dst(tt::DstMode::Half);

        sub_tiles(in0_cb, in1_cb, i, i, 0);

        exp_tile<true>(0);

        pack_tile(0, out_cb);

        cb_push_back(out_cb, 1);
        release_dst(tt::DstMode::Half);
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

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst(tt::DstMode::Half);
        copy_tile(in_cb, i, 0/*dst*/);
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        release_dst(tt::DstMode::Half);
    }
    cb_pop_front(in_cb, num_tiles);
}

void matmul_blocks(const uint32_t& in0_cb, const uint32_t& in1_cb, const uint32_t& out_cb, const uint32_t& M, const uint32_t& N, const uint32_t& K, const uint32_t& num_blocks, const uint32_t& in0_num_subblocks, const uint32_t& in1_num_subblocks,
                    const uint32_t& in0_block_w, const uint32_t& subblock_h, const uint32_t& subblock_w, const bool& transpose) {
    // precondition: in0_cb has M*K produced
    // preconditino: in1_cb has K*N produced
    // postcondition: in0_cb is full, in1_cb is empty
    // postcondition: out_cb has M*N produced

    mm_block_init_short(in0_cb, in1_cb, transpose /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    unpack_reconfig_data_format(in1_cb, in0_cb);
    cb_wait_front(in1_cb, K * N);

    uint32_t output_num_tiles = M * N;
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

void MAIN {
    constexpr uint32_t NKH = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);
    constexpr uint32_t _unused1 = get_compile_time_arg_val(2);
    constexpr uint32_t St = get_compile_time_arg_val(3);
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(7);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(8); // num chunks in valid_seq_len

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

    constexpr uint32_t cb_qk_im = tt::CB::c_intermed0;
    constexpr uint32_t cb_out_im = tt::CB::c_intermed1;
    constexpr uint32_t cb_out_accumulate_im = tt::CB::c_intermed2;
    constexpr uint32_t cb_cur_max = tt::CB::c_intermed3;
    constexpr uint32_t cb_prev_max = tt::CB::c_intermed4;
    constexpr uint32_t cb_cur_sum = tt::CB::c_intermed5;
    constexpr uint32_t cb_prev_sum = tt::CB::c_intermed6;
    constexpr uint32_t cb_exp_max_diff = tt::CB::c_intermed7;

    constexpr uint32_t cb_out = tt::CB::c_out0;


    mm_init();

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                cb_wait_front(cb_q_in, q_chunk_tiles);

                // loop while k_low < q_high
                for (uint32_t k_chunk = 0; k_chunk < k_num_chunks; ++k_chunk) {

                    /* QK = Q_CHUNK @ K_CHUNK */
                    unpack_reconfig_data_format(cb_q_in, cb_k_in); // DEBUG
                    pack_reconfig_data_format(cb_qk_im);
                    matmul_blocks(cb_q_in, cb_k_in, cb_qk_im, Sq_chunk_t, Sk_chunk_t, DHt, qk_num_blocks, qk_in0_num_subblocks, qk_in1_num_subblocks, qk_in0_block_w, qk_subblock_h, qk_subblock_w, true /*transpose*/);

                    /* QK *= SCALE */
                    mul_block_bcast_scalar_inplace<cb_qk_im, cb_scale_in, qk_chunk_tiles>();

                    // Finding the diagonal is harder now that q_chunk_size and k_chunk_size can differ
                    // Q-range = [q_low, q_high)
                    // K-range = [k_low, k_high)
                    // does_overlap = not (q_low >= k_high or k_low >= q_high)
                    // Due to loop bounds, we should never have k_low >= q_high. Can simplify this conditional check
                    /* QK += MASK */
                    unpack_reconfig_data_format(cb_qk_im, cb_mask_in);
                    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);

                    unpack_reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                    pack_reconfig_data_format(cb_cur_max);
                    reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, cb_cur_max, Sq_chunk_t, Sk_chunk_t>();

                    if (k_chunk > 0) {
                        unpack_reconfig_data_format(cb_cur_max, cb_prev_max);
                        max_block_inplace<cb_cur_max, cb_prev_max, Sq_chunk_t>();
                    }

                    /* QK -= cb_cur_max */
                    /* QK = exp(QK)*/
                    unpack_reconfig_data_format(cb_qk_im, cb_cur_max);
                    pack_reconfig_data_format(cb_qk_im);
                    sub_exp_block_bcast_cols_inplace<cb_qk_im, cb_cur_max, Sq_chunk_t, Sk_chunk_t>();

                    /* cb_cur_sum = sum(cb_qk_im, dim=-1) */
                    unpack_reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                    pack_reconfig_data_format(cb_cur_sum);
                    reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, cb_cur_sum, Sq_chunk_t, Sk_chunk_t>();

                    /* OUT_IM = QK @ V_CHUNK */
                    unpack_reconfig_data_format(cb_qk_im, cb_v_in); // DEBUG
                    pack_reconfig_data_format(cb_out_im);
                    matmul_blocks(cb_qk_im, cb_v_in, cb_out_im, Sq_chunk_t, DHt, Sk_chunk_t, out_num_blocks, out_in0_num_subblocks, out_in1_num_subblocks, out_in0_block_w, out_subblock_h, out_subblock_w, false /*transpose*/);
                    unpack_reconfig_data_format_srca(cb_out_im);
                    cb_pop_front(cb_qk_im, qk_chunk_tiles);

                    /* OUT_ACC += OUT_IM */
                    if (k_chunk == 0) {
                        unpack_reconfig_data_format_srca(cb_out_im);
                        pack_reconfig_data_format(cb_out_accumulate_im);
                        copy_block(cb_out_im, cb_out_accumulate_im, out_chunk_tiles);
                    } else {
                        unpack_reconfig_data_format(cb_prev_max, cb_cur_max); // DEBUG
                        pack_reconfig_data_format(cb_exp_max_diff);
                        /* cb_exp_max_diff = torch.exp(cb_prev_max - cb_cur_max) */
                        sub_exp_block(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
                        cb_pop_front(cb_prev_max, Sq_chunk_t);

                        /* cb_prev_sum *= cb_exp_max_diff */
                        mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);

                        /* cb_out_accumulate_im *= cb_exp_max_diff */
                        unpack_reconfig_data_format(cb_out_accumulate_im, cb_exp_max_diff); // DEBUG
                        pack_reconfig_data_format(cb_out_accumulate_im);
                        mul_block_bcast_cols_inplace(cb_out_accumulate_im, cb_exp_max_diff, Sq_chunk_t, DHt);

                        /* cb_cur_sum += cb_prev_sum */
                        unpack_reconfig_data_format(cb_cur_sum, cb_prev_sum); // DEBUG
                        pack_reconfig_data_format(cb_cur_sum);
                        add_block_inplace(cb_cur_sum, cb_prev_sum, Sq_chunk_t);

                        /* cb_out_accumulate_im += cb_out_im */
                        unpack_reconfig_data_format(cb_out_accumulate_im, cb_out_im); // DEBUG
                        pack_reconfig_data_format(cb_out_accumulate_im);
                        add_block_inplace(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                    }

                    // Set cb_prev_sum and cb_prev_max
                    unpack_reconfig_data_format(cb_cur_max, cb_cur_max); // DEBUG
                    pack_reconfig_data_format(cb_prev_max);
                    copy_block(cb_cur_max, cb_prev_max, Sq_chunk_t);
                    copy_block(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
                }


                /* cb_cur_sum = 1.0 / cb_cur_sum */
                cb_push_back(cb_cur_sum, Sq_chunk_t);

                unpack_reconfig_data_format(cb_cur_sum, cb_cur_sum); // DEBUG
                pack_reconfig_data_format(cb_cur_sum);
                recip_block_inplace(cb_cur_sum, Sq_chunk_t);

                /* cb_out_accumulate_im *= cb_cur_sum */
                unpack_reconfig_data_format(cb_out_accumulate_im, cb_cur_sum); // DEBUG
                pack_reconfig_data_format(cb_out_accumulate_im);
                mul_block_bcast_cols_inplace(cb_out_accumulate_im, cb_cur_sum, Sq_chunk_t, DHt);
                pack_reconfig_data_format(cb_out);
                copy_block(cb_out_accumulate_im, cb_out, out_chunk_tiles);

                cb_pop_front(cb_q_in, q_chunk_tiles);
                // free up cb_prev_max after K chunks
                cb_pop_front(cb_prev_max, Sq_chunk_t);
                cb_pop_front(cb_prev_sum, Sq_chunk_t);
            }
        }
    }


}
}
