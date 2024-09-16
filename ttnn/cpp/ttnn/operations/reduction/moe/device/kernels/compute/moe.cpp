// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#define REDUCE_OP (PoolType::SUM)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/unpack.h"
#include "compute_kernel_api/math.h"
#include "compute_kernel_api/pack.h"
#include "debug/dprint.h"
#include "ckernel_sfpu.h"
using namespace ckernel;
// topk llk needs a global variable atm
// this can only be removed once that's fixed
int32_t topk_replay_init = 0;

namespace NAMESPACE {
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


    constexpr uint32_t dst_tiles = 1; //SUB_EXP_GRANULARITY;
    constexpr uint32_t granularity = cols; // #>> LOG2_SUB_EXP_GRANULARITY;
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

void add_block_bcast_rows_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t rows, uint32_t cols, bool first_call) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    uint32_t num_tiles = rows * cols;
    if (first_call) {
        init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::ROW>(in0_cb, in1_cb);
    }
    else{
        unpack_reconfig_data_format(in0_cb, in1_cb);
        math_reconfig_data_format(in0_cb, in1_cb);
        add_bcast_rows_init_short(in0_cb, in1_cb);
    }
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, cols);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            acquire_dst();
            add_tiles_bcast_rows(in0_cb, in1_cb, 0, j, 0);
            cb_pop_front(in0_cb, 1);
            cb_reserve_back(in0_cb, 1);
            pack_reconfig_data_format(in0_cb);
            pack_tile(0, in0_cb);
            cb_push_back(in0_cb, 1);
            release_dst();
        }
    }
    cb_pop_front(in1_cb, cols);
}
void mul_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles produced
    unpack_reconfig_data_format(in0_cb, in1_cb);
    math_reconfig_data_format(in0_cb, in1_cb);
    mul_tiles_init();
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        mul_tiles(in0_cb, in1_cb, 0, i, 0);
        cb_pop_front(in0_cb, 1);
        cb_reserve_back(in0_cb, 1);
        pack_reconfig_data_format(in0_cb);
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, 1);
        release_dst();
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

void eqz_block_inplace(uint32_t in0_cb, uint32_t num_tiles) {
    // Precondition: in0_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced

    unpack_reconfig_data_format_srca(in0_cb);
    math_reconfig_data_format_srca(in0_cb);
    eqz_tile_init();
    cb_wait_front(in0_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        eqz_tile(0);
        cb_pop_front(in0_cb, 1);
        cb_reserve_back(in0_cb, 1);
        pack_reconfig_data_format(in0_cb);
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, 1);
        release_dst();
    }
}

void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Postcondition: in_cb has num_tiles produced
    copy_tile_to_dst_init_short(in_cb);
    recip_tile_init();

    cb_wait_front(in_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst();
        copy_tile(in_cb, 0, 0);
        cb_pop_front(in_cb, 1);
        recip_tile(0);
        cb_reserve_back(in_cb, 1);
        pack_tile(0, in_cb);
        cb_push_back(in_cb, 1);
        release_dst();
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
    math_reconfig_data_format(in0_cb, scale_cb);
    reduce_init_delta<false, pool_type, reduce_dim>(in0_cb, scale_cb, out_cb);

    const uint32_t num_tiles = rows * cols;
    cb_wait_front(scale_cb, 1);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, rows);

    constexpr uint32_t reduce_dst_idx = 0;

    for (uint32_t i = 0; i < rows; i++) {
        acquire_dst();
        for (uint32_t j = 0; j < cols; j++) {
            reduce_tile<pool_type, reduce_dim>(in0_cb, scale_cb, i*cols+j, 0, reduce_dst_idx);
        }

        cb_reserve_back(out_cb, 1);
        pack_reconfig_data_format(out_cb);
        pack_tile(reduce_dst_idx, out_cb);
        cb_push_back(out_cb, 1);
        release_dst();
    }

   reduce_revert_delta<reduce_dim>(out_cb);
   UNPACK(tensix_sync()); // Workaround for issue #9370
}

template<uint32_t Ht, uint32_t Wt, uint32_t K, uint32_t logWt, uint32_t logk, uint32_t input_cb_index, uint32_t index_cb_index, uint32_t input_transposed_cb_index, uint32_t index_transposed_cb_index, uint32_t values_cb_index, uint32_t output_ind_cb_index, bool first_call>
void top_k() {
    // dest indices for where to unpack the tiles for the llk
    // the input goes in index 0,1 and the index goes in index 2,3
    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;
    ckernel::topk_tile_init();

    if (first_call){
        transpose_wh_init(input_cb_index, input_transposed_cb_index);
    }
    for(uint32_t ht = 0; ht < Ht; ++ht) {
        bool ascending = false;
        cb_reserve_back(input_transposed_cb_index, Wt);
        cb_reserve_back(index_transposed_cb_index, Wt);

        // streaming in input and index tiles to transpose and bitonic local sort them, two tiles at a time
        for (uint32_t wt = 0; wt < Wt; wt+=2) {
            acquire_dst();
            // local sort into k groups
            cb_wait_front(input_cb_index, 2);
            cb_wait_front(index_cb_index, 2);

            unpack_reconfig_data_format_srca(input_cb_index);
            math_reconfig_data_format_srca(input_cb_index);
            transpose_wh_init_short(input_cb_index);
            transpose_wh_tile(input_cb_index, 0, 0);
            transpose_wh_tile(input_cb_index, 1, 1);

            unpack_reconfig_data_format_srca(index_cb_index);
            math_reconfig_data_format_srca(index_cb_index);
            transpose_wh_init_short(index_cb_index);
            transpose_wh_tile(index_cb_index, 0, 2);
            transpose_wh_tile(index_cb_index, 1, 3);

            // llk_topk_sort -> inplace
            ckernel::topk_local_sort(0, (int) ascending, logk - 1);

            // pack value tiles into cb_intermed0
            pack_reconfig_data_format(input_transposed_cb_index);
            pack_tile(0, input_transposed_cb_index);
            pack_tile(1, input_transposed_cb_index);

            // pack index tiles into cb_intermed1
            pack_reconfig_data_format(index_transposed_cb_index);
            pack_tile(2, index_transposed_cb_index);
            pack_tile(3, index_transposed_cb_index);

            cb_pop_front(input_cb_index, 2);
            cb_pop_front(index_cb_index, 2);
            release_dst();
        }

        cb_push_back(input_transposed_cb_index, Wt);
        cb_push_back(index_transposed_cb_index, Wt);

        // iterative divide and conquer on pairs of tiles (bitonic topk merge and rebuild)
        // first iteration we compare 0th and 1st tile, then 2nd and 3rd, etc. We get the sorted top 32 values in each pair.
        // second iteration we compare 0th and 2nd tile, then 4th and 6th, etc.
        // logWt iteration we compare 0th and Wt/2 tile
        // single buffer as we can pack tiles back in-place
        for (uint32_t m_iter = 0; m_iter < logWt; ++m_iter) {
            bool a = false;
            cb_wait_front(input_transposed_cb_index, Wt);
            cb_wait_front(index_transposed_cb_index, Wt);

            for (uint32_t left_ind = 0; left_ind < Wt - (1 << m_iter); left_ind += 2 << m_iter) {
                uint32_t right_ind = left_ind + (1 << m_iter);
                acquire_dst();

                copy_tile_to_dst_init_short_with_dt(index_transposed_cb_index, input_transposed_cb_index);
                copy_tile(input_transposed_cb_index, left_ind, input_dest_start);
                copy_tile(input_transposed_cb_index, right_ind, input_dest_end);

                // unpack indices into dest
                copy_tile_to_dst_init_short_with_dt(input_transposed_cb_index, index_transposed_cb_index);
                copy_tile(index_transposed_cb_index, left_ind, index_dest_start);
                copy_tile(index_transposed_cb_index, right_ind, index_dest_end);

                // merge values - move larger 32 values into 0th dest and lower 32 values into 1st dest
                ckernel::topk_merge(0, m_iter, K);
                // sort within the larger 32 values
                ckernel::topk_rebuild(0, (uint32_t) a, m_iter, K, logk, true);


                // pack value tiles in-place in the single-buffered cb_intermed0, we only need the upper 32 values for topk, which was in input_dest_start
                pack_reconfig_data_format(input_transposed_cb_index);
                pack_tile<true>(input_dest_start, input_transposed_cb_index, left_ind);

                // pack index tiles in-place in the single-buffered cb_intermed1, we only need the upper 32 values for topk, which was in index_dest_start
                pack_reconfig_data_format(index_transposed_cb_index);
                pack_tile<true>(index_dest_start, index_transposed_cb_index, left_ind);
                release_dst();
                a = !a;
            }

            cb_reserve_back(input_transposed_cb_index, Wt);
            cb_reserve_back(index_transposed_cb_index, Wt);

            cb_pop_front(input_transposed_cb_index, Wt);
            cb_pop_front(index_transposed_cb_index, Wt);

            cb_push_back(input_transposed_cb_index, Wt);
            cb_push_back(index_transposed_cb_index, Wt);
        }

        constexpr uint32_t Kt =  K % TILE_WIDTH == 0 ? K/TILE_WIDTH : K/TILE_WIDTH + 1;

        // transpose value tiles and pack into output buffer
        unpack_reconfig_data_format_srca(input_transposed_cb_index);
        math_reconfig_data_format_srca(input_transposed_cb_index);
        transpose_wh_init_short(input_transposed_cb_index);
        pack_reconfig_data_format(input_transposed_cb_index);
        cb_wait_front(input_transposed_cb_index, Kt);
        for (uint32_t i = 0; i < Kt; ++i) {
            acquire_dst();
            cb_reserve_back(values_cb_index, 1);
            transpose_wh_tile(input_transposed_cb_index, i, 0);
            pack_tile(0, values_cb_index);
            cb_push_back(values_cb_index, 1);
            release_dst();
        }
        cb_wait_front(input_transposed_cb_index, Wt);
        cb_pop_front(input_transposed_cb_index, Wt);

        // transpose index tiles and pack into output buffer
        unpack_reconfig_data_format_srca(index_transposed_cb_index);
        math_reconfig_data_format_srca(index_transposed_cb_index);
        transpose_wh_init_short(index_transposed_cb_index);
        pack_reconfig_data_format(index_transposed_cb_index);
        cb_wait_front(index_transposed_cb_index, Kt);
        for (uint32_t i = 0; i < Kt; ++i) {
            acquire_dst();
            cb_reserve_back(output_ind_cb_index, 1);
            transpose_wh_tile(index_transposed_cb_index, i, 0);
            pack_tile(0, output_ind_cb_index);
            cb_push_back(output_ind_cb_index, 1);
            release_dst();
        }
        cb_wait_front(index_transposed_cb_index, Wt);
        cb_pop_front(index_transposed_cb_index, Wt);
    }
    //sfpu::_init_sfpu_config_reg();
}

void MAIN {
    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t topk_mask_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t expert_mask_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t scale_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t index_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t input_transposed_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t index_transposed_cb_index = get_compile_time_arg_val(6);
    constexpr uint32_t values_cb_index = get_compile_time_arg_val(7);
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(8);
    constexpr uint32_t out_cb_index = get_compile_time_arg_val(9);

    constexpr uint32_t Ht = get_compile_time_arg_val(10);
    constexpr uint32_t Wt = get_compile_time_arg_val(11);
    constexpr uint32_t K = get_compile_time_arg_val(12);
    constexpr uint32_t logk = get_compile_time_arg_val(13);
    constexpr uint32_t logWt = get_compile_time_arg_val(14);

    constexpr uint32_t cb_cur_max = get_compile_time_arg_val(15);
    constexpr uint32_t cb_cur_sum = get_compile_time_arg_val(16);

    constexpr uint32_t Kt =  K % 32 == 0 ? K/32 : K/32 + 1;

    // mask out invalid experts
    // TODO: fix the bug that makes this give bad results
    //add_block_bcast_rows_inplace(input_cb_index, expert_mask_cb_index, Ht,Wt, true);

    // top-k
    top_k<Ht, Wt, K, logWt, logk, input_cb_index, index_cb_index, input_transposed_cb_index, index_transposed_cb_index, values_cb_index, output_ind_cb_index, true>();

    // mask out all experts except the top-k
    add_block_bcast_rows_inplace(values_cb_index, topk_mask_cb_index, Ht,Kt, false);
    eqz_block_inplace(output_ind_cb_index, Ht*Kt);

    //softmax
    reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, values_cb_index, scale_cb_index, cb_cur_max, Ht, Kt>();
    sub_exp_block_bcast_cols_inplace<values_cb_index, cb_cur_max, Ht, Kt>();
    reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, values_cb_index, scale_cb_index, cb_cur_sum, Ht, Kt>();
    recip_block_inplace(cb_cur_sum, Ht);
    mul_block_bcast_cols_inplace(values_cb_index, cb_cur_sum, Ht, Kt);

    // select 0th expert
    mul_block_inplace(values_cb_index, output_ind_cb_index, Ht*Kt);

    //final sum
    reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, values_cb_index, scale_cb_index, out_cb_index, Ht, Kt>();

}
}
