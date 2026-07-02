// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/reduce.h"
#include "api/compute/transpose.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "ckernel_sfpu.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

#define DEBUG_PRINT 0
using namespace ckernel;

void generate_rand_tile(const uint32_t cb_id, const uint32_t seed) {
    init_sfpu(cb_id, cb_id);

    DataflowBuffer cb_obj(cb_id);

    uint32_t rand_scale = 0;
    const float one_f = 1.0f;
    std::memcpy(&rand_scale, &one_f, sizeof(uint32_t));  // Alternative to std::bit_cast
    uint32_t rand_from = 0;

    if (seed != 0) {
        rand_tile_init(seed);
    }
    cb_obj.reserve_back(1);

    tile_regs_acquire();
    rand_tile(0, rand_from, rand_scale);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_id, 0);
    tile_regs_release();

    cb_obj.push_back(1);
}

template <uint32_t in0_cb, uint32_t in1_cb, uint32_t rows, uint32_t cols>
void sub_exp_block_bcast_cols_inplace() {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows produced

    DataflowBuffer in0_cb_obj(in0_cb);
    DataflowBuffer in1_cb_obj(in1_cb);

    sub_bcast_cols_init_short(in0_cb, in1_cb);
    exp_tile_init<true>();
    in0_cb_obj.wait_front(rows * cols);
    in1_cb_obj.wait_front(rows);

    constexpr uint32_t dst_tiles = 1;       // SUB_EXP_GRANULARITY;
    constexpr uint32_t granularity = cols;  // #>> LOG2_SUB_EXP_GRANULARITY;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; u++) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);
                exp_tile<true>(j);
            }
            tile_regs_commit();

            in0_cb_obj.pop_front(dst_tiles);
            in0_cb_obj.reserve_back(dst_tiles);

            tile_regs_wait();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile(j, in0_cb);
            }
            tile_regs_release();

            in0_cb_obj.push_back(dst_tiles);
        }
    }
}

void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles produced
    DataflowBuffer in0_cb_obj(in0_cb);
    DataflowBuffer in1_cb_obj(in1_cb);

    reconfig_data_format(in0_cb, in1_cb);
    add_tiles_init(in0_cb, in1_cb);
    in0_cb_obj.wait_front(num_tiles);
    in1_cb_obj.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        add_tiles(in0_cb, in1_cb, 0, i, 0);
        tile_regs_commit();

        in0_cb_obj.pop_front(1);
        in0_cb_obj.reserve_back(1);

        tile_regs_wait();
        pack_reconfig_data_format(in0_cb);
        pack_tile(0, in0_cb);
        tile_regs_release();

        in0_cb_obj.push_back(1);
    }
}

void mul_block_bcast_cols(
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t rows, uint32_t cols) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows consumed

    DataflowBuffer in0_cb_obj(in0_cb);
    DataflowBuffer in1_cb_obj(in1_cb);
    DataflowBuffer out_cb_obj(out_cb);

    uint32_t num_tiles = rows * cols;
    mul_bcast_cols_init_short(in0_cb, in1_cb);
    in0_cb_obj.wait_front(num_tiles);
    in1_cb_obj.wait_front(rows);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
            tile_regs_commit();

            in0_cb_obj.pop_front(1);
            out_cb_obj.reserve_back(1);

            tile_regs_wait();
            pack_tile(0, out_cb);
            tile_regs_release();

            out_cb_obj.push_back(1);
        }
    }
    in1_cb_obj.pop_front(rows);
}

void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Postcondition: in_cb has num_tiles produced
    DataflowBuffer in_cb_obj(in_cb);

    copy_tile_to_dst_init_short(in_cb);
    recip_tile_init();

    in_cb_obj.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        copy_tile(in_cb, 0, 0);
        recip_tile(0);
        tile_regs_commit();

        in_cb_obj.pop_front(1);
        in_cb_obj.reserve_back(1);

        tile_regs_wait();
        pack_tile(0, in_cb);
        tile_regs_release();

        in_cb_obj.push_back(1);
    }
}

template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_cb,
    uint32_t scale_cb,
    uint32_t out_cb,
    uint32_t rows,
    uint32_t cols>
void reduce_c() {
    // Postcondition: in0_cb has rows*cols produced (WaitUpfrontNoPop — tiles not consumed)
    // Postcondition: out_cb has rows produced
    compute_kernel_lib::reduce<
        pool_type,
        reduce_dim,
        in0_cb,
        scale_cb,
        out_cb,
        compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        compute_kernel_lib::ReduceInputBlockShape::of(rows, cols));
    UNPACK(tensix_sync());  // Workaround for issue #9370
}

template <
    uint32_t Ht,
    uint32_t Wt,
    uint32_t K,
    uint32_t logWt,
    uint32_t logk,
    uint32_t input_cb_index,
    uint32_t index_cb_index,
    uint32_t input_transposed_cb_index,
    uint32_t index_transposed_cb_index,
    uint32_t values_cb_index,
    uint32_t output_ind_cb_index,
    uint32_t tile_width,
    bool first_call>
void top_k() {
    // dest indices for where to unpack the tiles for the llk
    // the input goes in index 0,1 and the index goes in index 2,3
    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;
    ckernel::topk_tile_init();

    DataflowBuffer input_cb(input_cb_index);
    DataflowBuffer index_cb(index_cb_index);
    DataflowBuffer input_transposed_cb(input_transposed_cb_index);
    DataflowBuffer index_transposed_cb(index_transposed_cb_index);
    DataflowBuffer values_cb(values_cb_index);
    DataflowBuffer output_ind_cb(output_ind_cb_index);

    if (first_call) {
        transpose_init(input_cb_index);
    }
    for (uint32_t ht = 0; ht < Ht; ++ht) {
        bool ascending = false;
        input_transposed_cb.reserve_back(Wt);
        index_transposed_cb.reserve_back(Wt);

        // streaming in input and index tiles to transpose and bitonic local sort them, two tiles at a time
        for (uint32_t wt = 0; wt < Wt; wt += 2) {
            // local sort into k groups
            input_cb.wait_front(2);
            index_cb.wait_front(2);

            tile_regs_acquire();
            reconfig_data_format_srca(input_cb_index);
            transpose_init(input_cb_index);
            transpose_tile(input_cb_index, 0, 0);
            transpose_tile(input_cb_index, 1, 1);

            reconfig_data_format_srca(index_cb_index);
            transpose_init(index_cb_index);
            transpose_tile(index_cb_index, 0, 2);
            transpose_tile(index_cb_index, 1, 3);

            // llk_topk_sort -> inplace
            ckernel::topk_local_sort(0, (int)ascending, logk - 1);

            tile_regs_commit();

            input_cb.pop_front(2);
            index_cb.pop_front(2);

            tile_regs_wait();
            // pack value tiles into cb_intermed0
            pack_reconfig_data_format(input_transposed_cb_index);
            pack_tile(0, input_transposed_cb_index);
            pack_tile(1, input_transposed_cb_index);

            // pack index tiles into cb_intermed1
            pack_reconfig_data_format(index_transposed_cb_index);
            pack_tile(2, index_transposed_cb_index);
            pack_tile(3, index_transposed_cb_index);
            tile_regs_release();
        }

        input_transposed_cb.push_back(Wt);
        index_transposed_cb.push_back(Wt);

        // iterative divide and conquer on pairs of tiles (bitonic topk merge and rebuild)
        // first iteration we compare 0th and 1st tile, then 2nd and 3rd, etc. We get the sorted top 32 values in each
        // pair. second iteration we compare 0th and 2nd tile, then 4th and 6th, etc. logWt iteration we compare 0th and
        // Wt/2 tile single buffer as we can pack tiles back in-place
        for (uint32_t m_iter = 0; m_iter < logWt; ++m_iter) {
            bool a = false;
            input_transposed_cb.wait_front(Wt);
            index_transposed_cb.wait_front(Wt);

            for (uint32_t left_ind = 0; left_ind < Wt - (1 << m_iter); left_ind += 2 << m_iter) {
                uint32_t right_ind = left_ind + (1 << m_iter);
                tile_regs_acquire();

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
                ckernel::topk_rebuild(0, (uint32_t)a, m_iter, K, logk, true);

                tile_regs_commit();
                tile_regs_wait();
                // pack value tiles in-place in the single-buffered cb_intermed0, we only need the upper 32 values for
                // topk, which was in input_dest_start
                pack_reconfig_data_format(input_transposed_cb_index);
                pack_tile<true>(input_dest_start, input_transposed_cb_index, left_ind);

                // pack index tiles in-place in the single-buffered cb_intermed1, we only need the upper 32 values for
                // topk, which was in index_dest_start
                pack_reconfig_data_format(index_transposed_cb_index);
                pack_tile<true>(index_dest_start, index_transposed_cb_index, left_ind);
                tile_regs_release();
                a = !a;
            }

            input_transposed_cb.reserve_back(Wt);
            index_transposed_cb.reserve_back(Wt);

            input_transposed_cb.pop_front(Wt);
            index_transposed_cb.pop_front(Wt);

            input_transposed_cb.push_back(Wt);
            index_transposed_cb.push_back(Wt);
        }

        constexpr uint32_t Kt = K % tile_width == 0 ? K / tile_width : K / tile_width + 1;

        // transpose value tiles and pack into output buffer
        reconfig_data_format_srca(input_transposed_cb_index);
        transpose_init(input_transposed_cb_index);
        pack_reconfig_data_format(input_transposed_cb_index);
        input_transposed_cb.wait_front(Wt);
        for (uint32_t i = 0; i < Kt; ++i) {
            tile_regs_acquire();
            transpose_tile(input_transposed_cb_index, i, 0);
            tile_regs_commit();

            values_cb.reserve_back(1);

            tile_regs_wait();
            pack_tile(0, values_cb_index);
            tile_regs_release();

            values_cb.push_back(1);
        }
        input_transposed_cb.pop_front(Wt);

        // transpose index tiles and pack into output buffer
        reconfig_data_format_srca(index_transposed_cb_index);
        transpose_init(index_transposed_cb_index);
        pack_reconfig_data_format(index_transposed_cb_index);
        index_transposed_cb.wait_front(Wt);
        for (uint32_t i = 0; i < Kt; ++i) {
            tile_regs_acquire();
            transpose_tile(index_transposed_cb_index, i, 0);
            tile_regs_commit();

            output_ind_cb.reserve_back(1);

            tile_regs_wait();
            pack_tile(0, output_ind_cb_index);
            tile_regs_release();

            output_ind_cb.push_back(1);
        }
        index_transposed_cb.pop_front(Wt);
    }
    sfpu::_init_sfpu_config_reg();
}

template <uint32_t in0_cb, uint32_t in1_scalar_cb, uint32_t num_tiles>
void mul_block_bcast_scalar_inplace() {
    // Precondition: in0_cb has num_tiles produced
    // Precondition: in1_scalar_cb has 1 produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_scalar_cb has 1 produced

    DataflowBuffer in0_cb_obj(in0_cb);
    DataflowBuffer in1_scalar_cb_obj(in1_scalar_cb);

    uint32_t dst_tiles = num_tiles;
    uint32_t granularity = 1;

    reconfig_data_format(in0_cb, in1_scalar_cb);
    mul_tiles_bcast_scalar_init_short(in0_cb, in1_scalar_cb);
    in0_cb_obj.wait_front(num_tiles);
    in1_scalar_cb_obj.wait_front(1);

    for (uint32_t g = 0; g < granularity; ++g) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            mul_tiles_bcast_scalar(in0_cb, in1_scalar_cb, i, 0, i);
        }
        tile_regs_commit();

        in0_cb_obj.pop_front(dst_tiles);
        in0_cb_obj.reserve_back(dst_tiles);

        tile_regs_wait();
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            pack_tile(i, in0_cb);
        }
        tile_regs_release();

        in0_cb_obj.push_back(dst_tiles);
    }
}

void kernel_main() {
    constexpr uint32_t input_values_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t input_transposed_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t index_transposed_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t values_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(5);

    constexpr uint32_t topk_mask_cb_index = get_compile_time_arg_val(6);
    constexpr uint32_t scaler_max_cb_index = get_compile_time_arg_val(7);
    constexpr uint32_t scaler_sum_cb_index = get_compile_time_arg_val(8);
    constexpr uint32_t cb_cur_max = get_compile_time_arg_val(9);
    constexpr uint32_t cb_cur_sum = get_compile_time_arg_val(10);
    constexpr uint32_t Ht = get_compile_time_arg_val(11);
    constexpr uint32_t Wt = get_compile_time_arg_val(12);
    constexpr uint32_t logWt = get_compile_time_arg_val(13);
    constexpr uint32_t rand_tile_index = get_compile_time_arg_val(14);
    constexpr uint32_t seed = get_compile_time_arg_val(15);
    constexpr uint32_t cb_local_vals = get_compile_time_arg_val(16);
    constexpr uint32_t temp_cb_index = get_compile_time_arg_val(17);
    constexpr uint32_t tile_width = get_compile_time_arg_val(18);
    generate_rand_tile(rand_tile_index, seed);

    const uint32_t nearest32_K = 32;
    const uint32_t logk = 5;  // log(32)

    // top-k
    compute_kernel_hw_startup(input_values_cb_index, index_cb_index, input_transposed_cb_index);
    top_k<
        Ht,
        Wt,
        nearest32_K,
        logWt,
        logk,
        input_values_cb_index,
        index_cb_index,
        input_transposed_cb_index,
        index_transposed_cb_index,
        values_cb_index,
        output_ind_cb_index,
        tile_width,
        true>();
    constexpr uint32_t Kt = nearest32_K / tile_width;

    // scale temperature

    // mask out all values except the top-k
    DataflowBuffer topk_mask_cb(topk_mask_cb_index);
    topk_mask_cb.wait_front(Kt);
    add_block_inplace(values_cb_index, topk_mask_cb_index, Ht * Kt);
    mul_block_bcast_scalar_inplace<values_cb_index, temp_cb_index, Ht * Kt>();
    // softmax
    reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, values_cb_index, scaler_max_cb_index, cb_cur_max, Ht, Kt>();

    sub_exp_block_bcast_cols_inplace<values_cb_index, cb_cur_max, Ht, Kt>();
    reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, values_cb_index, scaler_sum_cb_index, cb_cur_sum, Ht, Kt>();
    recip_block_inplace(cb_cur_sum, Ht);
    mul_block_bcast_cols(values_cb_index, cb_cur_sum, cb_local_vals, Ht, Kt);
}
