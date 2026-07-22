// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/reduce.h"
#include "api/compute/transpose.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/dprint.h"
#include "ckernel_sfpu.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
using namespace ckernel;

template <uint32_t in0_dfb, uint32_t in1_dfb, uint32_t rows, uint32_t cols>
void sub_exp_block_bcast_cols_inplace() {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows produced

    DataflowBuffer in0_dfb_obj(in0_dfb);
    DataflowBuffer in1_dfb_obj(in1_dfb);

    sub_bcast_cols_init_short(in0_dfb, in1_dfb);
    exp_tile_init<true>();
    in0_dfb_obj.wait_front(rows * cols);
    in1_dfb_obj.wait_front(rows);

    constexpr uint32_t dst_tiles = 1;       // SUB_EXP_GRANULARITY;
    constexpr uint32_t granularity = cols;  // #>> LOG2_SUB_EXP_GRANULARITY;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; u++) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sub_tiles_bcast_cols(in0_dfb, in1_dfb, j, i, j);
                exp_tile<true>(j);
            }
            tile_regs_commit();

            in0_dfb_obj.pop_front(dst_tiles);
            in0_dfb_obj.reserve_back(dst_tiles);

            tile_regs_wait();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile(j, in0_dfb);
            }
            tile_regs_release();

            in0_dfb_obj.push_back(dst_tiles);
        }
    }
}

void add_block_bcast_rows_inplace(uint32_t in0_dfb, uint32_t in1_dfb, uint32_t rows, uint32_t cols, bool first_call) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    DataflowBuffer in0_dfb_obj(in0_dfb);
    DataflowBuffer in1_dfb_obj(in1_dfb);

    uint32_t num_tiles = rows * cols;
    if (first_call) {
        init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::ROW>(in0_dfb, in1_dfb, in0_dfb);
    } else {
        reconfig_data_format(in0_dfb, in1_dfb);
        add_bcast_rows_init_short(in0_dfb, in1_dfb);
    }
    in0_dfb_obj.wait_front(num_tiles);
    in1_dfb_obj.wait_front(cols);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            tile_regs_acquire();
            add_tiles_bcast_rows(in0_dfb, in1_dfb, 0, j, 0);
            tile_regs_commit();

            in0_dfb_obj.pop_front(1);
            in0_dfb_obj.reserve_back(1);

            tile_regs_wait();
            pack_reconfig_data_format(in0_dfb);
            pack_tile(0, in0_dfb);
            tile_regs_release();

            in0_dfb_obj.push_back(1);
        }
    }
    in1_dfb_obj.pop_front(cols);
}
void mul_block_inplace(uint32_t in0_dfb, uint32_t in1_dfb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles produced
    DataflowBuffer in0_dfb_obj(in0_dfb);
    DataflowBuffer in1_dfb_obj(in1_dfb);

    reconfig_data_format(in0_dfb, in1_dfb);
    mul_tiles_init(in0_dfb, in1_dfb);
    in0_dfb_obj.wait_front(num_tiles);
    in1_dfb_obj.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        mul_tiles(in0_dfb, in1_dfb, 0, i, 0);
        tile_regs_commit();

        in0_dfb_obj.pop_front(1);
        in0_dfb_obj.reserve_back(1);

        tile_regs_wait();
        pack_reconfig_data_format(in0_dfb);
        pack_tile(0, in0_dfb);
        tile_regs_release();

        in0_dfb_obj.push_back(1);
    }
}
void mul_block_bcast_cols_inplace(uint32_t in0_dfb, uint32_t in1_dfb, uint32_t rows, uint32_t cols) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows consumed

    DataflowBuffer in0_dfb_obj(in0_dfb);
    DataflowBuffer in1_dfb_obj(in1_dfb);

    uint32_t num_tiles = rows * cols;
    mul_bcast_cols_init_short(in0_dfb, in1_dfb);
    in0_dfb_obj.wait_front(num_tiles);
    in1_dfb_obj.wait_front(rows);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(in0_dfb, in1_dfb, 0, i, 0);
            tile_regs_commit();

            in0_dfb_obj.pop_front(1);
            in0_dfb_obj.reserve_back(1);

            tile_regs_wait();
            pack_tile(0, in0_dfb);
            tile_regs_release();

            in0_dfb_obj.push_back(1);
        }
    }
    in1_dfb_obj.pop_front(rows);
}

void eqz_block_inplace(uint32_t in0_dfb, uint32_t num_tiles) {
    // Precondition: in0_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced

    DataflowBuffer in0_dfb_obj(in0_dfb);

    reconfig_data_format_srca(in0_dfb);
    eqz_tile_init();
    copy_tile_to_dst_init_short(in0_dfb);
    in0_dfb_obj.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        copy_tile(in0_dfb, 0, 0);
        eqz_tile(0);
        tile_regs_commit();

        in0_dfb_obj.pop_front(1);
        in0_dfb_obj.reserve_back(1);

        tile_regs_wait();
        pack_reconfig_data_format(in0_dfb);
        pack_tile(0, in0_dfb);
        tile_regs_release();

        in0_dfb_obj.push_back(1);
    }
}

void recip_block_inplace(uint32_t in_dfb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Postcondition: in_cb has num_tiles produced
    DataflowBuffer in_dfb_obj(in_dfb);

    copy_tile_to_dst_init_short(in_dfb);
    recip_tile_init();

    in_dfb_obj.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        copy_tile(in_dfb, 0, 0);
        recip_tile(0);
        tile_regs_commit();

        in_dfb_obj.pop_front(1);
        in_dfb_obj.reserve_back(1);

        tile_regs_wait();
        pack_tile(0, in_dfb);
        tile_regs_release();

        in_dfb_obj.push_back(1);
    }
}

template <PoolType pool_type, ReduceDim reduce_dim, uint32_t in_dfb, uint32_t scale_dfb, uint32_t out_dfb>
void reduce_c(uint32_t rows, uint32_t cols) {
    // Precondition: in_cb has rows*cols produced. in_cb has tiles in row-major order
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has rows free
    // Postcondition: in_cb has rows*cols produced
    // Postcondition: scale_cb has 1 produced
    // Postcondition: out_cb has rows produced

    compute_kernel_lib::reduce<
        pool_type,
        reduce_dim,
        in_dfb,
        scale_dfb,
        out_dfb,
        compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
        compute_kernel_lib::ReduceInputBlockShape::of(rows, cols));

    UNPACK(tensix_sync());  // Workaround for issue #9370
}

template <
    uint32_t Ht,
    uint32_t Wt,
    uint32_t K,
    uint32_t logWt,
    uint32_t logk,
    uint32_t input_dfb_index,
    uint32_t expert_mask_dfb_index,
    uint32_t masked_input_dfb_index,
    uint32_t index_dfb_index,
    uint32_t input_transposed_dfb_index,
    uint32_t index_transposed_dfb_index,
    uint32_t values_dfb_index,
    uint32_t output_ind_dfb_index,
    uint32_t tile_width,
    bool first_call>
void mask_and_topk() {
    // dest indices for where to unpack the tiles for the llk
    // the input goes in index 0,1 and the index goes in index 2,3
    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;
    ckernel::topk_tile_init();

    DataflowBuffer input_dfb(input_dfb_index);
    DataflowBuffer expert_mask_dfb(expert_mask_dfb_index);
    DataflowBuffer masked_input_dfb(masked_input_dfb_index);
    DataflowBuffer index_dfb(index_dfb_index);
    DataflowBuffer input_transposed_dfb(input_transposed_dfb_index);
    DataflowBuffer index_transposed_dfb(index_transposed_dfb_index);
    DataflowBuffer values_dfb(values_dfb_index);
    DataflowBuffer output_ind_dfb(output_ind_dfb_index);

    if (first_call) {
        transpose_init(input_dfb_index);
    }

    // The expert mask is the same for all rows, so wait for all Wt tiles once before the loop.
    expert_mask_dfb.wait_front(Wt);

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        bool ascending = false;
        input_transposed_dfb.reserve_back(Wt);
        index_transposed_dfb.reserve_back(Wt);

        // streaming in input and index tiles to transpose and bitonic local sort them, two tiles at a time
        for (uint32_t wt = 0; wt < Wt; wt += 2) {
            input_dfb.wait_front(2);
            index_dfb.wait_front(2);

            // Before transposing, add expert_mask to the two input tiles and store the result in masked_input_dfb.
            tile_regs_acquire();
            reconfig_data_format(input_dfb_index, expert_mask_dfb_index);
            add_bcast_rows_init_short(input_dfb_index, expert_mask_dfb_index);
            add_tiles_bcast_rows(input_dfb_index, expert_mask_dfb_index, 0, wt, 0);
            add_tiles_bcast_rows(input_dfb_index, expert_mask_dfb_index, 1, wt + 1, 1);
            masked_input_dfb.reserve_back(2);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(masked_input_dfb_index);
            pack_tile(0, masked_input_dfb_index);
            pack_tile(1, masked_input_dfb_index);
            masked_input_dfb.push_back(2);
            input_dfb.pop_front(2);
            tile_regs_release();

            // Transpose masked input and index tiles, then locally sort into k groups.
            tile_regs_acquire();
            masked_input_dfb.wait_front(2);
            reconfig_data_format_srca(masked_input_dfb_index);
            transpose_init(masked_input_dfb_index);
            transpose_tile(masked_input_dfb_index, 0, 0);
            transpose_tile(masked_input_dfb_index, 1, 1);
            masked_input_dfb.pop_front(2);

            reconfig_data_format_srca(index_dfb_index);
            transpose_init(index_dfb_index);
            transpose_tile(index_dfb_index, 0, 2);
            transpose_tile(index_dfb_index, 1, 3);

            // llk_topk_sort -> inplace
            ckernel::topk_local_sort(0, (int)ascending, logk - 1);

            tile_regs_commit();

            index_dfb.pop_front(2);

            tile_regs_wait();
            // pack value tiles into cb_intermed0
            pack_reconfig_data_format(input_transposed_dfb_index);
            pack_tile(0, input_transposed_dfb_index);
            pack_tile(1, input_transposed_dfb_index);

            // pack index tiles into cb_intermed1
            pack_reconfig_data_format(index_transposed_dfb_index);
            pack_tile(2, index_transposed_dfb_index);
            pack_tile(3, index_transposed_dfb_index);
            tile_regs_release();
        }

        input_transposed_dfb.push_back(Wt);
        index_transposed_dfb.push_back(Wt);

        // iterative divide and conquer on pairs of tiles (bitonic topk merge and rebuild)
        // first iteration we compare 0th and 1st tile, then 2nd and 3rd, etc. We get the sorted top 32 values in each
        // pair. second iteration we compare 0th and 2nd tile, then 4th and 6th, etc. logWt iteration we compare 0th and
        // Wt/2 tile single buffer as we can pack tiles back in-place
        for (uint32_t m_iter = 0; m_iter < logWt; ++m_iter) {
            bool a = false;
            input_transposed_dfb.wait_front(Wt);
            index_transposed_dfb.wait_front(Wt);

            for (uint32_t left_ind = 0; left_ind < Wt - (1 << m_iter); left_ind += 2 << m_iter) {
                uint32_t right_ind = left_ind + (1 << m_iter);
                tile_regs_acquire();

                copy_tile_to_dst_init_short_with_dt(index_transposed_dfb_index, input_transposed_dfb_index);
                copy_tile(input_transposed_dfb_index, left_ind, input_dest_start);
                copy_tile(input_transposed_dfb_index, right_ind, input_dest_end);

                // unpack indices into dest
                copy_tile_to_dst_init_short_with_dt(input_transposed_dfb_index, index_transposed_dfb_index);
                copy_tile(index_transposed_dfb_index, left_ind, index_dest_start);
                copy_tile(index_transposed_dfb_index, right_ind, index_dest_end);

                // merge values - move larger 32 values into 0th dest and lower 32 values into 1st dest
                ckernel::topk_merge(0, m_iter, K);
                // sort within the larger 32 values
                ckernel::topk_rebuild(0, (uint32_t)a, m_iter, K, logk, true);

                tile_regs_commit();
                tile_regs_wait();
                // pack value tiles in-place in the single-buffered cb_intermed0, we only need the upper 32 values for
                // topk, which was in input_dest_start
                pack_reconfig_data_format(input_transposed_dfb_index);
                pack_tile<true>(input_dest_start, input_transposed_dfb_index, left_ind);

                // pack index tiles in-place in the single-buffered cb_intermed1, we only need the upper 32 values for
                // topk, which was in index_dest_start
                pack_reconfig_data_format(index_transposed_dfb_index);
                pack_tile<true>(index_dest_start, index_transposed_dfb_index, left_ind);
                tile_regs_release();
                a = !a;
            }

            input_transposed_dfb.reserve_back(Wt);
            index_transposed_dfb.reserve_back(Wt);

            input_transposed_dfb.pop_front(Wt);
            index_transposed_dfb.pop_front(Wt);

            input_transposed_dfb.push_back(Wt);
            index_transposed_dfb.push_back(Wt);
        }

        constexpr uint32_t Kt = K % tile_width == 0 ? K / tile_width : K / tile_width + 1;

        // transpose value tiles and pack into output buffer
        reconfig_data_format_srca(input_transposed_dfb_index);
        transpose_init(input_transposed_dfb_index);
        pack_reconfig_data_format(input_transposed_dfb_index);
        input_transposed_dfb.wait_front(Kt);
        for (uint32_t i = 0; i < Kt; ++i) {
            tile_regs_acquire();
            transpose_tile(input_transposed_dfb_index, i, 0);
            tile_regs_commit();

            values_dfb.reserve_back(1);

            tile_regs_wait();
            pack_tile(0, values_dfb_index);
            tile_regs_release();

            values_dfb.push_back(1);
        }
        input_transposed_dfb.wait_front(Wt);
        input_transposed_dfb.pop_front(Wt);

        // transpose index tiles and pack into output buffer
        reconfig_data_format_srca(index_transposed_dfb_index);
        transpose_init(index_transposed_dfb_index);
        pack_reconfig_data_format(index_transposed_dfb_index);
        index_transposed_dfb.wait_front(Kt);
        for (uint32_t i = 0; i < Kt; ++i) {
            tile_regs_acquire();
            transpose_tile(index_transposed_dfb_index, i, 0);
            tile_regs_commit();

            output_ind_dfb.reserve_back(1);

            tile_regs_wait();
            pack_tile(0, output_ind_dfb_index);
            tile_regs_release();

            output_ind_dfb.push_back(1);
        }
        index_transposed_dfb.wait_front(Wt);
        index_transposed_dfb.pop_front(Wt);
    }
    expert_mask_dfb.pop_front(Wt);
    // sfpu::_init_sfpu_config_reg();
}

void kernel_main() {
    constexpr uint32_t input_dfb_index = get_compile_time_arg_val(0);
    constexpr uint32_t topk_mask_dfb_index = get_compile_time_arg_val(1);
    constexpr uint32_t expert_mask_dfb_index = get_compile_time_arg_val(2);
    constexpr uint32_t scale_dfb_index = get_compile_time_arg_val(3);
    constexpr uint32_t index_dfb_index = get_compile_time_arg_val(4);
    constexpr uint32_t input_transposed_dfb_index = get_compile_time_arg_val(5);
    constexpr uint32_t index_transposed_dfb_index = get_compile_time_arg_val(6);
    constexpr uint32_t values_dfb_index = get_compile_time_arg_val(7);
    constexpr uint32_t output_ind_dfb_index = get_compile_time_arg_val(8);
    constexpr uint32_t out_dfb_index = get_compile_time_arg_val(9);

    constexpr uint32_t Ht = get_compile_time_arg_val(10);
    constexpr uint32_t Wt = get_compile_time_arg_val(11);
    constexpr uint32_t K = get_compile_time_arg_val(12);
    constexpr uint32_t logk = get_compile_time_arg_val(13);
    constexpr uint32_t logWt = get_compile_time_arg_val(14);

    constexpr uint32_t dfb_cur_max = get_compile_time_arg_val(15);
    constexpr uint32_t dfb_cur_sum = get_compile_time_arg_val(16);
    constexpr uint32_t tile_width = get_compile_time_arg_val(17);
    constexpr uint32_t masked_input_dfb_index = get_compile_time_arg_val(18);

    constexpr uint32_t Kt = K % tile_width == 0 ? K / tile_width : K / tile_width + 1;

    compute_kernel_hw_startup(input_dfb_index, input_transposed_dfb_index);

    // Apply expert_mask to each input tile pair and run top-k on the masked values.
    mask_and_topk<
        Ht,
        Wt,
        K,
        logWt,
        logk,
        input_dfb_index,
        expert_mask_dfb_index,
        masked_input_dfb_index,
        index_dfb_index,
        input_transposed_dfb_index,
        index_transposed_dfb_index,
        values_dfb_index,
        output_ind_dfb_index,
        tile_width,
        true>();

    // mask out all experts except the top-k
    add_block_bcast_rows_inplace(values_dfb_index, topk_mask_dfb_index, Ht, Kt, false);
    eqz_block_inplace(output_ind_dfb_index, Ht * Kt);

    // softmax
    reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, values_dfb_index, scale_dfb_index, dfb_cur_max>(Ht, Kt);
    sub_exp_block_bcast_cols_inplace<values_dfb_index, dfb_cur_max, Ht, Kt>();
    reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, values_dfb_index, scale_dfb_index, dfb_cur_sum>(Ht, Kt);
    recip_block_inplace(dfb_cur_sum, Ht);
    mul_block_bcast_cols_inplace(values_dfb_index, dfb_cur_sum, Ht, Kt);

    // select 0th expert
    mul_block_inplace(values_dfb_index, output_ind_dfb_index, Ht * Kt);

    // final sum
    reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, values_dfb_index, scale_dfb_index, out_dfb_index>(Ht, Kt);
}
