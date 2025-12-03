// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#define REDUCE_OP PoolType::SUM  // wtf why do we need to define this here?
#define REDUCE_DIM ReduceDim::REDUCE_COL

#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "ttnn/operations/reduction/topk/device/kernels/compute/topk_common_funcs.hpp"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/bcast.h"

#include "debug/dprint_tensix.h"

namespace NAMESPACE {

void print_tile(
    uint32_t cb_idx,
    uint32_t tile_idx,
    bool untilize = true,
    uint16_t start_row = 0,
    uint16_t end_row = 32,
    uint8_t start_col = 0,
    uint8_t end_col = 32) {
    DPRINT << "cb_idx: " << cb_idx << " tile_idx: " << tile_idx << ENDL();
    DPRINT << "======" << ENDL();
    for (uint16_t r = start_row; r < end_row; ++r) {
        DPRINT << (uint)r << " : "
               << TileSlice(
                      cb_idx,
                      tile_idx,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)start_col,
                          .w1 = (uint8_t)end_col,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

namespace blocks {
void add_bias_and_pack(
    uint32_t sigmoid_input_cb_index, uint32_t bias_cb_index, uint32_t add_bias_cb_index, uint32_t width_tiles) {
    // Perform add bias on sigmoid input – should I do full or partial init here?
    add_tiles_init(sigmoid_input_cb_index, bias_cb_index, false);
    cb_wait_front(sigmoid_input_cb_index, width_tiles);
    for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
        cb_wait_front(bias_cb_index, 1);

        tile_regs_acquire();
        add_tiles(sigmoid_input_cb_index, bias_cb_index, width_tile, 0, 0);
        tile_regs_commit();

        cb_reserve_back(add_bias_cb_index, 1);
        tile_regs_wait();
        pack_tile(0, add_bias_cb_index, 0);
        tile_regs_release();
        cb_push_back(add_bias_cb_index, 1);

        cb_pop_front(bias_cb_index, 1);
    }
}

void process_and_sort_tiles(
    uint32_t input_cb_index,
    uint32_t index_cb_index,
    uint32_t input_transposed_cb_index,
    uint32_t index_transposed_cb_index,
    uint32_t Wt,
    bool switch_dir,
    bool& ascending,
    int end_phase) {
    cb_wait_front(index_cb_index, Wt);
    cb_reserve_back(input_transposed_cb_index, Wt);
    cb_reserve_back(index_transposed_cb_index, Wt);

    // streaming in input and index tiles to transpose and bitonic local sort them, two tiles at a time
    for (uint32_t wt = 0; wt < Wt; wt += 2) {
        acquire_dst();
        // local sort into k groups
        cb_wait_front(input_cb_index, 2);

        // transpose and unpack into dest regs
        reconfig_data_format_srca(input_cb_index);
        transpose_wh_init_short(input_cb_index);
        transpose_wh_tile(input_cb_index, 0, 0);
        transpose_wh_tile(input_cb_index, 1, 1);

        // transpose and unpack into dest regs
        reconfig_data_format_srca(index_cb_index);
        transpose_wh_init_short(index_cb_index);
        transpose_wh_tile(index_cb_index, wt, 2);
        transpose_wh_tile(index_cb_index, wt + 1, 3);

        // llk_topk_sort -> inplace
        ckernel::topk_local_sort(0, (int)ascending, end_phase);

        // pack value tiles into cb_intermed0
        pack_reconfig_data_format(input_transposed_cb_index);
        pack_tile<true>(0, input_transposed_cb_index, wt);
        pack_tile<true>(1, input_transposed_cb_index, wt + 1);

        // pack index tiles into cb_intermed1
        pack_reconfig_data_format(index_transposed_cb_index);
        pack_tile<true>(2, index_transposed_cb_index, wt);
        pack_tile<true>(3, index_transposed_cb_index, wt + 1);

        cb_pop_front(input_cb_index, 2);
        // don't pop index_cb_index as it gets re-used for the next tile heights
        release_dst();
        ascending = switch_dir ? !ascending : ascending;
    }

    cb_push_back(input_transposed_cb_index, Wt);
    cb_push_back(index_transposed_cb_index, Wt);
}

void sum_top_experts_per_group(
    const uint32_t summed_experts_cb_index, const uint32_t group_scores_cb_index, uint32_t summed_experts_per_group) {
    // sum the top experts_per_group rows for each group
    binary_op_init_common(summed_experts_cb_index, summed_experts_cb_index, group_scores_cb_index);  // with full
    // init, good
    add_tiles_init(summed_experts_cb_index, summed_experts_cb_index, true);
    cb_wait_front(summed_experts_cb_index, summed_experts_per_group);

    cb_reserve_back(group_scores_cb_index, 1);
    tile_regs_acquire();
    for (uint32_t i = 0; i < summed_experts_per_group; i += 2) {
        add_tiles(summed_experts_cb_index, summed_experts_cb_index, i, i + 1, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, group_scores_cb_index);
    tile_regs_release();

    cb_push_back(group_scores_cb_index, 1);
    cb_pop_front(summed_experts_cb_index, summed_experts_per_group);
}

void topk_group_scores(
    const uint32_t group_scores_cb_index,
    const uint32_t group_indices_cb_index,
    const uint32_t sorted_group_indices_cb_index,
    bool switch_dir,
    bool& ascending,
    int log_topk_groups) {
    topk_tile_init();
    cb_reserve_back(sorted_group_indices_cb_index, 1);

    // Sort single input and index tile that have already ben transposed.
    acquire_dst();
    // local sort into k groups
    cb_wait_front(group_scores_cb_index, 1);
    cb_wait_front(group_indices_cb_index, 1);

    // copy scores tile to dest reg 0
    copy_tile_to_dst_init_short(group_scores_cb_index);
    copy_tile(group_scores_cb_index, 0, 0);

    // copy indices tile to dest reg 2
    // CVELE: Going to be correctly packed out if we use the reconfig call
    copy_tile_to_dst_init_short_with_dt(group_scores_cb_index, group_indices_cb_index);
    // CVELE: If we use this call, dprint will not use the correct data format
    // copy_tile_to_dst_init_short(group_indices_cb_index);
    copy_tile(group_indices_cb_index, 0, 2);

    // llk_topk_sort -> inplace
    ckernel::topk_local_sort(0, (int)ascending, log_topk_groups);

    // pack index tile into sorted_group_indices_cb_index
    pack_reconfig_data_format(sorted_group_indices_cb_index);
    pack_tile(2, sorted_group_indices_cb_index);
    // CVELE: If copy_tile_to_dst_init_short_with_dt is called, this will print correct values
    // PACK(print_tile(sorted_group_indices_cb_index, 0, true, 0, 8, 0, 16));

    cb_pop_front(group_scores_cb_index, 1);
    // don't pop group indices as it gets re-used for the next tile heights
    release_dst();

    cb_push_back(sorted_group_indices_cb_index, 1);
}

void transpose_and_pack(const uint32_t input_cb_index, const uint32_t output_cb_index, const uint32_t tiles) {
    // UNPACK(DPRINT << "Transpose and pack tiles: " << tiles << ENDL());
    reconfig_data_format_srca(input_cb_index);
    transpose_wh_init_short(input_cb_index);
    pack_reconfig_data_format(input_cb_index);
    for (uint32_t i = 0; i < tiles; i++) {
        cb_wait_front(input_cb_index, 1);
        // UNPACK(DPRINT << "Waiting for input cb DONE" << ENDL());

        cb_reserve_back(output_cb_index, 1);

        tile_regs_acquire();
        transpose_wh_tile(input_cb_index, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, output_cb_index);
        tile_regs_release();
        // PACK(print_tile(output_cb_index, i, true, 0, 8, 0, 1));
        cb_push_back(output_cb_index, 1);

        cb_pop_front(input_cb_index, 1);
    }
}

void topk(
    const uint32_t winning_group_scores_cb_index,
    const uint32_t winning_group_indices_cb_index,
    const uint32_t unnormalized_scores_cb_index,
    const uint32_t intermediate_local_sort_indices_cb_index,
    const uint32_t output_indices_cb_index,
    const uint32_t tiles,
    const uint32_t log_tiles,
    const uint32_t k,
    const uint32_t logk) {
    uint32_t Wt = tiles;
    uint32_t K = k;
    bool switch_dir = (K == 64);
    bool largest = true;
    bool ascending = !largest;
    int end_phase = (tiles <= 2) ? log_tiles - 1 : 5;
    // UNPACK(DPRINT << "Tiles: " << tiles << " Log Tiles: " << log_tiles << " End Phase: " << end_phase << ENDL());

    topk_tile_init();
    // acquire_dst();
    tile_regs_acquire();
    cb_wait_front(winning_group_scores_cb_index, tiles);
    cb_wait_front(winning_group_indices_cb_index, tiles);
    // local sort first two tiles:

    // transpose and unpack into dest regs
    reconfig_data_format_srca(winning_group_scores_cb_index);
    transpose_wh_init_short(winning_group_scores_cb_index);
    transpose_wh_tile(winning_group_scores_cb_index, 0, 0);
    transpose_wh_tile(winning_group_scores_cb_index, 1, 1);

    // transpose and unpack into dest regs
    reconfig_data_format_srca(winning_group_indices_cb_index);
    transpose_wh_init_short(winning_group_indices_cb_index);
    transpose_wh_tile(winning_group_indices_cb_index, 0, 2);
    transpose_wh_tile(winning_group_indices_cb_index, 1, 3);
    // llk_topk_sort -> inplace
    ckernel::topk_local_sort(0, (int)ascending, end_phase);

    // Use insertion sort; discard lower half and keep upper half
    // Compare upper half with the next tile; insert into correct position
    for (uint32_t j = 2; j < tiles; j++) {
        reconfig_data_format_srca(winning_group_scores_cb_index);
        transpose_wh_init_short(winning_group_scores_cb_index);
        transpose_wh_tile(winning_group_scores_cb_index, j, 1);

        reconfig_data_format_srca(winning_group_indices_cb_index);
        transpose_wh_init_short(winning_group_indices_cb_index);
        transpose_wh_tile(winning_group_indices_cb_index, j, 3);

        ckernel::topk_local_sort(0, (int)ascending, end_phase);
    }
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(unnormalized_scores_cb_index, 1);
    pack_reconfig_data_format(unnormalized_scores_cb_index);
    pack_tile(0, unnormalized_scores_cb_index);
    cb_push_back(unnormalized_scores_cb_index, 1);

    cb_reserve_back(intermediate_local_sort_indices_cb_index, 1);
    pack_reconfig_data_format(intermediate_local_sort_indices_cb_index);
    pack_tile(2, intermediate_local_sort_indices_cb_index);
    tile_regs_release();
    cb_push_back(intermediate_local_sort_indices_cb_index, 1);

    transpose_and_pack(intermediate_local_sort_indices_cb_index, output_indices_cb_index, 1);
}

void normalize_scores(
    const uint32_t unnormalized_scores_cb_index,
    const uint32_t reduce_scalar_cb_index,
    const uint32_t intermediate_reduce_cb_index,
    const uint32_t normalized_scores_cb_index) {
    compute_kernel_hw_startup(unnormalized_scores_cb_index, reduce_scalar_cb_index, normalized_scores_cb_index);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(
        unnormalized_scores_cb_index, reduce_scalar_cb_index, normalized_scores_cb_index);
    cb_wait_front(unnormalized_scores_cb_index, 1);
    cb_wait_front(reduce_scalar_cb_index, 1);
    // UNPACK(print_tile(unnormalized_scores_cb_index, 0, true, 0, 32, 0, 1));
    // UNPACK(print_tile(reduce_scalar_cb_index, 0, true, 0, 32, 0, 1));
    acquire_dst();

    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(unnormalized_scores_cb_index, reduce_scalar_cb_index, 0, 0, 0);
    reduce_uninit();

    recip_tile_init();
    recip_tile(0);  // DST[0] = 1/sum(unnormalized_scores)

    cb_reserve_back(intermediate_reduce_cb_index, 1);
    pack_tile(0, intermediate_reduce_cb_index);
    cb_push_back(intermediate_reduce_cb_index, 1);

    release_dst();
    // reduce_uninit();

    cb_wait_front(intermediate_reduce_cb_index, 1);
    // UNPACK(print_tile(intermediate_reduce_cb_index, 0, true, 0, 32, 0, 1));
    mul_bcast_cols_init_short(unnormalized_scores_cb_index, intermediate_reduce_cb_index);

    acquire_dst();
    cb_reserve_back(normalized_scores_cb_index, 1);
    UNPACK(print_tile(unnormalized_scores_cb_index, 0, true, 0, 32, 0, 1));
    UNPACK(print_tile(intermediate_reduce_cb_index, 0, true, 0, 1, 0, 1));
    mul_tiles_bcast<BroadcastType::COL>(
        unnormalized_scores_cb_index, intermediate_reduce_cb_index, 0, 0, 0);  // tile *= 1/(sum_col(tile))
    pack_tile(0, normalized_scores_cb_index);
    cb_push_back(normalized_scores_cb_index, 1);
    release_dst();
    cb_wait_front(normalized_scores_cb_index, 1);
    UNPACK(print_tile(normalized_scores_cb_index, 0, true, 0, 32, 0, 1));
}

}  // namespace blocks

void MAIN {
    // Dummy compute kernel
    constexpr uint32_t scores_cb_index = get_named_compile_time_arg_val("scores_cb_index");
    constexpr uint32_t bias_cb_index = get_named_compile_time_arg_val("bias_cb_index");
    constexpr uint32_t sigmoid_input_cb_index = get_named_compile_time_arg_val("sigmoid_input_cb_index");
    constexpr uint32_t add_bias_cb_index = get_named_compile_time_arg_val("add_bias_cb_index");
    constexpr uint32_t weights_cb_index = get_named_compile_time_arg_val("weights_cb_index");
    constexpr uint32_t indices_cb_index = get_named_compile_time_arg_val("indices_cb_index");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t scores_page_size = get_named_compile_time_arg_val("scores_page_size");
    constexpr uint32_t bias_page_size = get_named_compile_time_arg_val("bias_page_size");
    constexpr uint32_t topk_input_cb_index = get_named_compile_time_arg_val("topk_input_cb_index");
    constexpr uint32_t topk_index_cb_index = get_named_compile_time_arg_val("topk_index_cb_index");
    constexpr uint32_t topk_index_creation_cb_index = get_named_compile_time_arg_val("topk_index_creation_cb_index");
    constexpr uint32_t winning_group_scores_cb_index = get_named_compile_time_arg_val("winning_group_scores_cb_index");
    constexpr uint32_t winning_group_indices_cb_index =
        get_named_compile_time_arg_val("winning_group_indices_cb_index");

    constexpr uint32_t log_group_size = get_named_compile_time_arg_val("log_group_size");
    constexpr uint32_t group_size = get_named_compile_time_arg_val("group_size");
    constexpr uint32_t log_topk_groups = get_named_compile_time_arg_val("log_topk_groups");
    constexpr uint32_t topk_groups = get_named_compile_time_arg_val("topk_groups");
    constexpr uint32_t n_activated_experts = get_named_compile_time_arg_val("n_activated_experts");
    constexpr uint32_t log_n_activated_experts = get_named_compile_time_arg_val("log_n_activated_experts");
    constexpr uint32_t n_activated_expert_tiles = get_named_compile_time_arg_val("n_activated_expert_tiles");

    constexpr uint32_t group_indices_cb_index = get_named_compile_time_arg_val("group_indices_cb_index");
    constexpr uint32_t summed_experts_cb_index = get_named_compile_time_arg_val("summed_experts_cb_index");
    constexpr uint32_t group_scores_cb_index = get_named_compile_time_arg_val("group_scores_cb_index");
    constexpr uint32_t summed_experts_per_group = get_named_compile_time_arg_val("summed_experts_per_group");
    constexpr uint32_t sorted_group_indices_cb_index = get_named_compile_time_arg_val("sorted_group_indices_cb_index");
    constexpr uint32_t intermediate_local_sort_cb_index =
        get_named_compile_time_arg_val("intermediate_local_sort_cb_index");
    constexpr uint32_t intermediate_local_sort_indices_cb_index =
        get_named_compile_time_arg_val("intermediate_local_sort_indices_cb_index");
    constexpr uint32_t pre_normalized_scores_cb_index =
        get_named_compile_time_arg_val("pre_normalized_scores_cb_index");
    constexpr uint32_t reduce_scalar_cb_index = get_named_compile_time_arg_val("reduce_scalar_cb_index");

    constexpr uint32_t n_groups = get_named_compile_time_arg_val("n_groups");
    constexpr uint32_t log_n_groups = get_named_compile_time_arg_val("log_n_groups");

    constexpr uint32_t end_phase = log_group_size - 1;

    const uint32_t start_height_tile = get_arg_val<uint32_t>(0);
    const uint32_t end_height_tile = get_arg_val<uint32_t>(1);
    binary_op_init_common(scores_cb_index, bias_cb_index, add_bias_cb_index);

    for (uint32_t height_tile = start_height_tile; height_tile < end_height_tile; height_tile++) {
        uint32_t base_page = height_tile * width_tiles;

        // Perform sigmoid on scores
        for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
            cb_wait_front(scores_cb_index, 1);

            tile_regs_acquire();
            // copy tile from scores cb to destination register 0
            copy_tile_to_dst_init_short(scores_cb_index);
            copy_tile(scores_cb_index, width_tile, 0);

            sigmoid_tile_init();
            sigmoid_tile(0);
            tile_regs_commit();

            cb_reserve_back(sigmoid_input_cb_index, 1);
            tile_regs_wait();
            pack_tile<true>(0, sigmoid_input_cb_index, 0);
            tile_regs_release();
            // if (width_tile == 5 || width_tile == 0 || width_tile == 1 || width_tile == 3) {
            //     PACK(print_tile(sigmoid_input_cb_index, 0, true, 0, 1));
            // }
            cb_push_back(sigmoid_input_cb_index, 1);
        }

        // Perform add bias on sigmoid input – should I do full or partial init here?
        blocks::add_bias_and_pack(sigmoid_input_cb_index, bias_cb_index, add_bias_cb_index, width_tiles);

        // Transpose tiles into dest and then perform topk_local_sort
        bool ascending = false;
        bool switch_dir = false;
        cb_wait_front(add_bias_cb_index, width_tiles);
        cb_wait_front(topk_index_creation_cb_index, width_tiles);

        topk_tile_init();
        blocks::process_and_sort_tiles(
            add_bias_cb_index,
            topk_index_creation_cb_index,
            topk_input_cb_index,
            topk_index_cb_index,
            width_tiles,
            switch_dir,
            ascending,
            end_phase);

        cb_wait_front(topk_index_cb_index, width_tiles);

        blocks::sum_top_experts_per_group(summed_experts_cb_index, group_scores_cb_index, summed_experts_per_group);
        blocks::topk_group_scores(
            group_scores_cb_index,
            group_indices_cb_index,
            sorted_group_indices_cb_index,
            switch_dir,
            ascending,
            log_n_groups - 1);

        blocks::topk(
            winning_group_scores_cb_index,
            winning_group_indices_cb_index,
            pre_normalized_scores_cb_index,
            intermediate_local_sort_indices_cb_index,
            indices_cb_index,
            topk_groups,
            log_topk_groups,
            n_activated_experts,
            log_n_activated_experts);

        blocks::normalize_scores(
            pre_normalized_scores_cb_index, reduce_scalar_cb_index, intermediate_local_sort_cb_index, scores_cb_index);
    }
}
}  // namespace NAMESPACE
