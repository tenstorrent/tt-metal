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
#include "compute_kernel_api/eltwise_binary_sfpu.h"

#include "debug/dprint_tensix.h"

namespace NAMESPACE {

namespace blocks {
void sigmoid(uint32_t cb_in_scores, uint32_t cb_sigmoid_scores, uint32_t width_tiles) {
    // Perform sigmoid on scores
    // Reconfigure pack/unpack for bfloat16 after topk operations used UInt16
    for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
        cb_wait_front(cb_in_scores, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_in_scores);
        // copy tile from scores cb to destination register 0
        copy_tile_to_dst_init_short(cb_in_scores);
        copy_tile(cb_in_scores, 0, 0);
        // perform sigmoid on tile
        sigmoid_tile_init();
        sigmoid_tile(0);
        tile_regs_commit();
        cb_pop_front(cb_in_scores, 1);

        cb_reserve_back(cb_sigmoid_scores, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_sigmoid_scores);
        pack_tile(0, cb_sigmoid_scores);
        tile_regs_release();
        cb_push_back(cb_sigmoid_scores, 1);
    }
}

void add_bias(uint32_t cb_sigmoid_scores, uint32_t cb_in_bias, uint32_t cb_biased_scores, uint32_t width_tiles) {
    // Perform add bias on sigmoid scores
    add_tiles_init(cb_sigmoid_scores, cb_in_bias, false);
    cb_wait_front(cb_sigmoid_scores, width_tiles);
    for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
        cb_wait_front(cb_in_bias, 1);
        tile_regs_acquire();
        add_tiles(cb_sigmoid_scores, cb_in_bias, width_tile, 0, 0);
        tile_regs_commit();
        cb_pop_front(cb_in_bias, 1);

        cb_reserve_back(cb_biased_scores, 1);
        tile_regs_wait();
        pack_tile(0, cb_biased_scores);
        tile_regs_release();
        cb_push_back(cb_biased_scores, 1);
    }
}

void process_and_sort_tiles(
    uint32_t cb_biased_scores,
    uint32_t cb_expert_index_template,
    uint32_t cb_sorted_group_scores,
    uint32_t cb_sorted_expert_indices_temp,
    uint32_t Wt,
    bool switch_dir,
    bool ascending,
    int end_phase) {
    topk_tile_init();
    // streaming in input and index tiles to transpose and bitonic local sort them, two tiles at a time
    cb_wait_front(cb_expert_index_template, Wt);
    cb_wait_front(cb_biased_scores, Wt);
    for (uint32_t wt = 0; wt < Wt; wt += 2) {
        acquire_dst();
        // transpose and unpack into dest regs
        reconfig_data_format_srca(cb_biased_scores);
        transpose_wh_init_short(cb_biased_scores);
        transpose_wh_tile(cb_biased_scores, wt, 0);
        transpose_wh_tile(cb_biased_scores, wt + 1, 1);

        // transpose and unpack into dest regs
        reconfig_data_format_srca(cb_expert_index_template);
        transpose_wh_init_short(cb_expert_index_template);
        transpose_wh_tile(cb_expert_index_template, wt, 2);
        transpose_wh_tile(cb_expert_index_template, wt + 1, 3);

        // llk_topk_sort -> inplace
        ckernel::topk_local_sort(0, (int)ascending, end_phase);

        // pack sorted score tiles
        pack_reconfig_data_format(cb_sorted_group_scores);
        cb_reserve_back(cb_sorted_group_scores, 1);
        pack_tile(0, cb_sorted_group_scores);
        cb_push_back(cb_sorted_group_scores, 1);

        cb_reserve_back(cb_sorted_group_scores, 1);
        pack_tile(1, cb_sorted_group_scores);
        cb_push_back(cb_sorted_group_scores, 1);

        // pack sorted index tiles
        pack_reconfig_data_format(cb_sorted_expert_indices_temp);
        cb_reserve_back(cb_sorted_expert_indices_temp, 1);
        pack_tile(2, cb_sorted_expert_indices_temp);
        cb_push_back(cb_sorted_expert_indices_temp, 1);

        cb_reserve_back(cb_sorted_expert_indices_temp, 1);
        pack_tile(3, cb_sorted_expert_indices_temp);
        cb_push_back(cb_sorted_expert_indices_temp, 1);

        cb_wait_front(cb_sorted_expert_indices_temp, 2);
        cb_pop_front(cb_sorted_expert_indices_temp, 2);

        release_dst();
        ascending = switch_dir ? !ascending : ascending;
    }
}

void sum_top_experts_per_group(
    const uint32_t cb_top_experts_per_group, const uint32_t cb_group_summed_scores, uint32_t summed_experts_per_group) {
    // sum the top experts_per_group rows for each group
    binary_op_init_common(cb_top_experts_per_group, cb_top_experts_per_group, cb_group_summed_scores);
    add_tiles_init(cb_top_experts_per_group, cb_top_experts_per_group, true);
    cb_wait_front(cb_top_experts_per_group, summed_experts_per_group);

    cb_reserve_back(cb_group_summed_scores, 1);
    tile_regs_acquire();
    for (uint32_t i = 0; i < summed_experts_per_group; i += 2) {
        add_tiles(cb_top_experts_per_group, cb_top_experts_per_group, i, i + 1, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_group_summed_scores);
    tile_regs_release();

    cb_push_back(cb_group_summed_scores, 1);
    cb_pop_front(cb_top_experts_per_group, summed_experts_per_group);
}

void topk_group_scores(
    const uint32_t cb_group_summed_scores,
    const uint32_t cb_group_index_template,
    const uint32_t cb_sorted_group_order,
    bool switch_dir,
    bool ascending,
    int log_topk_groups) {
    topk_tile_init();
    cb_reserve_back(cb_sorted_group_order, 1);

    // Sort single input and index tile that have already ben transposed.
    acquire_dst();
    // local sort into k groups
    cb_wait_front(cb_group_summed_scores, 1);
    cb_wait_front(cb_group_index_template, 1);

    // copy scores tile to dest reg 0
    copy_tile_to_dst_init_short(cb_group_summed_scores);
    copy_tile(cb_group_summed_scores, 0, 0);

    // copy indices tile to dest reg 2
    // CVELE: Going to be correctly packed out if we use the reconfig call
    copy_tile_to_dst_init_short_with_dt(cb_group_summed_scores, cb_group_index_template);
    // CVELE: If we use this call, dprint will not use the correct data format
    // copy_tile_to_dst_init_short(cb_group_index_template);
    copy_tile(cb_group_index_template, 0, 2);

    // llk_topk_sort -> inplace
    ckernel::topk_local_sort(0, (int)ascending, log_topk_groups);

    // pack index tile into cb_sorted_group_order
    pack_reconfig_data_format(cb_sorted_group_order);
    pack_tile(2, cb_sorted_group_order);
    cb_pop_front(cb_group_summed_scores, 1);
    // don't pop group indices as it gets re-used for the next tile heights
    release_dst();

    cb_push_back(cb_sorted_group_order, 1);
}

void transpose_and_pack(const uint32_t input_cb_index, const uint32_t output_cb_index, const uint32_t tiles) {
    reconfig_data_format_srca(input_cb_index);
    transpose_wh_init_short(input_cb_index);
    pack_reconfig_data_format(output_cb_index);  // uncommenting this causes a hang
    for (uint32_t i = 0; i < tiles; i++) {
        tile_regs_acquire();
        cb_wait_front(input_cb_index, 1);
        transpose_wh_tile(input_cb_index, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(output_cb_index, 1);

        pack_tile(0, output_cb_index);
        tile_regs_release();
        cb_push_back(output_cb_index, 1);

        cb_pop_front(input_cb_index, 1);
    }
}

void topk(
    const uint32_t cb_winning_group_scores,
    const uint32_t cb_winning_group_indices,
    const uint32_t cb_final_indices_transposed,
    const uint32_t cb_out_indices,
    const uint32_t tiles,
    const uint32_t log_tiles,
    const uint32_t k,
    const uint32_t logk) {
    bool switch_dir = (k == 64);
    bool ascending = false;
    int end_phase = (tiles <= 2) ? log_tiles - 1 : 5;

    topk_tile_init();
    // acquire_dst();
    tile_regs_acquire();
    cb_wait_front(cb_winning_group_scores, tiles);
    cb_wait_front(cb_winning_group_indices, tiles);
    // local sort first two tiles:

    // transpose and unpack into dest regs
    reconfig_data_format_srca(cb_winning_group_scores);
    transpose_wh_init_short(cb_winning_group_scores);
    transpose_wh_tile(cb_winning_group_scores, 0, 0);
    transpose_wh_tile(cb_winning_group_scores, 1, 1);

    // transpose and unpack into dest regs
    reconfig_data_format_srca(cb_winning_group_indices);
    transpose_wh_init_short(cb_winning_group_indices);
    transpose_wh_tile(cb_winning_group_indices, 0, 2);
    transpose_wh_tile(cb_winning_group_indices, 1, 3);
    // llk_topk_sort -> inplace
    ckernel::topk_local_sort(0, (int)ascending, 4);
    ckernel::topk_merge(0, 0, 32);

    // Use insertion sort; discard lower half and keep upper half
    // Compare upper half with the next tile; insert into correct position
    for (uint32_t j = 2; j < tiles; j++) {
        reconfig_data_format_srca(cb_winning_group_scores);
        transpose_wh_init_short(cb_winning_group_scores);
        transpose_wh_tile(cb_winning_group_scores, j, 1);

        reconfig_data_format_srca(cb_winning_group_indices);
        transpose_wh_init_short(cb_winning_group_indices);
        transpose_wh_tile(cb_winning_group_indices, j, 3);

        ckernel::topk_local_sort(0, (int)ascending, 4);
        ckernel::topk_merge(0, 0, 32);
    }
    ckernel::topk_rebuild(0, (int)ascending, 0, 32, 5, true);
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_final_indices_transposed, 1);
    pack_reconfig_data_format(cb_final_indices_transposed);
    pack_tile(2, cb_final_indices_transposed);
    tile_regs_release();
    cb_push_back(cb_final_indices_transposed, 1);

    transpose_and_pack(cb_final_indices_transposed, cb_out_indices, 1);

    cb_pop_front(cb_winning_group_scores, tiles);
    cb_pop_front(cb_winning_group_indices, tiles);
}

void normalize_scores(
    const uint32_t cb_gathered_sigmoid,
    const uint32_t cb_reduce_ones_scalar,
    const uint32_t cb_reduce_intermediate,
    const uint32_t cb_reciprocal_sums,
    const uint32_t cb_epsilon_scalar,
    const uint32_t cb_normalized_scores) {
    reconfig_data_format(cb_gathered_sigmoid, cb_reduce_ones_scalar);
    pack_reconfig_data_format(cb_normalized_scores);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(
        cb_gathered_sigmoid, cb_reduce_ones_scalar, cb_reduce_intermediate);

    cb_wait_front(cb_gathered_sigmoid, 1);
    cb_wait_front(cb_reduce_ones_scalar, 1);

    // 1. Sum row (experts) to get row vector of sums [1, 32]
    tile_regs_acquire();
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_gathered_sigmoid, cb_reduce_ones_scalar, 0, 0, 0);
    tile_regs_commit();
    reduce_uninit();

    // 2. Pack sums to intermediate to add epsilon
    tile_regs_wait();
    cb_reserve_back(cb_reduce_intermediate, 1);
    pack_tile(0, cb_reduce_intermediate);
    tile_regs_release();
    cb_push_back(cb_reduce_intermediate, 1);
    // 3. Add epsilon
    tile_regs_acquire();
    cb_wait_front(cb_epsilon_scalar, 1);
    cb_wait_front(cb_reduce_intermediate, 1);

    reconfig_data_format(cb_reduce_intermediate, cb_epsilon_scalar);
    pack_reconfig_data_format(cb_reciprocal_sums);

    add_bcast_scalar_init_short(cb_reduce_intermediate, cb_epsilon_scalar);
    add_tiles_bcast<BroadcastType::SCALAR>(cb_reduce_intermediate, cb_epsilon_scalar, 0, 0, 0);

    // 4. Recip
    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();

    cb_pop_front(cb_reduce_intermediate, 1);

    // 5. Pack reciprocals
    tile_regs_wait();
    cb_reserve_back(cb_reciprocal_sums, 1);
    pack_tile(0, cb_reciprocal_sums);
    cb_push_back(cb_reciprocal_sums, 1);
    tile_regs_release();

    // 6. Broadcast multiply
    tile_regs_acquire();
    cb_wait_front(cb_reciprocal_sums, 1);
    mul_bcast_cols_init_short(cb_gathered_sigmoid, cb_reciprocal_sums);
    mul_tiles_bcast<BroadcastType::COL>(cb_gathered_sigmoid, cb_reciprocal_sums, 0, 0, 0);  // tile *= 1/(sum_col(tile))
    tile_regs_commit();
    cb_pop_front(cb_reciprocal_sums, 1);
    cb_pop_front(cb_gathered_sigmoid, 1);

    tile_regs_wait();
    cb_reserve_back(cb_normalized_scores, 1);
    pack_reconfig_data_format(cb_normalized_scores);
    pack_tile(0, cb_normalized_scores);
    cb_push_back(cb_normalized_scores, 1);
    tile_regs_release();
}

void scale(const uint32_t cb_normalized_scores, const uint32_t cb_route_scale_scalar, const uint32_t cb_out_weights) {
    cb_wait_front(cb_normalized_scores, 1);
    cb_wait_front(cb_route_scale_scalar, 1);
    mul_tiles_bcast_scalar_init_short(cb_normalized_scores, cb_route_scale_scalar);

    tile_regs_acquire();

    mul_tiles_bcast<BroadcastType::SCALAR>(cb_normalized_scores, cb_route_scale_scalar, 0, 0, 0);
    tile_regs_commit();

    cb_reserve_back(cb_out_weights, 1);
    tile_regs_wait();
    pack_tile(0, cb_out_weights);
    cb_push_back(cb_out_weights, 1);
    tile_regs_release();

    cb_pop_front(cb_normalized_scores, 1);
}

}  // namespace blocks

void MAIN {
    // Circular buffer indices
    constexpr uint32_t cb_in_scores = get_named_compile_time_arg_val("cb_in_scores");
    constexpr uint32_t cb_in_bias = get_named_compile_time_arg_val("cb_in_bias");
    constexpr uint32_t cb_sigmoid_scores = get_named_compile_time_arg_val("cb_sigmoid_scores");
    constexpr uint32_t cb_biased_scores = get_named_compile_time_arg_val("cb_biased_scores");
    constexpr uint32_t cb_out_weights = get_named_compile_time_arg_val("cb_out_weights");
    constexpr uint32_t cb_out_indices = get_named_compile_time_arg_val("cb_out_indices");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t scores_page_size = get_named_compile_time_arg_val("scores_page_size");
    constexpr uint32_t bias_page_size = get_named_compile_time_arg_val("bias_page_size");
    constexpr uint32_t cb_sorted_group_scores = get_named_compile_time_arg_val("cb_sorted_group_scores");
    constexpr uint32_t cb_sorted_expert_indices_temp = get_named_compile_time_arg_val("cb_sorted_expert_indices_temp");
    constexpr uint32_t cb_expert_index_template = get_named_compile_time_arg_val("cb_expert_index_template");
    constexpr uint32_t cb_winning_group_scores = get_named_compile_time_arg_val("cb_winning_group_scores");
    constexpr uint32_t cb_winning_group_indices = get_named_compile_time_arg_val("cb_winning_group_indices");

    constexpr uint32_t log_group_size = get_named_compile_time_arg_val("log_group_size");
    constexpr uint32_t group_size = get_named_compile_time_arg_val("group_size");
    constexpr uint32_t log_topk_groups = get_named_compile_time_arg_val("log_topk_groups");
    constexpr uint32_t topk_groups = get_named_compile_time_arg_val("topk_groups");
    constexpr uint32_t n_activated_experts = get_named_compile_time_arg_val("n_activated_experts");
    constexpr uint32_t log_n_activated_experts = get_named_compile_time_arg_val("log_n_activated_experts");
    constexpr uint32_t n_activated_expert_tiles = get_named_compile_time_arg_val("n_activated_expert_tiles");

    constexpr uint32_t cb_group_index_template = get_named_compile_time_arg_val("cb_group_index_template");
    constexpr uint32_t cb_top_experts_per_group = get_named_compile_time_arg_val("cb_top_experts_per_group");
    constexpr uint32_t cb_group_summed_scores = get_named_compile_time_arg_val("cb_group_summed_scores");
    constexpr uint32_t summed_experts_per_group = get_named_compile_time_arg_val("summed_experts_per_group");
    constexpr uint32_t cb_sorted_group_order = get_named_compile_time_arg_val("cb_sorted_group_order");
    constexpr uint32_t cb_reduce_intermediate = get_named_compile_time_arg_val("cb_reduce_intermediate");
    constexpr uint32_t cb_final_indices_transposed = get_named_compile_time_arg_val("cb_final_indices_transposed");
    constexpr uint32_t cb_reduce_ones_scalar = get_named_compile_time_arg_val("cb_reduce_ones_scalar");
    constexpr uint32_t cb_epsilon_scalar = get_named_compile_time_arg_val("cb_epsilon_scalar");
    constexpr uint32_t cb_route_scale_scalar = get_named_compile_time_arg_val("cb_route_scale_scalar");
    constexpr uint32_t cb_normalized_scores = get_named_compile_time_arg_val("cb_normalized_scores");
    constexpr uint32_t cb_reciprocal_sums = get_named_compile_time_arg_val("cb_reciprocal_sums");
    constexpr uint32_t cb_gathered_sigmoid = get_named_compile_time_arg_val("cb_gathered_sigmoid");

    constexpr uint32_t n_groups = get_named_compile_time_arg_val("n_groups");
    constexpr uint32_t log_n_groups = get_named_compile_time_arg_val("log_n_groups");

    constexpr uint32_t end_phase = log_group_size - 1;

    const uint32_t start_height_tile = get_arg_val<uint32_t>(0);
    const uint32_t end_height_tile = get_arg_val<uint32_t>(1);
    binary_op_init_common(cb_in_scores, cb_in_bias, cb_biased_scores);

    for (uint32_t height_tile = start_height_tile; height_tile < end_height_tile; height_tile++) {
        blocks::sigmoid(cb_in_scores, cb_sigmoid_scores, width_tiles);

        // Perform add bias on sigmoid scores
        blocks::add_bias(cb_sigmoid_scores, cb_in_bias, cb_biased_scores, width_tiles);
        // Note: cb_sigmoid_scores is NOT popped here - writer will pop it after gather
        // Transpose tiles into dest and then perform topk_local_sort
        blocks::process_and_sort_tiles(
            cb_biased_scores,
            cb_expert_index_template,
            cb_sorted_group_scores,
            cb_sorted_expert_indices_temp,
            width_tiles,
            false,
            false,
            end_phase);
        blocks::sum_top_experts_per_group(cb_top_experts_per_group, cb_group_summed_scores, summed_experts_per_group);
        blocks::topk_group_scores(
            cb_group_summed_scores, cb_group_index_template, cb_sorted_group_order, false, false, log_n_groups - 1);
        blocks::topk(
            cb_winning_group_scores,
            cb_winning_group_indices,
            cb_final_indices_transposed,
            cb_out_indices,
            topk_groups,
            log_topk_groups,
            n_activated_experts,
            log_n_activated_experts);
        blocks::normalize_scores(
            cb_gathered_sigmoid,
            cb_reduce_ones_scalar,
            cb_reduce_intermediate,
            cb_reciprocal_sums,
            cb_epsilon_scalar,
            cb_normalized_scores);
        blocks::scale(cb_normalized_scores, cb_route_scale_scalar, cb_out_weights);
    }
}
}  // namespace NAMESPACE
