// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared MoE-gate compute blocks. Included by both moe_grouped_topk and moe_hash_gate compute
// kernels so the activation / normalize / scale (and top-k) stages live in exactly one place.
// Follows the same cross-op kernel-header pattern as reduction/topk/.../topk_common_funcs.hpp.

#pragma once

// REDUCE_OP / REDUCE_DIM must be defined before including the reduce compute API (it reads them at
// preprocessor time to select the reduce kernel variant).
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_COL

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "ttnn/operations/reduction/topk/device/kernels/compute/topk_common_funcs.hpp"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"

namespace blocks {
// Router affinity activations applied to the raw gate logits.
constexpr uint32_t SCORE_FUNC_SIGMOID = 0;       // DeepSeek-V3 / Kimi
constexpr uint32_t SCORE_FUNC_SQRTSOFTPLUS = 1;  // DeepSeek-V4: sqrt(softplus(x))

// Applies the selected router affinity activation to each logits tile. The output CB (historically
// "sigmoid" scores) holds the unbiased activated scores that the writer later gathers for the weights.
template <uint32_t score_func>
void apply_score_func(uint32_t cb_in_scores_id, uint32_t cb_activated_scores_id, uint32_t width_tiles) {
    CircularBuffer cb_in_scores(cb_in_scores_id);
    CircularBuffer cb_activated_scores(cb_activated_scores_id);
    for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
        cb_in_scores.wait_front(1);
        tile_regs_acquire();
        // Reconfigure the unpacker for float32 input (a prior top-k iteration may have left it on UInt16).
        reconfig_data_format_srca(cb_in_scores_id);
        // copy tile from scores cb to destination register 0
        copy_tile_to_dst_init_short(cb_in_scores_id);
        copy_tile(cb_in_scores_id, 0, 0);
        if constexpr (score_func == SCORE_FUNC_SQRTSOFTPLUS) {
            // sqrt(softplus(x)) with beta=1, threshold=20 (matches torch.nn.functional.softplus defaults).
            constexpr uint32_t const_1_fp32 = 0x3F800000;   // 1.0f -> beta and beta_reciprocal
            constexpr uint32_t const_20_fp32 = 0x41A00000;  // 20.0f -> threshold
            softplus_tile_init();
            softplus_tile(0, const_1_fp32, const_1_fp32, const_20_fp32);
            sqrt_tile_init();
            sqrt_tile(0);
        } else {
            sigmoid_tile_init();
            sigmoid_tile(0);
        }
        tile_regs_commit();
        cb_in_scores.pop_front(1);

        cb_activated_scores.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_activated_scores_id);
        pack_tile(0, cb_activated_scores_id);
        tile_regs_release();
        cb_activated_scores.push_back(1);
    }
}

void add_bias(
    uint32_t cb_sigmoid_scores_id, uint32_t cb_in_bias_id, uint32_t cb_biased_scores_id, uint32_t width_tiles) {
    CircularBuffer cb_sigmoid_scores(cb_sigmoid_scores_id);
    CircularBuffer cb_in_bias(cb_in_bias_id);
    CircularBuffer cb_biased_scores(cb_biased_scores_id);
    // Perform add bias on sigmoid scores
    add_init(cb_sigmoid_scores_id, cb_in_bias_id, false);
    cb_sigmoid_scores.wait_front(width_tiles);
    for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
        cb_in_bias.wait_front(1);
        tile_regs_acquire();
        add_tiles(cb_sigmoid_scores_id, cb_in_bias_id, width_tile, 0, 0);
        tile_regs_commit();
        cb_in_bias.pop_front(1);

        cb_biased_scores.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_biased_scores_id);
        pack_tile(0, cb_biased_scores_id);
        tile_regs_release();
        cb_biased_scores.push_back(1);
    }
}

template <bool stable_sort = false>
void process_and_sort_tiles(
    uint32_t cb_biased_scores_id,
    uint32_t cb_expert_index_template_id,
    uint32_t cb_sorted_group_scores_id,
    uint32_t cb_sorted_expert_indices_temp_id,
    uint32_t Wt,
    bool switch_dir,
    bool ascending,
    int end_phase) {
    CircularBuffer cb_biased_scores(cb_biased_scores_id);
    CircularBuffer cb_expert_index_template(cb_expert_index_template_id);
    CircularBuffer cb_sorted_group_scores(cb_sorted_group_scores_id);
    CircularBuffer cb_sorted_expert_indices_temp(cb_sorted_expert_indices_temp_id);
    topk_tile_init();
    // streaming in input and index tiles to transpose and bitonic local sort them, two tiles at a time
    cb_expert_index_template.wait_front(Wt);
    cb_biased_scores.wait_front(Wt);
    for (uint32_t wt = 0; wt < Wt; wt += 2) {
        tile_regs_acquire();
        // transpose and unpack into dest regs
        reconfig_data_format_srca(cb_biased_scores_id);
        transpose_init(cb_biased_scores_id);
        transpose_tile(cb_biased_scores_id, wt, 0);
        transpose_tile(cb_biased_scores_id, wt + 1, 1);

        // transpose and unpack into dest regs
        reconfig_data_format_srca(cb_expert_index_template_id);
        transpose_init(cb_expert_index_template_id);
        transpose_tile(cb_expert_index_template_id, wt, 2);
        transpose_tile(cb_expert_index_template_id, wt + 1, 3);

        // llk_topk_sort -> inplace
        ckernel::topk_local_sort<stable_sort>(0, (int)ascending, end_phase);

        // pack sorted score tiles
        pack_reconfig_data_format(cb_sorted_group_scores_id);
        cb_sorted_group_scores.reserve_back(1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sorted_group_scores_id);
        cb_sorted_group_scores.push_back(1);

        cb_sorted_group_scores.reserve_back(1);
        pack_tile(1, cb_sorted_group_scores_id);
        cb_sorted_group_scores.push_back(1);

        // pack sorted index tiles
        pack_reconfig_data_format(cb_sorted_expert_indices_temp_id);
        cb_sorted_expert_indices_temp.reserve_back(1);
        pack_tile(2, cb_sorted_expert_indices_temp_id);
        cb_sorted_expert_indices_temp.push_back(1);

        cb_sorted_expert_indices_temp.reserve_back(1);
        pack_tile(3, cb_sorted_expert_indices_temp_id);
        cb_sorted_expert_indices_temp.push_back(1);

        cb_sorted_expert_indices_temp.wait_front(2);
        cb_sorted_expert_indices_temp.pop_front(2);

        tile_regs_release();
        ascending = switch_dir ? !ascending : ascending;
    }
}

void sum_top_experts_per_group(
    const uint32_t cb_top_experts_per_group_id,
    const uint32_t cb_group_summed_scores_id,
    uint32_t summed_experts_per_group) {
    CircularBuffer cb_top_experts_per_group(cb_top_experts_per_group_id);
    CircularBuffer cb_group_summed_scores(cb_group_summed_scores_id);
    // sum the top experts_per_group rows for each group
    compute_kernel_hw_startup(cb_top_experts_per_group_id, cb_top_experts_per_group_id, cb_group_summed_scores_id);
    add_init(cb_top_experts_per_group_id, cb_top_experts_per_group_id, true);
    cb_top_experts_per_group.wait_front(summed_experts_per_group);

    cb_group_summed_scores.reserve_back(1);
    tile_regs_acquire();
    for (uint32_t i = 0; i < summed_experts_per_group; i += 2) {
        add_tiles(cb_top_experts_per_group_id, cb_top_experts_per_group_id, i, i + 1, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_group_summed_scores_id);
    pack_tile(0, cb_group_summed_scores_id);
    tile_regs_release();

    cb_group_summed_scores.push_back(1);
    cb_top_experts_per_group.pop_front(summed_experts_per_group);
}

template <bool stable_sort = false>
void topk_group_scores(
    const uint32_t cb_group_summed_scores_id,
    const uint32_t cb_group_index_template_id,
    const uint32_t cb_sorted_group_order_id,
    bool switch_dir,
    bool ascending,
    int log_topk_groups) {
    CircularBuffer cb_group_summed_scores(cb_group_summed_scores_id);
    CircularBuffer cb_group_index_template(cb_group_index_template_id);
    CircularBuffer cb_sorted_group_order(cb_sorted_group_order_id);
    topk_tile_init();
    cb_sorted_group_order.reserve_back(1);

    // Sort single input and index tile that have already ben transposed.
    tile_regs_acquire();
    // local sort into k groups
    cb_group_summed_scores.wait_front(1);
    cb_group_index_template.wait_front(1);

    // copy scores tiles to dest reg 0 and index tiles to dest reg 2
    copy_tile_to_dst_init_short(cb_group_summed_scores_id);
    copy_tile(cb_group_summed_scores_id, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_group_summed_scores_id, cb_group_index_template_id);
    copy_tile(cb_group_index_template_id, 0, 2);

    // llk_topk_sort -> inplace
    ckernel::topk_local_sort<stable_sort>(0, (int)ascending, log_topk_groups);

    tile_regs_commit();
    tile_regs_wait();
    // pack index tile into cb_sorted_group_order
    pack_reconfig_data_format(cb_sorted_group_order_id);
    pack_tile(2, cb_sorted_group_order_id);
    cb_group_summed_scores.pop_front(1);
    // don't pop group indices as it gets reused for the next tile heights
    tile_regs_release();

    cb_sorted_group_order.push_back(1);
}

void transpose_and_pack(const uint32_t input_cb_index, const uint32_t output_cb_index, const uint32_t tiles) {
    CircularBuffer input_cb(input_cb_index);
    CircularBuffer output_cb(output_cb_index);
    reconfig_data_format_srca(input_cb_index);
    transpose_init(input_cb_index);
    pack_reconfig_data_format(output_cb_index);
    for (uint32_t i = 0; i < tiles; i++) {
        tile_regs_acquire();
        input_cb.wait_front(1);
        transpose_tile(input_cb_index, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        output_cb.reserve_back(1);

        pack_tile(0, output_cb_index);
        tile_regs_release();
        output_cb.push_back(1);

        input_cb.pop_front(1);
    }
}

template <bool stable_sort = false>
void topk(
    const uint32_t cb_winning_group_scores_id,
    const uint32_t cb_winning_group_indices_id,
    const uint32_t cb_final_indices_transposed_id,
    const uint32_t cb_out_indices_id,
    const uint32_t tiles,
    const uint32_t log_tiles,
    const uint32_t k) {
    CircularBuffer cb_winning_group_scores(cb_winning_group_scores_id);
    CircularBuffer cb_winning_group_indices(cb_winning_group_indices_id);
    CircularBuffer cb_final_indices_transposed(cb_final_indices_transposed_id);
    bool switch_dir = (k == 64);
    bool ascending = false;
    int end_phase = (tiles <= 2) ? log_tiles - 1 : 5;

    topk_tile_init();
    tile_regs_acquire();
    cb_winning_group_scores.wait_front(tiles);
    cb_winning_group_indices.wait_front(tiles);
    // local sort first two tiles:

    // transpose and unpack into dest regs
    reconfig_data_format_srca(cb_winning_group_scores_id);
    transpose_init(cb_winning_group_scores_id);
    transpose_tile(cb_winning_group_scores_id, 0, 0);
    transpose_tile(cb_winning_group_scores_id, 1, 1);

    // transpose and unpack into dest regs
    reconfig_data_format_srca(cb_winning_group_indices_id);
    transpose_init(cb_winning_group_indices_id);
    transpose_tile(cb_winning_group_indices_id, 0, 2);
    transpose_tile(cb_winning_group_indices_id, 1, 3);
    // llk_topk_sort -> inplace
    ckernel::topk_local_sort<stable_sort>(0, (int)ascending, 4);
    ckernel::topk_merge<false, stable_sort>(0, 0, 32);

    // Use insertion sort; discard lower half and keep upper half
    // Compare upper half with the next tile; insert into correct position
    for (uint32_t j = 2; j < tiles; j++) {
        reconfig_data_format_srca(cb_winning_group_scores_id);
        transpose_init(cb_winning_group_scores_id);
        transpose_tile(cb_winning_group_scores_id, j, 1);

        reconfig_data_format_srca(cb_winning_group_indices_id);
        transpose_init(cb_winning_group_indices_id);
        transpose_tile(cb_winning_group_indices_id, j, 3);

        ckernel::topk_local_sort<stable_sort>(0, (int)ascending, 4);
        ckernel::topk_merge<false, stable_sort>(0, 0, 32);
    }
    ckernel::topk_rebuild<stable_sort>(0, (int)ascending, 0, 32, 5, true);
    tile_regs_commit();
    tile_regs_wait();
    cb_final_indices_transposed.reserve_back(1);
    pack_reconfig_data_format(cb_final_indices_transposed_id);
    pack_tile(2, cb_final_indices_transposed_id);
    tile_regs_release();
    cb_final_indices_transposed.push_back(1);

    transpose_and_pack(cb_final_indices_transposed_id, cb_out_indices_id, 1);

    cb_winning_group_scores.pop_front(tiles);
    cb_winning_group_indices.pop_front(tiles);
}

void normalize_scores(
    const uint32_t cb_gathered_sigmoid_id,
    const uint32_t cb_reduce_ones_scalar_id,
    const uint32_t cb_reduce_intermediate_id,
    const uint32_t cb_reciprocal_sums_id,
    const uint32_t cb_epsilon_scalar_id,
    const uint32_t cb_normalized_scores_id) {
    CircularBuffer cb_gathered_sigmoid(cb_gathered_sigmoid_id);
    CircularBuffer cb_reduce_ones_scalar(cb_reduce_ones_scalar_id);
    CircularBuffer cb_reduce_intermediate(cb_reduce_intermediate_id);
    CircularBuffer cb_reciprocal_sums(cb_reciprocal_sums_id);
    CircularBuffer cb_epsilon_scalar(cb_epsilon_scalar_id);
    CircularBuffer cb_normalized_scores(cb_normalized_scores_id);
    reconfig_data_format(cb_reduce_ones_scalar_id, cb_gathered_sigmoid_id);
    pack_reconfig_data_format(cb_normalized_scores_id);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(
        cb_gathered_sigmoid_id, cb_reduce_ones_scalar_id, cb_reduce_intermediate_id);

    cb_gathered_sigmoid.wait_front(1);
    cb_reduce_ones_scalar.wait_front(1);

    // 1. Sum row (experts) to get row vector of sums [1, 32]
    tile_regs_acquire();
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_gathered_sigmoid_id, cb_reduce_ones_scalar_id, 0, 0, 0);
    tile_regs_commit();
    reduce_uninit();

    // 2. Pack sums to intermediate to add epsilon
    tile_regs_wait();
    cb_reduce_intermediate.reserve_back(1);
    pack_reconfig_data_format(cb_reduce_intermediate_id);
    pack_tile(0, cb_reduce_intermediate_id);
    tile_regs_release();
    cb_reduce_intermediate.push_back(1);
    // 3. Add epsilon
    tile_regs_acquire();
    cb_epsilon_scalar.wait_front(1);
    cb_reduce_intermediate.wait_front(1);

    reconfig_data_format(cb_reduce_intermediate_id, cb_epsilon_scalar_id);
    pack_reconfig_data_format(cb_reciprocal_sums_id);

    add_bcast_scalar_init_short(cb_reduce_intermediate_id, cb_epsilon_scalar_id);
    add_tiles_bcast<BroadcastType::SCALAR>(cb_reduce_intermediate_id, cb_epsilon_scalar_id, 0, 0, 0);

    // 4. Recip
    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();

    cb_reduce_intermediate.pop_front(1);

    // 5. Pack reciprocals
    tile_regs_wait();
    cb_reciprocal_sums.reserve_back(1);
    pack_tile(0, cb_reciprocal_sums_id);
    cb_reciprocal_sums.push_back(1);
    tile_regs_release();

    // 6. Broadcast multiply
    tile_regs_acquire();
    cb_reciprocal_sums.wait_front(1);
    mul_bcast_cols_init_short(cb_gathered_sigmoid_id, cb_reciprocal_sums_id);
    mul_tiles_bcast<BroadcastType::COL>(
        cb_gathered_sigmoid_id, cb_reciprocal_sums_id, 0, 0, 0);  // tile *= 1/(sum_col(tile))
    tile_regs_commit();
    cb_reciprocal_sums.pop_front(1);
    cb_gathered_sigmoid.pop_front(1);

    tile_regs_wait();
    cb_normalized_scores.reserve_back(1);
    pack_reconfig_data_format(cb_normalized_scores_id);
    pack_tile(0, cb_normalized_scores_id);
    cb_normalized_scores.push_back(1);
    tile_regs_release();
}

void scale(
    const uint32_t cb_normalized_scores_id, const uint32_t cb_route_scale_scalar_id, const uint32_t cb_out_weights_id) {
    CircularBuffer cb_normalized_scores(cb_normalized_scores_id);
    CircularBuffer cb_route_scale_scalar(cb_route_scale_scalar_id);
    CircularBuffer cb_out_weights(cb_out_weights_id);
    cb_normalized_scores.wait_front(1);
    cb_route_scale_scalar.wait_front(1);
    mul_tiles_bcast_scalar_init_short(cb_normalized_scores_id, cb_route_scale_scalar_id);

    tile_regs_acquire();

    mul_tiles_bcast<BroadcastType::SCALAR>(cb_normalized_scores_id, cb_route_scale_scalar_id, 0, 0, 0);
    tile_regs_commit();

    cb_out_weights.reserve_back(1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_out_weights_id);
    pack_tile(0, cb_out_weights_id);
    cb_out_weights.push_back(1);
    tile_regs_release();

    cb_normalized_scores.pop_front(1);
}

}  // namespace blocks
