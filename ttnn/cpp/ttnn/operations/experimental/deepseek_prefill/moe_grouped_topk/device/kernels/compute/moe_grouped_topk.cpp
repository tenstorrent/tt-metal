// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_COL

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"

namespace blocks {
void sigmoid(uint32_t cb_in_scores, uint32_t cb_sigmoid_scores, uint32_t width_tiles) {
    for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
        cb_wait_front(cb_in_scores, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_in_scores);
        copy_tile_to_dst_init_short(cb_in_scores);
        copy_tile(cb_in_scores, 0, 0);
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
        pack_reconfig_data_format(cb_biased_scores);
        pack_tile(0, cb_biased_scores);
        tile_regs_release();
        cb_push_back(cb_biased_scores, 1);
    }
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

    tile_regs_acquire();
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_gathered_sigmoid, cb_reduce_ones_scalar, 0, 0, 0);
    tile_regs_commit();
    reduce_uninit();

    tile_regs_wait();
    cb_reserve_back(cb_reduce_intermediate, 1);
    pack_reconfig_data_format(cb_reduce_intermediate);
    pack_tile(0, cb_reduce_intermediate);
    tile_regs_release();
    cb_push_back(cb_reduce_intermediate, 1);

    tile_regs_acquire();
    cb_wait_front(cb_epsilon_scalar, 1);
    cb_wait_front(cb_reduce_intermediate, 1);

    reconfig_data_format(cb_reduce_intermediate, cb_epsilon_scalar);
    pack_reconfig_data_format(cb_reciprocal_sums);

    add_bcast_scalar_init_short(cb_reduce_intermediate, cb_epsilon_scalar);
    add_tiles_bcast<BroadcastType::SCALAR>(cb_reduce_intermediate, cb_epsilon_scalar, 0, 0, 0);

    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();

    cb_pop_front(cb_reduce_intermediate, 1);

    tile_regs_wait();
    cb_reserve_back(cb_reciprocal_sums, 1);
    pack_tile(0, cb_reciprocal_sums);
    cb_push_back(cb_reciprocal_sums, 1);
    tile_regs_release();

    tile_regs_acquire();
    cb_wait_front(cb_reciprocal_sums, 1);
    mul_bcast_cols_init_short(cb_gathered_sigmoid, cb_reciprocal_sums);
    mul_tiles_bcast<BroadcastType::COL>(cb_gathered_sigmoid, cb_reciprocal_sums, 0, 0, 0);
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
    pack_reconfig_data_format(cb_out_weights);
    pack_tile(0, cb_out_weights);
    cb_push_back(cb_out_weights, 1);
    tile_regs_release();

    cb_pop_front(cb_normalized_scores, 1);
}

}  // namespace blocks

void kernel_main() {
    constexpr uint32_t cb_in_scores = get_named_compile_time_arg_val("cb_in_scores");
    constexpr uint32_t cb_in_bias = get_named_compile_time_arg_val("cb_in_bias");
    constexpr uint32_t cb_sigmoid_scores = get_named_compile_time_arg_val("cb_sigmoid_scores");
    constexpr uint32_t cb_biased_scores = get_named_compile_time_arg_val("cb_biased_scores");
    constexpr uint32_t cb_out_weights = get_named_compile_time_arg_val("cb_out_weights");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t cb_reduce_intermediate = get_named_compile_time_arg_val("cb_reduce_intermediate");
    constexpr uint32_t cb_reduce_ones_scalar = get_named_compile_time_arg_val("cb_reduce_ones_scalar");
    constexpr uint32_t cb_epsilon_scalar = get_named_compile_time_arg_val("cb_epsilon_scalar");
    constexpr uint32_t cb_route_scale_scalar = get_named_compile_time_arg_val("cb_route_scale_scalar");
    constexpr uint32_t cb_normalized_scores = get_named_compile_time_arg_val("cb_normalized_scores");
    constexpr uint32_t cb_reciprocal_sums = get_named_compile_time_arg_val("cb_reciprocal_sums");
    constexpr uint32_t cb_gathered_sigmoid = get_named_compile_time_arg_val("cb_gathered_sigmoid");

    const uint32_t start_height_tile = get_arg_val<uint32_t>(0);
    const uint32_t end_height_tile = get_arg_val<uint32_t>(1);
    binary_op_init_common(cb_in_scores, cb_in_bias, cb_biased_scores);

    for (uint32_t height_tile = start_height_tile; height_tile < end_height_tile; height_tile++) {
        blocks::sigmoid(cb_in_scores, cb_sigmoid_scores, width_tiles);
        blocks::add_bias(cb_sigmoid_scores, cb_in_bias, cb_biased_scores, width_tiles);
        // Writer performs iterative topk + gather, then pushes cb_gathered_sigmoid
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
