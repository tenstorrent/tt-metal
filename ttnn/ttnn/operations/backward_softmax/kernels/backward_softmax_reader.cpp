// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for backward_softmax (VJP of softmax).
//
// Two input-buffering strategies (Refinement 2):
//
//   STRATEGY_IS_WHOLE_ROW == 1 (WHOLE_ROW_DB or WHOLE_ROW_SB):
//     The compute kernel caches each lane's dy and y rows in L1 across both
//     passes. The reader fetches each tile EXACTLY ONCE per output tile —
//     a single pass per lane. With DB-sized CBs (cb_*_pages == 2 *
//     reduce_dim_tiles), the reader can prefetch lane N+1 into the second
//     half while compute is mid-lane on lane N.
//
//   STRATEGY_IS_WHOLE_ROW == 0 (PER_TILE_STREAM, fallback):
//     Phase-0 behavior. The reader fetches each lane's tiles TWICE — once
//     for pass-1 (mul + reduce) and once for pass-2 (sub + mul). CBs are
//     sized small (a few pages); used when the reduce dimension is too
//     large to cache in L1.
//
// Tiles are pushed in lockstep across cb_grad_output and cb_output: one dy
// tile and one y tile per inner iteration. Lockstep is required because the
// pass-1 mul helper (and pass-2 mul) consume one tile from each input per
// step.
//
// The scaler tile (1.0 for SUM) is pushed exactly once, at startup, before
// the lane loop begins. The reduce LLK waits on it but never pops, so the
// same tile serves every lane and every pass.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // RT args (per-core under multi-core distribution): the lane partition
    // [start_lane, start_lane + num_lanes) is the contiguous slice this core
    // owns.
    uint32_t grad_output_addr = get_arg_val<uint32_t>(0);
    uint32_t output_addr = get_arg_val<uint32_t>(1);
    uint32_t start_lane = get_arg_val<uint32_t>(2);
    uint32_t num_lanes = get_arg_val<uint32_t>(3);

    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(1);
    constexpr uint32_t DIM_IS_W = get_compile_time_arg_val(2);  // 1 = dim=-1, 0 = dim=-2
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t STRATEGY_IS_WHOLE_ROW = get_compile_time_arg_val(5);  // 1 = whole row, 0 = per-tile streaming
    constexpr auto grad_output_args = TensorAccessorArgs<6>();
    constexpr auto output_args = TensorAccessorArgs<grad_output_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_grad_output = 0;
    constexpr uint32_t cb_output = 1;
    constexpr uint32_t cb_scaler = 2;

    // ---- Push the scaler tile (once per program) ----
    // SUM-reduce needs scaler = 1.0 (SUM_AND_MAX_REDUCE_FACTOR=1 by default).
    // The pool-type-aware overload picks the correct fill pattern:
    //   REDUCE_ROW + SUM → matmul-path col-0 fill
    //   REDUCE_COL + SUM → reduce-path row-0 fill
    // Both place the scalar at face[0][0] of the resulting scaler tile, which
    // is what the reduce helpers and `sub<COL/ROW>` consume later.
    if constexpr (DIM_IS_W) {
        dataflow_kernel_lib::
            calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    } else {
        dataflow_kernel_lib::
            calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>();
    }

    // ---- Tile streaming setup ----
    const uint32_t tile_bytes = get_tile_size(cb_grad_output);
    const auto grad_output_accessor = TensorAccessor(grad_output_args, grad_output_addr, tile_bytes);
    const auto output_accessor = TensorAccessor(output_args, output_addr, tile_bytes);

    // Lanes per (n, c) plane:
    //   dim=-1: one lane per H-tile row    → reduce_lanes_per_nc = Ht
    //   dim=-2: one lane per W-tile column → reduce_lanes_per_nc = Wt
    constexpr uint32_t reduce_lanes_per_nc = DIM_IS_W ? Ht : Wt;
    constexpr uint32_t reduce_dim_tiles = DIM_IS_W ? Wt : Ht;

    // Number of times the reader pushes each lane's tile set:
    //   whole-row strategies: once (compute reads from L1 across both passes).
    //   per-tile-stream:      twice (each pass re-reads from DRAM).
    constexpr uint32_t NUM_PASSES = STRATEGY_IS_WHOLE_ROW ? 1u : 2u;

    for (uint32_t lane_idx = 0; lane_idx < num_lanes; ++lane_idx) {
        const uint32_t lane = start_lane + lane_idx;
        const uint32_t nc = lane / reduce_lanes_per_nc;
        const uint32_t idx = lane % reduce_lanes_per_nc;
        const uint32_t tile_id_origin = nc * Ht * Wt;

        for (uint32_t pass = 0; pass < NUM_PASSES; ++pass) {
            for (uint32_t t = 0; t < reduce_dim_tiles; ++t) {
                uint32_t tile_id;
                if constexpr (DIM_IS_W) {
                    // Lane = one row of tiles within (n,c,h)-slice.
                    tile_id = tile_id_origin + idx * Wt + t;
                } else {
                    // Lane = one column of tiles within (n,c,w)-slice.
                    tile_id = tile_id_origin + t * Wt + idx;
                }

                cb_reserve_back(cb_grad_output, 1);
                cb_reserve_back(cb_output, 1);

                const uint32_t l1_grad_output = get_write_ptr(cb_grad_output);
                const uint32_t l1_output = get_write_ptr(cb_output);

                noc_async_read_tile(tile_id, grad_output_accessor, l1_grad_output);
                noc_async_read_tile(tile_id, output_accessor, l1_output);
                noc_async_read_barrier();

                cb_push_back(cb_grad_output, 1);
                cb_push_back(cb_output, 1);
            }
        }
    }
}
