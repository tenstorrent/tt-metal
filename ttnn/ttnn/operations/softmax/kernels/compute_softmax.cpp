// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax - Compute Kernel
// Full softmax for dim=-1 (REDUCE_ROW) and dim=-2 (REDUCE_COL)
// 3-phase pipeline per row/col:
//   Phase 1: reduce<MAX> to find row/col max -> c_3
//   Phase 2: sub_bcast(input, max) + exp -> c_4 (all tiles); reduce<SUM> + recip -> c_5
//   Phase 3: sub_bcast(input, max) + exp -> c_4 -> mul_bcast(c_4, 1/sum) -> c_16

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

constexpr uint32_t cb_input = tt::CBIndex::c_0;
constexpr uint32_t cb_scaler = tt::CBIndex::c_1;
constexpr uint32_t cb_max = tt::CBIndex::c_3;
constexpr uint32_t cb_exp = tt::CBIndex::c_4;
constexpr uint32_t cb_recip_sum = tt::CBIndex::c_5;
constexpr uint32_t cb_output = tt::CBIndex::c_16;

void kernel_main() {
    constexpr uint32_t num_rows_or_cols = get_compile_time_arg_val(0);
    constexpr uint32_t inner_dim = get_compile_time_arg_val(1);

    // Initialize hardware
    compute_kernel_hw_startup(cb_input, cb_scaler, cb_output);

    // Wait for persistent scaler tile (pushed once by reader, never popped)
    cb_wait_front(cb_scaler, 1);

    for (uint32_t rc = 0; rc < num_rows_or_cols; ++rc) {
        // ============================================================
        // Phase 1: Find max along inner dimension using reduce helper
        // ============================================================
        compute_kernel_lib::reduce<
            REDUCE_OP,
            REDUCE_DIM,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
            cb_input, cb_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::row(inner_dim));

        cb_wait_front(cb_max, 1);

        // ============================================================
        // Phase 2a: Compute exp(x - max) for all tiles, pack to c_4
        // ============================================================
        for (uint32_t t = 0; t < inner_dim; ++t) {
            cb_wait_front(cb_input, 1);

            tile_regs_acquire();
#ifdef DIM_W
            sub_bcast_cols_init_short(cb_input, cb_max);
            sub_tiles_bcast<BroadcastType::COL>(cb_input, cb_max, 0, 0, 0);
#endif
#ifdef DIM_H
            sub_bcast_rows_init_short(cb_input, cb_max);
            sub_tiles_bcast<BroadcastType::ROW>(cb_input, cb_max, 0, 0, 0);
#endif
            exp_tile_init<false>();
            exp_tile<false>(0);

            cb_reserve_back(cb_exp, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_exp);
            tile_regs_release();
            cb_push_back(cb_exp, 1);

            cb_pop_front(cb_input, 1);
        }

        // ============================================================
        // Phase 2b: reduce<SUM> over exp tiles + recip -> c_5
        // All inner_dim exp tiles are now in c_4.
        // reduce helper consumes them and produces 1 recip_sum tile.
        // ============================================================
        compute_kernel_lib::reduce<
            PoolType::SUM,
            REDUCE_DIM,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            cb_exp,
            cb_scaler,
            cb_recip_sum,
            compute_kernel_lib::ReduceInputBlockShape::row(inner_dim),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t dst_idx) {
                recip_tile_init();
                recip_tile(dst_idx);
            });

        cb_wait_front(cb_recip_sum, 1);

        // ============================================================
        // Phase 3: Recompute exp(x - max) * (1/sum) -> output
        // ============================================================
        for (uint32_t t = 0; t < inner_dim; ++t) {
            cb_wait_front(cb_input, 1);

            // Compute exp(x - max), pack to c_4
            tile_regs_acquire();
#ifdef DIM_W
            sub_bcast_cols_init_short(cb_input, cb_max);
            sub_tiles_bcast<BroadcastType::COL>(cb_input, cb_max, 0, 0, 0);
#endif
#ifdef DIM_H
            sub_bcast_rows_init_short(cb_input, cb_max);
            sub_tiles_bcast<BroadcastType::ROW>(cb_input, cb_max, 0, 0, 0);
#endif
            exp_tile_init<false>();
            exp_tile<false>(0);

            cb_reserve_back(cb_exp, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_exp);
            tile_regs_release();
            cb_push_back(cb_exp, 1);

            cb_pop_front(cb_input, 1);

            // mul_bcast: exp(x-max) * (1/sum) -> output
            cb_wait_front(cb_exp, 1);
            cb_reserve_back(cb_output, 1);

            tile_regs_acquire();
#ifdef DIM_W
            mul_bcast_cols_init_short(cb_exp, cb_recip_sum);
            mul_tiles_bcast<BroadcastType::COL>(cb_exp, cb_recip_sum, 0, 0, 0);
#endif
#ifdef DIM_H
            mul_bcast_rows_init_short(cb_exp, cb_recip_sum);
            mul_tiles_bcast<BroadcastType::ROW>(cb_exp, cb_recip_sum, 0, 0, 0);
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_output);
            tile_regs_release();

            cb_pop_front(cb_exp, 1);
            cb_push_back(cb_output, 1);
        }

        // Free max and recip_sum tiles for next row/col
        cb_pop_front(cb_max, 1);
        cb_pop_front(cb_recip_sum, 1);
    }
}
