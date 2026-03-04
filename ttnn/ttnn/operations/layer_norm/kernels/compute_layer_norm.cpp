// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Compute Kernel (Stage 2: Mean Subtract)
//
// Pass 1: Cross-tile accumulation (sum across Wt tiles), then reduce_row to get mean.
// Pass 2: Compute (x - mean) with COL broadcast, output to c_16.
// Pass 3: Consume and discard tiles from c_0.
//
// Compile-time args:
//   [0] num_rows_per_core : uint32  Tile-rows assigned to this core
//   [1] Wt               : uint32  Width in tiles
//   [2] has_gamma        : uint32  1 if gamma tensor is present
//   [3] has_beta         : uint32  1 if beta tensor is present

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    constexpr uint32_t cb_input = 0;    // c_0: streaming input tiles
    constexpr uint32_t cb_scaler = 1;   // c_1: 1/W scaler for reduce
    constexpr uint32_t cb_output = 16;  // c_16: output tiles
    constexpr uint32_t cb_mean = 24;    // c_24: mean result (persists within row)
    constexpr uint32_t cb_accum = 25;   // c_25: cross-tile accumulator

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    binary_op_init_common(cb_input, cb_scaler, cb_output);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // =====================================================================
        // Pass 1: Compute mean via cross-tile accumulation + reduce_row
        // =====================================================================
        for (uint32_t col = 0; col < Wt; ++col) {
            cb_wait_front(cb_input, onetile);

            if (col == 0) {
                // First tile: copy c_0 -> c_25 (accumulator)
                tile_regs_acquire();
                cb_reserve_back(cb_accum, onetile);
                copy_tile_init_with_dt(cb_input);
                copy_tile(cb_input, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_accum);
                tile_regs_release();

                cb_push_back(cb_accum, onetile);
            } else {
                // Subsequent tiles: add c_0 + c_25 -> c_25
                cb_wait_front(cb_accum, onetile);
                tile_regs_acquire();
                cb_reserve_back(cb_accum, onetile);
                add_tiles_init_with_dt(cb_input, cb_accum);
                add_tiles(cb_input, cb_accum, 0, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_accum);
                tile_regs_release();

                cb_pop_front(cb_accum, onetile);
                cb_push_back(cb_accum, onetile);
            }

            cb_pop_front(cb_input, onetile);
        }

        // Intra-tile reduce_row: cb_accum -> cb_mean (scaler has 1/W, so result is mean)
        // The reduce helper handles cb_wait/pop on cb_accum and cb_reserve/push on cb_mean.
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_accum, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::single());

        // =====================================================================
        // Pass 2: Compute (x - mean) and output to c_16
        // =====================================================================
        // cb_mean has 1 tile that persists across the entire row (don't pop per tile)
        cb_wait_front(cb_mean, onetile);

        for (uint32_t col = 0; col < Wt; ++col) {
            cb_wait_front(cb_input, onetile);
            cb_reserve_back(cb_output, onetile);

            tile_regs_acquire();
            sub_bcast_cols_init_short_with_dt(cb_input, cb_mean);
            sub_tiles_bcast<BroadcastType::COL>(cb_input, cb_mean, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_output);
            tile_regs_release();

            cb_pop_front(cb_input, onetile);
            cb_push_back(cb_output, onetile);
        }

        // Pop mean after all tiles in this row are processed
        cb_pop_front(cb_mean, onetile);

        // =====================================================================
        // Pass 3: Consume and discard tiles from c_0
        // =====================================================================
        for (uint32_t col = 0; col < Wt; ++col) {
            cb_wait_front(cb_input, onetile);
            cb_pop_front(cb_input, onetile);
        }
    }
}
