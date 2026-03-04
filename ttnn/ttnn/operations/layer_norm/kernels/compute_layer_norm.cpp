// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Compute Kernel (Stage 3: Variance)
//
// Pass 1: Cross-tile accumulation (sum across Wt tiles), then reduce_row to get mean.
// Pass 2: Compute variance: (x-mean)^2 accumulated, reduce_row, add eps, rsqrt -> cb_var.
// Pass 3: Output (x - mean) * rsqrt(var + eps) to c_16.
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
    constexpr uint32_t cb_eps = 2;      // c_2: epsilon tile
    constexpr uint32_t cb_output = 16;  // c_16: output tiles
    constexpr uint32_t cb_mean = 24;    // c_24: mean result (persists within row)
    constexpr uint32_t cb_accum = 25;   // c_25: cross-tile accumulator
    constexpr uint32_t cb_var = 26;     // c_26: variance / rsqrt(var+eps)
    constexpr uint32_t cb_tmp = 27;     // c_27: scratch tile

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
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_accum, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::single());

        // =====================================================================
        // Pass 2: Compute variance -> rsqrt(var + eps) into cb_var
        // =====================================================================
        cb_wait_front(cb_mean, onetile);

        for (uint32_t col = 0; col < Wt; ++col) {
            cb_wait_front(cb_input, onetile);

            // Step 1: (x - mean) -> cb_tmp
            tile_regs_acquire();
            cb_reserve_back(cb_tmp, onetile);
            sub_bcast_cols_init_short_with_dt(cb_input, cb_mean);
            sub_tiles_bcast<BroadcastType::COL>(cb_input, cb_mean, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_tmp);
            tile_regs_release();

            cb_push_back(cb_tmp, onetile);

            // Step 2: (x - mean)^2 via mul_tiles(cb_tmp, cb_tmp)
            // cb_tmp has 1 page with (x-mean). Reading same tile index twice for squaring.
            cb_wait_front(cb_tmp, onetile);
            tile_regs_acquire();
            mul_tiles_init_with_dt(cb_tmp, cb_tmp);
            mul_tiles(cb_tmp, cb_tmp, 0, 0, dst0);
            tile_regs_commit();

            // Unpack from cb_tmp is complete after commit. Pop it now to free the slot.
            cb_pop_front(cb_tmp, onetile);

            // Step 3: Cross-tile accumulation of (x-mean)^2
            if (col == 0) {
                // First tile: pack squared result directly to cb_accum
                cb_reserve_back(cb_accum, onetile);
                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_accum);
                tile_regs_release();
                cb_push_back(cb_accum, onetile);
            } else {
                // Pack squared result to cb_tmp, then add with cb_accum
                cb_reserve_back(cb_tmp, onetile);
                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_tmp);
                tile_regs_release();
                cb_push_back(cb_tmp, onetile);

                // Add cb_tmp((x-mean)^2) + cb_accum(running sum) -> cb_accum
                cb_wait_front(cb_tmp, onetile);
                cb_wait_front(cb_accum, onetile);
                tile_regs_acquire();
                cb_reserve_back(cb_accum, onetile);
                add_tiles_init_with_dt(cb_tmp, cb_accum);
                add_tiles(cb_tmp, cb_accum, 0, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_accum);
                tile_regs_release();

                cb_pop_front(cb_tmp, onetile);
                cb_pop_front(cb_accum, onetile);
                cb_push_back(cb_accum, onetile);
            }

            cb_pop_front(cb_input, onetile);
        }

        // Intra-tile reduce_row: cb_accum -> cb_var (scaler has 1/W)
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_accum, cb_scaler, cb_var, compute_kernel_lib::ReduceInputBlockShape::single());

        // Add epsilon and rsqrt: cb_var = 1/sqrt(var + eps)
        cb_wait_front(cb_var, onetile);
        cb_wait_front(cb_eps, onetile);
        tile_regs_acquire();
        cb_reserve_back(cb_var, onetile);
        add_bcast_scalar_init_short_with_dt(cb_var, cb_eps);
        add_tiles_bcast<BroadcastType::SCALAR>(cb_var, cb_eps, 0, 0, dst0);

        // rsqrt in-place in DST: 1/sqrt(var + eps)
        rsqrt_tile_init();
        rsqrt_tile(dst0);

        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_var);
        tile_regs_release();

        cb_pop_front(cb_var, onetile);
        cb_push_back(cb_var, onetile);
        // NOTE: cb_eps persists across rows (don't pop)

        // =====================================================================
        // Pass 3: Output (x - mean) * rsqrt(var + eps)
        // =====================================================================
        // cb_mean and cb_var persist from previous passes
        cb_wait_front(cb_var, onetile);

        for (uint32_t col = 0; col < Wt; ++col) {
            cb_wait_front(cb_input, onetile);

            // (x - mean) -> cb_tmp
            tile_regs_acquire();
            cb_reserve_back(cb_tmp, onetile);
            sub_bcast_cols_init_short_with_dt(cb_input, cb_mean);
            sub_tiles_bcast<BroadcastType::COL>(cb_input, cb_mean, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_tmp);
            tile_regs_release();

            cb_push_back(cb_tmp, onetile);

            // (x - mean) * rsqrt(var + eps) -> cb_output (COL broadcast from cb_var)
            cb_wait_front(cb_tmp, onetile);
            cb_reserve_back(cb_output, onetile);
            tile_regs_acquire();
            mul_bcast_cols_init_short_with_dt(cb_tmp, cb_var);
            mul_tiles_bcast_cols(cb_tmp, cb_var, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_output);
            tile_regs_release();

            cb_pop_front(cb_tmp, onetile);
            cb_pop_front(cb_input, onetile);
            cb_push_back(cb_output, onetile);
        }

        // Pop mean and var at end of row
        cb_pop_front(cb_mean, onetile);
        cb_pop_front(cb_var, onetile);
    }
}
