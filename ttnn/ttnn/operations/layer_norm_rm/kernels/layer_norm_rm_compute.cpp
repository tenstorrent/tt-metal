// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Stage 2 (reduce_mean): tilize input, reduce row for mean, broadcast mean to output, untilize.

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace NAMESPACE {

void MAIN {
    // ========== Compile-time args ==========
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t has_beta = get_compile_time_arg_val(2);

    // ========== Runtime args ==========
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    // ========== CB IDs ==========
    constexpr uint32_t cb_input_rm = 0;       // c_0: RM sticks from reader
    constexpr uint32_t cb_tilized = 1;        // c_1: tilized input (2*Wt for reuse)
    constexpr uint32_t cb_reduce_scaler = 8;  // c_8: reduce scaler (1/W)
    constexpr uint32_t cb_out_rm = 16;        // c_16: untilized output for writer
    constexpr uint32_t cb_mean = 24;          // c_24: mean tile (1 tile)
    constexpr uint32_t cb_norm = 29;          // c_29: reused as broadcast output buffer

    // ========== Hardware init ==========
    compute_kernel_hw_startup(cb_input_rm, cb_reduce_scaler, cb_out_rm);

    // ========== Main loop ==========
    for (uint32_t tr = 0; tr < num_tile_rows; tr++) {
        // Phase 1: Tilize c_0 -> c_1 (Wt tiles)
        compute_kernel_lib::tilize<Wt, cb_input_rm, cb_tilized>(1);

        // Phase 2: Reduce row for mean: c_1 -> c_24
        // WaitUpfrontNoPop: c_1 tiles persist (not popped) -- but we don't need them for Stage 2
        // Actually for Stage 2 we don't reuse c_1 after reduce, so just use default WaitAndPopPerTile
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_tilized, cb_reduce_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // Now c_24 has 1 mean tile (valid data in Col0)
        // Broadcast mean column across Wt tiles using unary_bcast<COL>

        // Wait for the mean tile
        cb_wait_front(cb_mean, 1);

        // Reserve Wt tiles in cb_norm for the broadcast result
        cb_reserve_back(cb_norm, Wt);

        // Initialize unary broadcast for COL
        unary_bcast_init<BroadcastType::COL>(cb_mean, cb_norm);

        for (uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            // Broadcast mean tile's column across all columns -> DST[0]
            unary_bcast<BroadcastType::COL>(cb_mean, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_norm);
            tile_regs_release();
        }

        cb_push_back(cb_norm, Wt);
        cb_pop_front(cb_mean, 1);

        // Phase 10: Untilize cb_norm -> c_16
        compute_kernel_lib::untilize<Wt, cb_norm, cb_out_rm>(1);
    }
}

}  // namespace NAMESPACE
