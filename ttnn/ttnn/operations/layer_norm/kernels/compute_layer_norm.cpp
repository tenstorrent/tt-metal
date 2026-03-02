// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm - Compute Kernel
// Stage 2: mean_reduction - reduces input row to mean (REDUCE_ROW),
// then subtracts mean from each input tile (COL broadcast).

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t num_rows = get_compile_time_arg_val(1);
    constexpr uint32_t gamma_has_value = get_compile_time_arg_val(2);
    constexpr uint32_t beta_has_value = get_compile_time_arg_val(3);

    constexpr uint32_t cb_input = 0;
    constexpr uint32_t cb_scaler = 1;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t cb_mean = 24;

    // CRITICAL: Must call binary_op_init_common before any reduce/bcast operations.
    // This sets up the hardware for binary (two-input) operations.
    binary_op_init_common(cb_input, cb_scaler, cb_out);

    // Wait for scaler tile (filled once by reader, used every row)
    cb_wait_front(cb_scaler, 1);

    for (uint32_t row = 0; row < num_rows; ++row) {
        // Wait for reader to push Wt input tiles for this row
        cb_wait_front(cb_input, Wt);

        // ====== Phase 1: Reduce input row to mean (REDUCE_ROW) ======
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_input, cb_scaler, cb_mean);

        cb_reserve_back(cb_mean, 1);
        tile_regs_acquire();
        for (uint32_t w = 0; w < Wt; ++w) {
            reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_input, cb_scaler, w, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_mean);
        tile_regs_release();
        cb_push_back(cb_mean, 1);

        // Reset packer state after reduce before bcast
        reduce_uninit();

        // ====== Phase 2: Subtract mean from each input tile (COL broadcast) ======
        cb_wait_front(cb_mean, 1);

        // Full init for sub bcast cols (configures unpack, math, AND packer for cb_out)
        init_bcast<EltwiseBinaryType::ELWSUB, BroadcastType::COL>(cb_input, cb_mean, cb_out);

        for (uint32_t w = 0; w < Wt; ++w) {
            cb_reserve_back(cb_out, 1);
            tile_regs_acquire();
            sub_tiles_bcast<BroadcastType::COL>(cb_input, cb_mean, w, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }

        // Pop input and mean for this row
        cb_pop_front(cb_input, Wt);
        cb_pop_front(cb_mean, 1);
    }

    // Pop scaler (filled once, used for all rows)
    cb_pop_front(cb_scaler, 1);
}
