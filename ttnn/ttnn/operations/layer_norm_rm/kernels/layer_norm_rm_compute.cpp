// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
//
// Stage 1: Tilize, reduce mean, sub mean, untilize  -> x - mean(x)
//
// Architecture: Following the same pattern as the softmax kernel
// (softmax/device/kernels/attention/compute/softmax.cpp):
//   - Tilize helper: cb_in_rm -> cb_tilized
//   - Reduce helper (WaitUpfrontNoPop): cb_tilized -> cb_mean (tiles persist)
//   - Raw LLK sub_bcast_cols: cb_tilized - cb_mean -> cb_centered (manual tile loop)
//   - Manual pop of cb_tilized and cb_mean
//   - Untilize helper: cb_centered -> cb_out_rm
//
// Compile-time args:
//   [0] Wt - tiles per row (W / 32)
//
// Runtime args:
//   [0] num_blocks - number of tile-rows for this core

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

constexpr uint32_t cb_in_rm = 0;
constexpr uint32_t cb_tilized = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_out_rm = 16;
constexpr uint32_t cb_mean = 24;
constexpr uint32_t cb_centered = 25;

constexpr uint32_t Wt = get_compile_time_arg_val(0);

void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in_rm, cb_scaler, cb_out_rm);

    if (num_blocks == 0) {
        return;
    }

    constexpr uint32_t ndst = compute_kernel_lib::DEST_AUTO_LIMIT;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Phase 1: Tilize (cb_in_rm -> cb_tilized)
        compute_kernel_lib::tilize<
            cb_in_rm,
            cb_tilized,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(Wt, 1);

        // Phase 2: Reduce SUM for mean -- tiles persist in cb_tilized (WaitUpfrontNoPop)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: Subtract mean using raw LLK calls (following softmax pattern)
        // cb_tilized: Wt tiles still present (from WaitUpfrontNoPop)
        // cb_mean: 1 tile just pushed by reduce
        reconfig_data_format_srcb(cb_mean);
        cb_wait_front(cb_mean, 1);
        sub_bcast_cols_init_short(cb_tilized, cb_mean);

        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
            uint32_t chunk = (wt + ndst <= Wt) ? ndst : (Wt - wt);
            tile_regs_acquire();
            for (uint32_t i = 0; i < chunk; ++i) {
                sub_tiles_bcast_cols(cb_tilized, cb_mean, wt + i, 0, i);
            }
            cb_reserve_back(cb_centered, chunk);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < chunk; ++i) {
                pack_tile(i, cb_centered);
            }
            tile_regs_release();
            cb_push_back(cb_centered, chunk);
        }
        cb_pop_front(cb_tilized, Wt);
        cb_pop_front(cb_mean, 1);

        // Phase 4: Untilize (cb_centered -> cb_out_rm)
        compute_kernel_lib::untilize<
            Wt,
            cb_centered,
            cb_out_rm,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock>(1);
    }
}
