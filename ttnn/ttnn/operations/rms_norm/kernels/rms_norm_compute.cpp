// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Compute Kernel
// Runs on math RISC-V core, performs FPU/SFPU operations
//
// Stage 1 (data_pipeline): Identity passthrough
//   RM: tilize cb_in_rm -> cb_out, untilize cb_out -> cb_out_rm
//   TILE: copy cb_in -> cb_out tile by tile

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// CB indices
constexpr uint32_t cb_in_rm = 0;
constexpr uint32_t cb_in = 1;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_out_rm = 17;

// Compile-time args
constexpr uint32_t is_rm_input = get_compile_time_arg_val(0);
constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);
constexpr uint32_t Ht = get_compile_time_arg_val(3);
constexpr uint32_t NC = get_compile_time_arg_val(4);

void kernel_main() {
    uint32_t num_rows = get_arg_val<uint32_t>(0);

    if constexpr (is_rm_input) {
        // RM path: interleave tilize and untilize per tile-row to avoid
        // cb_out overflow (cb_out only holds Wt pages = one tile-row).
        compute_kernel_hw_startup(cb_in_rm, cb_in_rm, cb_out);

        for (uint32_t row = 0; row < num_rows; ++row) {
            // Phase 1: Tilize 1 tile-row of RM sticks -> cb_out
            compute_kernel_lib::tilize<
                Wt,
                cb_in_rm,
                cb_out,
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(1);

            // Phase 8: Untilize 1 tile-row from cb_out -> cb_out_rm
            compute_kernel_lib::untilize<
                Wt,
                cb_out,
                cb_out_rm,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock>(1);
        }
    } else {
        // TILE path: copy tiles from cb_in to cb_out
        compute_kernel_hw_startup(cb_in, cb_in, cb_out);

        copy_tile_to_dst_init_short(cb_in);

        for (uint32_t row = 0; row < num_rows; ++row) {
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_wait_front(cb_in, 1);
                tile_regs_acquire();
                copy_tile(cb_in, 0, 0);
                tile_regs_commit();

                cb_reserve_back(cb_out, 1);
                tile_regs_wait();
                pack_tile(0, cb_out);
                tile_regs_release();
                cb_push_back(cb_out, 1);

                cb_pop_front(cb_in, 1);
            }
        }
    }
}
