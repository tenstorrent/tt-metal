// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Group Norm - Compute Kernel
// Phase 0: Tilize (cb_input_rm -> cb_tilized, persistent)
// Phase 4: Normalize pass (identity copy for Stage 1)
// Phase 6: Untilize (cb_normalized -> cb_output_rm)

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    // ========== COMPILE-TIME ARGS ==========
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Ct = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2);
    constexpr uint32_t Ct_g = get_compile_time_arg_val(3);
    constexpr uint32_t num_samples = get_compile_time_arg_val(4);

    // ========== CB INDICES ==========
    constexpr uint32_t cb_input_rm = 0;
    constexpr uint32_t cb_tilized = 1;
    constexpr uint32_t cb_normalized = 16;
    constexpr uint32_t cb_output_rm = 17;

    // ========== HW STARTUP ==========
    // First operation is tilize: srcA=srcB=cb_input_rm, pack=cb_tilized
    compute_kernel_hw_startup(cb_input_rm, cb_tilized);

    for (uint32_t n = 0; n < num_samples; ++n) {
        // ========== PHASE 0: TILIZE ==========
        // Tilize Ht blocks of Ct tiles each from cb_input_rm -> cb_tilized
        compute_kernel_lib::tilize<cb_input_rm, cb_tilized>(Ct, Ht);

        // Wait for all tilized data (persistent CB)
        cb_wait_front(cb_tilized, Ht * Ct);

        // ========== PHASE 4: IDENTITY COPY (Stage 1) ==========
        // For each tile-row, copy Ct tiles from cb_tilized to cb_normalized
        // cb_tilized is persistent -- we use indexed tile access via copy_tile
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // Reserve space in output
            cb_reserve_back(cb_normalized, Ct);

            for (uint32_t c = 0; c < Ct; ++c) {
                // Acquire DST, copy tile from cb_tilized at index ht*Ct+c
                tile_regs_acquire();
                copy_tile_to_dst_init_short(cb_tilized);
                copy_tile(cb_tilized, ht * Ct + c, 0);  // tile_idx, dst_idx
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_normalized);
                tile_regs_release();
            }

            cb_push_back(cb_normalized, Ct);

            // ========== PHASE 6: UNTILIZE ==========
            // Untilize this tile-row from cb_normalized -> cb_output_rm
            compute_kernel_lib::untilize<Ct, cb_normalized, cb_output_rm>(1);
        }

        // Pop all persistent tiles after processing sample
        cb_pop_front(cb_tilized, Ht * Ct);
    }
}
