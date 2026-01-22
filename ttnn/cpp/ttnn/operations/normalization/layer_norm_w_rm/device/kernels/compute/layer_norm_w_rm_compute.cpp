// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {

void MAIN {
    // ============================================================
    // Compile-time args
    // ============================================================
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    // ============================================================
    // CB IDs
    // ============================================================
    constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;      // Input RM sticks
    constexpr uint32_t cb_gamma_rm = tt::CBIndex::c_10;  // Gamma RM sticks
    constexpr uint32_t cb_beta_rm = tt::CBIndex::c_12;   // Beta RM sticks
    constexpr uint32_t cb_out_rm = tt::CBIndex::c_16;    // Output RM sticks

    // ============================================================
    // Initialize compute kernel hardware
    // ============================================================
    copy_tile_init(cb_in_rm);

    // ============================================================
    // STUB: Wait for gamma and beta (read once at program start)
    // ============================================================
    // In real implementation, these would be tilized and used for computation
    // For stub, we just wait to maintain CB sync
    cb_wait_front(cb_gamma_rm, Wt);
    cb_wait_front(cb_beta_rm, Wt);
    // Note: Do NOT pop gamma/beta - they persist for program lifetime

    // ============================================================
    // STUB: Passthrough input to output (per tile-row)
    // ============================================================
    // Real implementation would do 11 phases (tilize, mean, centralize, square,
    // variance, add_eps+rsqrt, multiply rsqrt, gamma multiply, beta add, untilize)
    // Stub just copies RM pages from input to output to verify CB sync

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        // Wait for input (Wt pages of RM sticks)
        cb_wait_front(cb_in_rm, Wt);

        // Reserve output space
        cb_reserve_back(cb_out_rm, Wt);

        // Copy each RM page (stub passthrough using copy_tile on RM data)
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            tile_regs_acquire();
            copy_tile(cb_in_rm, wt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out_rm);
            tile_regs_release();
        }

        // Push output and pop input
        cb_push_back(cb_out_rm, Wt);
        cb_pop_front(cb_in_rm, Wt);
    }
}

}  // namespace NAMESPACE
