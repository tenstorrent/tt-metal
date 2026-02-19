// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Standalone SwiGLU SFPU test kernel.
//
// Reads gate tiles from CB0 and up tiles from CB1, runs SwiGLU SFPU on the
// MATH thread, and packs the result to CB16.

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

#include "swiglu_sfpu.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr auto cb_gate = tt::CBIndex::c_0;
    constexpr auto cb_up = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // Init unpacker AND packer (init_sfpu configures both)
    init_sfpu(cb_gate, cb_out);

    // Init binary SFPU for SwiGLU
    MATH((llk_math_eltwise_binary_sfpu_swiglu_init<true>()));

    for (uint32_t t = 0; t < num_tiles; t++) {
        tile_regs_acquire();

        cb_wait_front(cb_gate, 1);
        cb_wait_front(cb_up, 1);

        // Copy gate tile from CB0 to dest[0]
        copy_tile(cb_gate, 0, 0);

        // Copy up tile from CB1 to dest[1]
        copy_tile(cb_up, 0, 1);

        // Run SwiGLU on MATH thread: result -> dest[0]
        MATH((llk_math_eltwise_binary_sfpu_swiglu<true, false>(0, 1, 0)));

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        tile_regs_release();

        cb_pop_front(cb_gate, 1);
        cb_pop_front(cb_up, 1);
    }
}
