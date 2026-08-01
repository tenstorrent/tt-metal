// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// comm_skeleton probe 5: DEST SYNCHRONISATION WITH NO MATH.
//
// `tile_regs_acquire` / `tile_regs_commit` / `tile_regs_wait` / `tile_regs_release` is the MATH<->PACK
// handshake every eltwise call in the op pays exactly once, whatever it puts between them. Here it
// gets NOTHING between them, so the slope of ns vs N_ITERS is the handshake alone — the fixed toll
// on an eltwise chain, independent of the chain's arithmetic.
//
// `compute_kernel_hw_startup` is still called: without it the DEST section state is never
// configured and the handshake would be measured against an un-initialised machine rather than the
// one the op actually runs on. It is outside the loop, so it lands in the intercept, not the slope.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reg_api.h"

constexpr uint32_t CB_IN = get_compile_time_arg_val(0);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(1);

void kernel_main() {
    compute_kernel_hw_startup(CB_IN, CB_OUT);

    // RUNTIME (see cb_probe.cpp): one build serves the whole sweep.
    const uint32_t n_iters = get_arg_val<uint32_t>(0);
    for (uint32_t i = 0; i < n_iters; ++i) {
        tile_regs_acquire();
        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();
    }
}
