// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// double_buffer compute (unpack/math/pack TRISCs) — IDENTICAL for both variants.
//
// The middle stage of the pipeline: pull one tile from cb_in, apply the unary
// op, push it to cb_out. The unary op is relu, applied `compute_passes` times.
// relu is idempotent (relu(relu(x)) == relu(x)), so the number of passes is a
// pure COMPUTE-COST knob: the result is always relu(x) no matter how many
// passes, but each pass is real SFPU work on the math engine. That lets the
// example dial how heavy the compute stage is relative to the DRAM read/write,
// which is exactly what decides how much there is to overlap.
//
// This kernel is unchanged between single- and double-buffered runs. It waits
// for 1 tile, computes it, packs 1 tile — the pipelining comes entirely from
// how deep cb_in / cb_out are (set in the program descriptor), never from here.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/relu.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(0);
    constexpr uint32_t compute_passes = get_compile_time_arg_val(1);

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    init_sfpu(cb_in, cb_out);
    relu_tile_init();

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t t = 0; t < num_tiles; ++t) {
            cb_wait_front(cb_in, 1);
            cb_reserve_back(cb_out, 1);

            tile_regs_acquire();
            copy_tile(cb_in, 0, 0);
            for (uint32_t p = 0; p < compute_passes; ++p) {
                relu_tile(0);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();

            cb_pop_front(cb_in, 1);
            cb_push_back(cb_out, 1);
        }
    }
}
