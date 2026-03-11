// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax Compute Kernel (dim=-1, width reduction)
// Stage 2: exp(input) per tile
// Later stages: 4-phase per row: max(REDUCE_ROW), sub+exp, sum(REDUCE_ROW)+recip, mul

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/exp.h"

constexpr uint32_t cb_input = 0;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_out = 16;

namespace NAMESPACE {

void MAIN {
    // Compile-time args
    constexpr uint32_t Ht = get_compile_time_arg_val(0);  // 1 for dim=-1
    constexpr uint32_t Wt = get_compile_time_arg_val(1);  // tile-cols per row
    constexpr uint32_t NC = get_compile_time_arg_val(2);  // always 1 (batch folded)
    constexpr uint32_t numeric_stable = get_compile_time_arg_val(3);

    // Runtime args
    const uint32_t num_units = get_arg_val<uint32_t>(0);

    // Hardware startup
    compute_kernel_hw_startup(cb_input, cb_scaler, cb_out);

    // Init copy and exp
    copy_tile_to_dst_init_short(cb_input);
    exp_tile_init();

    for (uint32_t unit = 0; unit < num_units; ++unit) {
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            // Wait for input tile
            cb_wait_front(cb_input, 1);

            // Acquire DST, copy tile, apply exp
            tile_regs_acquire();
            copy_tile(cb_input, 0, 0);
            exp_tile(0);
            tile_regs_commit();

            // Pack to output
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();

            // Pop the input tile
            cb_pop_front(cb_input, 1);
        }
    }
}

}  // namespace NAMESPACE
