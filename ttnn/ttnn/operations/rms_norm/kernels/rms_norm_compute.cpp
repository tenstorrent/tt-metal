// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Compute Kernel
// Stage 1: tilize (RM) + copy_tiles + untilize (RM)

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_x = 24;

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t Wt = get_arg_val<uint32_t>(1);
    const uint32_t origin_w = get_arg_val<uint32_t>(2);

    if (num_rows == 0) {
        return;
    }

#if IS_INPUT_RM
    // RM path: tilize cb_in -> cb_x, copy cb_x -> cb_x (reuse as output staging), untilize cb_x -> cb_out
    compute_kernel_hw_startup(cb_in, cb_x);

    for (uint32_t row = 0; row < num_rows; ++row) {
        // Phase 1: Tilize RM input -> cb_x
        compute_kernel_lib::tilize<cb_in, cb_x>(Wt, 1);

        // Phase 8: Untilize cb_x -> cb_out (RM output)
        compute_kernel_lib::untilize<RMS_Wt, cb_x, cb_out>(1);
    }
#else
    // TILE path: copy tiles from cb_in -> cb_out directly
    compute_kernel_hw_startup(cb_in, cb_out);

    copy_tile_to_dst_init_short(cb_in);

    for (uint32_t row = 0; row < num_rows; ++row) {
        for (uint32_t t = 0; t < Wt; ++t) {
            cb_wait_front(cb_in, 1);
            cb_reserve_back(cb_out, 1);

            tile_regs_acquire();
            copy_tile(cb_in, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();

            cb_pop_front(cb_in, 1);
            cb_push_back(cb_out, 1);
        }
    }
#endif
}
