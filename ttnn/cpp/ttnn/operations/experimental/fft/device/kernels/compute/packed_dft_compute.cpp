// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// packed_dft_compute.cpp — TRISC compute for the PACKED DIRECT-DFT kernel.
//

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_api.h"
#include "packed_dft_common.h"

void kernel_main() {
    const uint32_t tiles_per_core = get_compile_time_arg_val(0);

    mm_init(CB_A, CB_B, CB_OUT_R);

    for (uint32_t k = 0; k < tiles_per_core; ++k) {
        // ── out_R = in_R · T_R  +  in_I · T_I_neg ────────────────────────
        tile_regs_acquire();

        cb_wait_front(CB_A, 1);
        cb_wait_front(CB_B, 1);
        matmul_tiles(CB_A, CB_B, 0, 0, 0);
        cb_pop_front(CB_A, 1);
        cb_pop_front(CB_B, 1);

        cb_wait_front(CB_A, 1);
        cb_wait_front(CB_B, 1);
        matmul_tiles(CB_A, CB_B, 0, 0, 0);
        cb_pop_front(CB_A, 1);
        cb_pop_front(CB_B, 1);

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_OUT_R, 1);
        pack_tile(0, CB_OUT_R);
        cb_push_back(CB_OUT_R, 1);
        tile_regs_release();

        // ── out_I = in_R · T_I  +  in_I · T_R ────────────────────────────
        tile_regs_acquire();

        cb_wait_front(CB_A, 1);
        cb_wait_front(CB_B, 1);
        matmul_tiles(CB_A, CB_B, 0, 0, 0);
        cb_pop_front(CB_A, 1);
        cb_pop_front(CB_B, 1);

        cb_wait_front(CB_A, 1);
        cb_wait_front(CB_B, 1);
        matmul_tiles(CB_A, CB_B, 0, 0, 0);
        cb_pop_front(CB_A, 1);
        cb_pop_front(CB_B, 1);

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_OUT_I, 1);
        pack_tile(0, CB_OUT_I);
        cb_push_back(CB_OUT_I, 1);
        tile_regs_release();
    }
}
