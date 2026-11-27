// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    const uint32_t blocks = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_IN0, CB_IN1, CB_OUT);
    matmul_init(CB_IN0, CB_IN1);
    cb_wait_front(CB_IN0, MM_KT);
    cb_wait_front(CB_IN1, MM_KT);
    cb_reserve_back(CB_OUT, 1);
    for (uint32_t b = 0; b < blocks; ++b) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < MM_KT; ++k) {
            matmul_tiles(CB_IN0, CB_IN1, k, k, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        // Out-of-order mode pins every pack to tile 0; sequential mode would walk past the 1-tile CB into L1.
        pack_tile<true>(0, CB_OUT, 0);
        tile_regs_release();
    }
    // The readers poll this push to know the load is still running.
    cb_push_back(CB_OUT, 1);
    cb_pop_front(CB_IN0, MM_KT);
    cb_pop_front(CB_IN1, MM_KT);
}
