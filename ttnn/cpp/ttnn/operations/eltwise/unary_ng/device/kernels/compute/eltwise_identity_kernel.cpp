// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);
    copy_tile_init(tt::CBIndex::c_0);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();

        cb_wait_front(tt::CBIndex::c_0, 1);
        cb_reserve_back(tt::CBIndex::c_2, 1);

        copy_tile(tt::CBIndex::c_0, 0, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, tt::CBIndex::c_2);

        cb_pop_front(tt::CBIndex::c_0, 1);
        cb_push_back(tt::CBIndex::c_2, 1);

        tile_regs_release();
    }
}
