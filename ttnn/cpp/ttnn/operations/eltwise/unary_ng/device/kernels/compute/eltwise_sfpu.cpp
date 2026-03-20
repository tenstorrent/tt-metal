// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Runtime arg 0: num_tiles (tile count for this core), arg 1: packed_scalar1, arg 2: packed_scalar2.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();

        cb_wait_front(tt::CBIndex::c_0, 1);
        cb_reserve_back(tt::CBIndex::c_2, 1);

        copy_tile(tt::CBIndex::c_0, 0, 0);

#ifdef SFPU_OP_CHAIN_0
        SFPU_OP_CHAIN_0
#endif

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, tt::CBIndex::c_2);

        cb_pop_front(tt::CBIndex::c_0, 1);
        cb_push_back(tt::CBIndex::c_2, 1);

        tile_regs_release();
    }
}
