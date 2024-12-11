// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_2;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init();

    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_in1, onetile);

    for(uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        cb_wait_front(cb_in0, onetile);
        cb_reserve_back(cb_out0, onetile);

        tile_regs_acquire();
        add_tiles(cb_in0, cb_in1, 0, 0, 0);
#ifdef SFPU_OP_CHAIN_0
        SFPU_OP_CHAIN_0
#endif
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out0);
        tile_regs_release();

        cb_pop_front(cb_in0, onetile);
        cb_push_back(cb_out0, onetile);
    }
}
}  // namespace NAMESPACE
