// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

namespace NAMESPACE {
void MAIN {
    constexpr int onetile = 1;
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    bool enable_reload = false;
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        bool last_out = block == (per_core_block_cnt - 1);

        // elemwise-mul
        ACQ();
        cb_wait_front(tt::CBIndex::c_0, onetile);
        cb_wait_front(tt::CBIndex::c_1, onetile);

        cb_reserve_back(tt::CBIndex::c_24, onetile);
        mul_tiles_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
        // dst0 = c_in0 x c_in1
        mul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
        // c_intermed0 = pack(dst0)
        pack_tile(0, tt::CBIndex::c_24);
        cb_push_back(tt::CBIndex::c_24, onetile);

        cb_pop_front(tt::CBIndex::c_0, onetile);
        cb_pop_front(tt::CBIndex::c_1, onetile);
        REL();

        // reduce-w
        ACQ();
        if (enable_reload) {
            cb_wait_front(tt::CBIndex::c_25, onetile);
            copy_tile_to_dst_init_short(tt::CBIndex::c_25);
            copy_tile(tt::CBIndex::c_25, 0, 0);
            cb_pop_front(tt::CBIndex::c_25, onetile);
        }

        cb_wait_front(tt::CBIndex::c_24, onetile);
        reduce_init(tt::CBIndex::c_24, tt::CBIndex::c_2, tt::CBIndex::c_16);
        reduce_tile(tt::CBIndex::c_24, tt::CBIndex::c_2, 0, 0, 0);
        cb_pop_front(tt::CBIndex::c_24, onetile);
        reduce_uninit();

        if (last_out) {
            cb_reserve_back(tt::CBIndex::c_16, onetile);
            pack_tile(0, tt::CBIndex::c_16);
            cb_push_back(tt::CBIndex::c_16, onetile);
        } else {
            cb_reserve_back(tt::CBIndex::c_25, onetile);
            pack_tile(0, tt::CBIndex::c_25);
            cb_push_back(tt::CBIndex::c_25, onetile);
        }
        REL();
        enable_reload = true;
    }
}
}  // namespace NAMESPACE
