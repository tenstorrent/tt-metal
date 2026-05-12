// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "experimental/circular_buffer.h"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

void kernel_main() {
    constexpr int onetile = 1;
    uint32_t has_input_grad = get_arg_val<uint32_t>(0);
    uint32_t has_other_grad = get_arg_val<uint32_t>(1);
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(2);

    experimental::CircularBuffer cb_c0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb_c1(tt::CBIndex::c_1);
    experimental::CircularBuffer cb_c2(tt::CBIndex::c_2);
    experimental::CircularBuffer cb_c16(tt::CBIndex::c_16);
    experimental::CircularBuffer cb_c17(tt::CBIndex::c_17);

    init_bcast<ELWMUL, BroadcastType::SCALAR>(tt::CBIndex::c_2, tt::CBIndex::c_0, tt::CBIndex::c_16);
    cb_c0.wait_front(onetile);
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        if (has_input_grad) {
            cb_c2.wait_front(onetile);
            ACQ();
            mul_tiles_bcast<BroadcastType::SCALAR>(tt::CBIndex::c_2, tt::CBIndex::c_0, 0, 0, 0);
            pack_tile(0, tt::CBIndex::c_16);
            cb_c16.push_back(onetile);
            cb_c2.pop_front(onetile);
            REL();
        }

        if (has_other_grad) {
            cb_c1.wait_front(onetile);
            ACQ();
            mul_tiles_bcast<BroadcastType::SCALAR>(tt::CBIndex::c_1, tt::CBIndex::c_0, 0, 0, 0);
            pack_tile(0, tt::CBIndex::c_17);
            cb_c17.push_back(onetile);
            cb_c1.pop_front(onetile);
            REL();
        }
    }
}
