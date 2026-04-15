// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t onetile = 1;
    uint32_t B = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    init_bcast<BCAST_LLKOP, BCAST_DIM>(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    experimental::CircularBuffer cb_in0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb_in1(tt::CBIndex::c_1);
    experimental::CircularBuffer cb_out(tt::CBIndex::c_16);

    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t h = 0; h < Ht; h++) {
            for (uint32_t w = 0; w < Wt; w++) {
                // For this bcast-h op the reader will wrap the RHS source tile around at Wt
                // so here we just linearly read 2 parallel arrays and apply bcast op per tile
                // (bcast_h propagates the op down the H dimension, so it can be though of as bcast to H)
                cb_in1.wait_front(onetile);

                cb_out.reserve_back(onetile);

                acquire_dst();

                cb_in0.wait_front(onetile);

                BCAST_OP<BroadcastType::ROW>(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
                pack_tile(0, tt::CBIndex::c_16);

                cb_in0.pop_front(onetile);

                release_dst();

                cb_out.push_back(onetile);
                cb_in1.pop_front(onetile);
            }
        }
    }
}
