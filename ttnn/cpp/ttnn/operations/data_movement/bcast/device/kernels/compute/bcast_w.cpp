// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    uint32_t w = 0;
    constexpr uint32_t onetile = 1;
    uint32_t B = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);

    experimental::CircularBuffer cb_src0(static_cast<tt::CBIndex>(cb_id_in0));
    experimental::CircularBuffer cb_src1(static_cast<tt::CBIndex>(cb_id_in1));
    experimental::CircularBuffer cb_dst(static_cast<tt::CBIndex>(cb_id_out));

    init_bcast<BCAST_LLKOP, BCAST_DIM>(cb_src0.get_cb_id(), cb_src1.get_cb_id(), cb_dst.get_cb_id());

    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t h = 0; h < Ht; h++) {
            cb_src1.wait_front(onetile);
            for (uint32_t w = 0; w < Wt; w++) {
                cb_dst.reserve_back(onetile);

                acquire_dst();

                cb_src0.wait_front(onetile);
                BCAST_OP<BroadcastType::COL>(cb_src0.get_cb_id(), cb_src1.get_cb_id(), 0, 0, 0);
                pack_tile(0, cb_dst.get_cb_id());
                cb_src0.pop_front(onetile);

                release_dst();

                cb_dst.push_back(onetile);
            }
            cb_src1.pop_front(onetile);
        }
    }
}
