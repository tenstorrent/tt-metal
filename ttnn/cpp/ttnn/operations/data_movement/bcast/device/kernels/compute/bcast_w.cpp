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

    constexpr auto cb_id_in0 = static_cast<tt::CBIndex>(cb_in0);
    constexpr auto cb_id_in1 = static_cast<tt::CBIndex>(cb_in1);
    constexpr auto cb_id_out = static_cast<tt::CBIndex>(cb_out);

    init_bcast<BCAST_LLKOP, BCAST_DIM>(cb_id_in0, cb_id_in1, cb_id_out);

    experimental::CircularBuffer cb_src0(cb_id_in0);
    experimental::CircularBuffer cb_src1(cb_id_in1);
    experimental::CircularBuffer cb_dst(cb_id_out);

    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t h = 0; h < Ht; h++) {
            cb_src1.wait_front(onetile);
            for (uint32_t w = 0; w < Wt; w++) {
                cb_dst.reserve_back(onetile);

                acquire_dst();

                cb_src0.wait_front(onetile);
                BCAST_OP<BroadcastType::COL>(cb_id_in0, cb_id_in1, 0, 0, 0);
                pack_tile(0, cb_id_out);
                cb_src0.pop_front(onetile);

                release_dst();

                cb_dst.push_back(onetile);
            }
            cb_src1.pop_front(onetile);
        }
    }
}
