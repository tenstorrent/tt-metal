// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/bcast.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;
    uint32_t B = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    init_bcast<BCAST_LLKOP, BCAST_DIM>(tt::CB::c_in0, tt::CB::c_in1);

    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t h = 0; h < Ht; h++) {
            for (uint32_t w = 0; w < Wt; w++) {
                // For this bcast-h op the reader will wrap the RHS source tile around at Wt
                // so here we just linearly read 2 parallel arrays and apply bcast op per tile
                // (bcast_h propagates the op down the H dimension, so it can be though of as bcast to H)
                cb_wait_front(tt::CB::c_in1, onetile);

                cb_reserve_back(tt::CB::c_out0, onetile);

                acquire_dst();

                cb_wait_front(tt::CB::c_in0, onetile);

                BCAST_OP<BroadcastType::ROW>(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0);
                pack_tile(0, tt::CB::c_out0);

                cb_pop_front(tt::CB::c_in0, onetile);

                release_dst();

                cb_push_back(tt::CB::c_out0, onetile);
                cb_pop_front(tt::CB::c_in1, onetile);
            }
        }
    }
}
}  // namespace NAMESPACE
