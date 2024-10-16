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

#ifdef BCAST_SCALAR
    cb_wait_front(tt::CB::c_in1, onetile);
#endif

    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t h = 0; h < Ht; h++) {
            for (uint32_t w = 0; w < Wt; w++) {
#ifndef BCAST_SCALAR
                cb_wait_front(tt::CB::c_in1, onetile);
#endif
                cb_reserve_back(tt::CB::c_out0, onetile);

                acquire_dst();

                cb_wait_front(tt::CB::c_in0, onetile);

                BCAST_OP<BroadcastType::SCALAR>(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0);
                pack_tile(0, tt::CB::c_out0);

                cb_pop_front(tt::CB::c_in0, onetile);
#ifndef BCAST_SCALAR
                cb_pop_front(tt::CB::c_in1, onetile);
#endif
                release_dst();

                cb_push_back(tt::CB::c_out0, onetile);
            }
        }
    }
}
}  // namespace NAMESPACE
