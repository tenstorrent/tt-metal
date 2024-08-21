// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/bcast.h"

namespace NAMESPACE {
void MAIN {
    uint32_t w = 0;
    constexpr uint32_t onetile = 1;
    uint32_t B = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);

    init_bcast<BCAST_LLKOP, BCAST_DIM>(tt::CB::c_in0, tt::CB::c_in1);

    for (uint32_t b = 0; b < B; b++) {
    for (uint32_t h = 0; h < Ht; h++) {
        cb_wait_front(tt::CB::c_in1, onetile);
        for (uint32_t w = 0; w < Wt; w++) {

            cb_reserve_back(tt::CB::c_out0, onetile);

            acquire_dst(tt::DstMode::Half);

            cb_wait_front(tt::CB::c_in0, onetile);
            BCAST_OP<BroadcastType::COL>(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0);
            pack_tile(0, tt::CB::c_out0);
            cb_pop_front(tt::CB::c_in0, onetile);

            release_dst(tt::DstMode::Half);

            cb_push_back(tt::CB::c_out0, onetile);

        }
        cb_pop_front(tt::CB::c_in1, onetile);
    }}
}
} // NAMESPACE
