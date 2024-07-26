// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/bcast.h"


namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;
    uint32_t NC = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    uint32_t h_blk = get_arg_val<uint32_t>(3);
    uint32_t batch_b = get_arg_val<uint32_t>(4);
    uint32_t Ht_per_batch_b = get_arg_val<uint32_t>(5);

    init_bcast<BCAST_LLKOP, BCAST_DIM>(tt::CB::c_in0, tt::CB::c_in1, tt::CB::c_out0);

    cb_wait_front(tt::CB::c_in0, Wt*Ht);
    cb_reserve_back(tt::CB::c_out0, Wt*Ht);
    uint32_t b_offset = 0;
    for (uint32_t bn = 0; bn < batch_b; bn++) {
        for (uint32_t wt = 0; wt < Wt; wt++) {
            cb_wait_front(tt::CB::c_in1, onetile);
            for (uint32_t ht = 0; ht < Ht_per_batch_b; ht+=h_blk) {
                acquire_dst(tt::DstMode::Half);
                for (uint32_t htr = 0; htr<h_blk; htr++) {
                    uint32_t current_index = b_offset + (ht + htr) * Wt + wt;
                    BCAST_OP<BroadcastType::ROW>(tt::CB::c_in0, tt::CB::c_in1, current_index, 0, htr);
                    pack_tile<true>(htr, tt::CB::c_out0, current_index);
                }
                release_dst(tt::DstMode::Half);
            }
            cb_pop_front(tt::CB::c_in1, onetile);
        }
        b_offset += Ht_per_batch_b * Wt;
    }
    cb_pop_front(tt::CB::c_in0, Wt*Ht);
    cb_push_back(tt::CB::c_out0, Wt*Ht);
}
} // NAMESPACE
