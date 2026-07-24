// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_a_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_b_id = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_id = tt::CBIndex::c_16;

    DataflowBuffer dfb_a(cb_a_id);
    DataflowBuffer dfb_b(cb_b_id);
    DataflowBuffer dfb_out(cb_out_id);

    uint32_t B = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    init_bcast<BCAST_LLKOP, BCAST_DIM>(cb_a_id, cb_b_id, cb_out_id);

#ifdef BCAST_SCALAR
    dfb_b.wait_front(onetile);
#endif

    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t h = 0; h < Ht; h++) {
            for (uint32_t w = 0; w < Wt; w++) {
#ifndef BCAST_SCALAR
                dfb_b.wait_front(onetile);
#endif
                dfb_a.wait_front(onetile);

                tile_regs_acquire();
                BCAST_OP<BroadcastType::SCALAR>(cb_a_id, cb_b_id, 0, 0, 0);
                tile_regs_commit();

                dfb_a.pop_front(onetile);
#ifndef BCAST_SCALAR
                dfb_b.pop_front(onetile);
#endif

                dfb_out.reserve_back(onetile);

                tile_regs_wait();
                pack_tile(0, cb_out_id);
                tile_regs_release();

                dfb_out.push_back(onetile);
            }
        }
    }
}
