// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This is the Metal 2.0 fork of bcast_hw.cpp, which lives beside it. Ops ported to Metal 2.0 bind
// this file; the original serves the consumers still on the legacy API. Until the last of them migrates
// and the original is retired, changes here likely belong there too.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb_a(dfb::in0);
    DataflowBuffer dfb_b(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);

    uint32_t B = get_arg(args::B);
    uint32_t Ht = get_arg(args::Ht);
    uint32_t Wt = get_arg(args::Wt);
    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);
    bcast_init<BCAST_LLKOP, BCAST_DIM>(dfb::in0, dfb::in1);

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
                BCAST_OP<BroadcastType::SCALAR>(dfb::in0, dfb::in1, 0, 0, 0);
                tile_regs_commit();

                dfb_a.pop_front(onetile);
#ifndef BCAST_SCALAR
                dfb_b.pop_front(onetile);
#endif

                dfb_out.reserve_back(onetile);

                tile_regs_wait();
                pack_tile(0, dfb::out);
                tile_regs_release();

                dfb_out.push_back(onetile);
            }
        }
    }
}
