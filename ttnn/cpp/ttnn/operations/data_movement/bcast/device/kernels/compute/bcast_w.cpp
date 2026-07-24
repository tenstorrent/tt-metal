// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    std::uint32_t w = 0;
    constexpr std::uint32_t onetile = 1;

    DataflowBuffer dfb_a(dfb::in0);
    DataflowBuffer dfb_b(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);

    auto B = get_arg(args::B);
    auto Ht = get_arg(args::Ht);
    auto Wt = get_arg(args::Wt);

    init_bcast<BCAST_LLKOP, BCAST_DIM>(dfb::in0, dfb::in1, dfb::out);

    for (std::uint32_t b = 0; b < B; b++) {
        for (std::uint32_t h = 0; h < Ht; h++) {
            dfb_b.wait_front(onetile);
            for (std::uint32_t w = 0; w < Wt; w++) {
                dfb_a.wait_front(onetile);

                tile_regs_acquire();
                BCAST_OP<BroadcastType::COL>(dfb::in0, dfb::in1, 0, 0, 0);
                tile_regs_commit();

                dfb_a.pop_front(onetile);

                dfb_out.reserve_back(onetile);

                tile_regs_wait();
                pack_tile(0, dfb::out);
                tile_regs_release();

                dfb_out.push_back(onetile);
            }
            dfb_b.pop_front(onetile);
        }
    }
}
