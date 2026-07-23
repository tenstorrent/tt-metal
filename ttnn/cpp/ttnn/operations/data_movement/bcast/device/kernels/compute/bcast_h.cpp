// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
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
            for (std::uint32_t w = 0; w < Wt; w++) {
                // For this bcast-h op the reader will wrap the RHS source tile around at Wt
                // so here we just linearly read 2 parallel arrays and apply bcast op per tile
                // (bcast_h propagates the op down the H dimension, so it can be though of as bcast to H)
                dfb_b.wait_front(onetile);
                dfb_a.wait_front(onetile);

                tile_regs_acquire();
                BCAST_OP<BroadcastType::ROW>(dfb::in0, dfb::in1, 0, 0, 0);
                tile_regs_commit();

                dfb_a.pop_front(onetile);
                dfb_b.pop_front(onetile);

                dfb_out.reserve_back(onetile);

                tile_regs_wait();
                pack_tile(0, dfb::out);
                tile_regs_release();

                dfb_out.push_back(onetile);
            }
        }
    }
}
