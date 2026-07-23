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

    auto NC = get_arg(args::NC);
    auto Ht = get_arg(args::Ht);
    auto Wt = get_arg(args::Wt);
    auto h_blk = get_arg(args::h_blk);
    auto batch_b = get_arg(args::batch_b);
    auto Ht_per_batch_b = get_arg(args::Ht_per_batch_b);

    init_bcast<BCAST_LLKOP, BCAST_DIM>(dfb::in0, dfb::in1, dfb::out);

    dfb_a.wait_front(Wt * Ht);
    dfb_out.reserve_back(Wt * Ht);
    std::uint32_t b_offset = 0;
    for (std::uint32_t bn = 0; bn < batch_b; bn++) {
        for (std::uint32_t wt = 0; wt < Wt; wt++) {
            dfb_b.wait_front(onetile);
            for (std::uint32_t ht = 0; ht < Ht_per_batch_b; ht += h_blk) {
                tile_regs_acquire();
                for (std::uint32_t htr = 0; htr < h_blk; htr++) {
                    std::uint32_t current_index = b_offset + (ht + htr) * Wt + wt;
                    BCAST_OP<BroadcastType::ROW>(dfb::in0, dfb::in1, current_index, 0, htr);
                }
                tile_regs_commit();

                tile_regs_wait();
                for (std::uint32_t htr = 0; htr < h_blk; htr++) {
                    std::uint32_t current_index = b_offset + (ht + htr) * Wt + wt;
                    pack_tile<true>(htr, dfb::out, current_index);
                }
                tile_regs_release();
            }
            dfb_b.pop_front(onetile);
        }
        b_offset += Ht_per_batch_b * Wt;
    }
    dfb_a.pop_front(Wt * Ht);
    dfb_out.push_back(Wt * Ht);
}
