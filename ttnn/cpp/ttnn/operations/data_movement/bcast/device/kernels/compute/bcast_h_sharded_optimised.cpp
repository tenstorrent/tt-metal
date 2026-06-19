// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_a_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_b_id = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_id = tt::CBIndex::c_16;

    CircularBuffer cb_a(cb_a_id);
    CircularBuffer cb_b(cb_b_id);
    CircularBuffer cb_out(cb_out_id);

    uint32_t NC = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    uint32_t h_blk = get_arg_val<uint32_t>(3);
    uint32_t batch_b = get_arg_val<uint32_t>(4);
    uint32_t Ht_per_batch_b = get_arg_val<uint32_t>(5);

    init_bcast<BCAST_LLKOP, BCAST_DIM>(cb_a_id, cb_b_id, cb_out_id);

    cb_a.wait_front(Wt * Ht);
    cb_out.reserve_back(Wt * Ht);
    uint32_t b_offset = 0;
    for (uint32_t bn = 0; bn < batch_b; bn++) {
        for (uint32_t wt = 0; wt < Wt; wt++) {
            cb_b.wait_front(onetile);
            for (uint32_t ht = 0; ht < Ht_per_batch_b; ht += h_blk) {
                tile_regs_acquire();
                for (uint32_t htr = 0; htr < h_blk; htr++) {
                    uint32_t current_index = b_offset + (ht + htr) * Wt + wt;
                    BCAST_OP<BroadcastType::ROW>(cb_a_id, cb_b_id, current_index, 0, htr);
                }
                tile_regs_commit();

                tile_regs_wait();
                for (uint32_t htr = 0; htr < h_blk; htr++) {
                    uint32_t current_index = b_offset + (ht + htr) * Wt + wt;
                    pack_tile<true>(htr, cb_out_id, current_index);
                }
                tile_regs_release();
            }
            cb_b.pop_front(onetile);
        }
        b_offset += Ht_per_batch_b * Wt;
    }
    cb_a.pop_front(Wt * Ht);
    cb_out.push_back(Wt * Ht);
}
