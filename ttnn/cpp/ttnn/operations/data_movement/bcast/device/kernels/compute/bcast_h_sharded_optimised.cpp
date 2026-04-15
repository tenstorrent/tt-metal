// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t onetile = 1;
    uint32_t NC = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    uint32_t h_blk = get_arg_val<uint32_t>(3);
    uint32_t batch_b = get_arg_val<uint32_t>(4);
    uint32_t Ht_per_batch_b = get_arg_val<uint32_t>(5);

    constexpr auto cb_id_in0 = static_cast<tt::CBIndex>(cb_in0);
    constexpr auto cb_id_in1 = static_cast<tt::CBIndex>(cb_in1);
    constexpr auto cb_id_out = static_cast<tt::CBIndex>(cb_out);

    init_bcast<BCAST_LLKOP, BCAST_DIM>(cb_id_in0, cb_id_in1, cb_id_out);

    experimental::CircularBuffer cb_src0(cb_id_in0);
    experimental::CircularBuffer cb_src1(cb_id_in1);
    experimental::CircularBuffer cb_dst(cb_id_out);

    cb_src0.wait_front(Wt * Ht);
    cb_dst.reserve_back(Wt * Ht);
    uint32_t b_offset = 0;
    for (uint32_t bn = 0; bn < batch_b; bn++) {
        for (uint32_t wt = 0; wt < Wt; wt++) {
            cb_src1.wait_front(onetile);
            for (uint32_t ht = 0; ht < Ht_per_batch_b; ht += h_blk) {
                acquire_dst();
                for (uint32_t htr = 0; htr < h_blk; htr++) {
                    uint32_t current_index = b_offset + (ht + htr) * Wt + wt;
                    BCAST_OP<BroadcastType::ROW>(cb_id_in0, cb_id_in1, current_index, 0, htr);
                    pack_tile<true>(htr, cb_id_out, current_index);
                }
                release_dst();
            }
            cb_src1.pop_front(onetile);
        }
        b_offset += Ht_per_batch_b * Wt;
    }
    cb_src0.pop_front(Wt * Ht);
    cb_dst.push_back(Wt * Ht);
}
