// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This code is temporarily copied from ttnn/cpp/ttnn/operations/datamovement/binary/device/ to demonstrate
// the new ability to keep the CircularBufferConfigs continuous during dispatching.  See the use of CBIndex::c_2 below.
// When broadcating is properly supported we expect this code to be deleted or refactored substantially.

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_num_tiles = get_arg_val<uint32_t>(3);
    uint32_t src1_addr = get_arg_val<uint32_t>(4);
    // skip args 1,2,5,6,7 for compat with single bank readers and reader_diff_lengths
    uint32_t NCHtWt = get_arg_val<uint32_t>(8);
    uint32_t NC = get_arg_val<uint32_t>(9);
    uint32_t Ht = get_arg_val<uint32_t>(10);
    uint32_t Wt = get_arg_val<uint32_t>(11);
    uint32_t nc1 = get_arg_val<uint32_t>(12);  // if 1 we expect the bcast tensor to have NC=1

    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t onetile = 1;

    // single-tile ublocks
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);

    const auto s0 = TensorAccessor(src0_args, src0_addr, tile_bytes);
    const auto s1 = TensorAccessor(src1_args, src1_addr, tile_bytes);

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    uint32_t num_tiles = src0_num_tiles;
    uint32_t i = 0;
    uint32_t i1 = 0;
    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t ht = 0; ht < Ht; ht++) {
            for (uint32_t wt = 0; wt < Wt; wt++) {
                cb_reserve_back(cb_id_in0, onetile);
                l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read_tile(i, s0, l1_write_addr_in0);
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, onetile);

                // for each W-tile of the first tensor we push one tile from the second arg tile list
                // but we loop the second list around
                cb_reserve_back(cb_id_in1, onetile);
                l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                noc_async_read_tile(i1, s1, l1_write_addr_in1);
                noc_async_read_barrier();
                cb_push_back(cb_id_in1, onetile);
                i1++;
                i++;  // input tile iterates over NC Ht Wt
            }

            // bcast tensor should be NC1W (actually NC32W padded with 0s in H)
            // wrap W around for each h (broadcast)
            i1 -= Wt;
        }
        // we reused Wt tiles out of NCWt bcast tensor Ht times, now advance for next NC
        if (nc1 == 0) {  // if bcast NC==1 we don't advance but reuse the tensor
            i1 += Wt;
        }
    }
}
