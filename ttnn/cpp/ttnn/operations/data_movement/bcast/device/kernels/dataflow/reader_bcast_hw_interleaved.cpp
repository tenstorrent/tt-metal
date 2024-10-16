// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
    uint32_t nc1 = get_arg_val<uint32_t>(12);  // if 1 we expect the bcast tensor to have NC=1 and wrap around in NC

#ifndef IN0_SHARDED
    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
#endif

    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t onetile = 1;

    // single-tile ublocks
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat in0_data_format = get_dataformat(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);
    const DataFormat in1_data_format = get_dataformat(cb_id_in1);

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    uint32_t num_tiles = src0_num_tiles;
    uint32_t i1 = 0;

#ifndef IN0_SHARDED
    uint32_t i = 0;
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = in0_tile_bytes, .data_format = in0_data_format};
#else
    cb_reserve_back(cb_id_in0, num_tiles);
    cb_push_back(cb_id_in0, num_tiles);
#endif

    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr, .page_size = in1_tile_bytes, .data_format = in1_data_format};

#ifdef BCAST_SCALAR
    cb_reserve_back(cb_id_in1, onetile);
    l1_write_addr_in1 = get_write_ptr(cb_id_in1);
    noc_async_read_tile(i1, s1, l1_write_addr_in1);
    noc_async_read_barrier();
    cb_push_back(cb_id_in1, onetile);
#endif

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t ht = 0; ht < Ht; ht++) {
            for (uint32_t wt = 0; wt < Wt; wt++) {
#ifndef IN0_SHARDED
                cb_reserve_back(cb_id_in0, onetile);
                l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read_tile(i, s0, l1_write_addr_in0);
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, onetile);
                i++;  // input tile iterates over NC Ht Wt
#endif

#ifndef BCAST_SCALAR
                // for each H,W-tile of the first tensor we push one tile from the second arg tile list
                // but we don't advance the second tile index for H,W
                cb_reserve_back(cb_id_in1, onetile);
                l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                noc_async_read_tile(i1, s1, l1_write_addr_in1);
                noc_async_read_barrier();
                cb_push_back(cb_id_in1, onetile);
#endif
            }  // wt loop
        }  // ht loop
#ifndef BCAST_SCALAR
        if (nc1 == 0) {
            i1++;  // bcast-HW tile iterates only for nc loop and only if NC>1
        }
#endif
    }  // nc loop
}
