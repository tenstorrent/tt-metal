// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src1_addr = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    uint32_t offset = get_arg_val<uint32_t>(3);
    uint32_t batch_offset = get_arg_val<uint32_t>(4);  // if weight has multiple batches
    uint32_t w_blk = get_arg_val<uint32_t>(5);
    uint32_t batch_b = get_arg_val<uint32_t>(6);

    // constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);

    // constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t onetile = 1;

    // single-tile ublocks
    const uint32_t tile_bytes = get_tile_size(cb_id_in1);
    const DataFormat data_format = get_dataformat(cb_id_in1);

    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    cb_push_back(cb_id_in0, Ht * Wt);
    for (uint32_t b = 0; b < batch_b; b++) {
        for (uint32_t wt = 0; wt < Wt; wt += w_blk) {
            cb_reserve_back(cb_id_in1, w_blk);
            l1_write_addr_in1 = get_write_ptr(cb_id_in1);
            for (uint32_t r = 0; r < w_blk; r++) {
                noc_async_read_tile(offset + wt + r, s1, l1_write_addr_in1);
                l1_write_addr_in1 += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_in1, w_blk);
        }
        offset += batch_offset;
    }
}
