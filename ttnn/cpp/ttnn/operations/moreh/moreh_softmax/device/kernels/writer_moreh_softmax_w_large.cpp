// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t N = get_arg_val<uint32_t>(1);
    uint32_t tile_offset = get_arg_val<uint32_t>(2);
    uint32_t Wt = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_out);

    const DataFormat data_format = get_dataformat(cb_id_out);
    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t blk = 1;

    uint32_t tile_id = tile_offset;
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t w = 0; w < Wt; w++) {
            cb_wait_front(cb_id_out, blk);
            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
            noc_async_write_tile(tile_id, s, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_out, blk);
            tile_id++;
        }
    }
}
