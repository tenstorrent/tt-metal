// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;

    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = 16;

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++ i) {
        cb_wait_front(cb_id_out, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

        volatile tt_l1_ptr int32_t* out_l1_ptr = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1_read_addr);
        for (uint32_t h = 0; h < 16; h++) {
            for (uint32_t subw = 1; subw < 8; ++subw) {
                out_l1_ptr[h * 16] += out_l1_ptr[h * 16 + (2 * subw)];
                out_l1_ptr[(h * 16) + 512] += out_l1_ptr[(h * 16) + 512 + (2 * subw)];
            }
        }

        noc_async_write_tile(i, s, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, onetile);
    }
}
