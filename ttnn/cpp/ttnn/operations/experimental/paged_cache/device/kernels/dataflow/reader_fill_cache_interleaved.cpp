// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr  = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id_in = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);

    const uint32_t tile_bytes = get_tile_size(cb_id_in);
    const DataFormat data_format = get_dataformat(cb_id_in);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    // uint32_t end_id = start_id + num_tiles;
    uint32_t tile_id = start_tile_id;
    for (uint32_t row_num = 0; row_num < num_rows; ++row_num) {
        cb_reserve_back(cb_id_in, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in);
        for (uint32_t w = 0; w < Wt; ++w) {
            noc_async_read_tile(tile_id, s, l1_write_addr);
            l1_write_addr += tile_bytes;
            tile_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in, Wt);
    }
}
