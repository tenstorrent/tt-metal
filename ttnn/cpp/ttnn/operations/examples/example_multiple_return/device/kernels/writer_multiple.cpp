// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "dprint.h"

void kernel_main() {
    uint32_t dst_addr1  = get_arg_val<uint32_t>(0);
    uint32_t dst_addr2  = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram1 = get_compile_time_arg_val(1) == 1;
    constexpr bool dst_is_dram2 = get_compile_time_arg_val(2) == 1;

    #ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, num_tiles);
    #else

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<dst_is_dram1> s1 = {
        .bank_base_address = dst_addr1,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    const InterleavedAddrGenFast<dst_is_dram2> s2 = {
        .bank_base_address = dst_addr2,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    #ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; -- i) {
    #else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++ i) {
    #endif
        cb_wait_front(cb_id_out, onetile);

        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        noc_async_write_tile(i, s1, l1_read_addr);
        noc_async_write_barrier();

        noc_async_write_tile(i, s2, l1_read_addr);
        noc_async_write_barrier();

        cb_pop_front(cb_id_out, onetile);
    }
    #endif
}
