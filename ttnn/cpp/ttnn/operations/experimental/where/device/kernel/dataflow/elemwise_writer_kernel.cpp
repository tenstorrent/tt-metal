// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_base_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t tile_ofs = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_dst = get_compile_time_arg_val(0);
    constexpr bool is_dst_dram = get_compile_time_arg_val(1) == 1;

    const InterleavedAddrGenFast<is_dst_dram> dst_tensor_addr_gen = {
        .bank_base_address = dst_base_addr, .page_size = get_tile_size(cb_dst), .data_format = get_dataformat(cb_dst)};

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    uint32_t end_id = tile_ofs + num_tiles;
    for (uint32_t i = tile_ofs; i < end_id; ++i) {
        cb_wait_front(cb_dst, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_dst);
        noc_async_write_tile(i, dst_tensor_addr_gen, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_dst, onetile);
    }
}
