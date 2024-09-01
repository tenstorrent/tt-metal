// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

constexpr uint32_t onetile = 1;

void kernel_main() {
    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_dst = tt::CB::c_out0;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t rows_offset = get_arg_val<uint32_t>(2);
    const uint32_t Wt = get_arg_val<uint32_t>(3);
    const bool flip = get_arg_val<uint32_t>(4) == 1;

    const uint32_t tile_size = get_tile_size(cb_dst);
    const DataFormat data_format = get_dataformat(cb_dst);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_size, .data_format = data_format};

    for (uint32_t r = rows_offset; r < rows_offset + num_rows; r++) {
        for (uint32_t i = 0; i < Wt; i++) {
            const uint32_t tile_index = flip ? r * Wt + (Wt - i - 1) : r * Wt + i;
            cb_wait_front(cb_dst, onetile);
            const uint32_t cb_dst_addr = get_read_ptr(cb_dst);
            noc_async_write_tile(tile_index, s, cb_dst_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_dst, onetile);
        }
    }
}
