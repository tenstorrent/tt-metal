// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

constexpr uint32_t onetile = 1;

void kernel_main() {
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_src = tt::CB::c_in0;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t rows_offset = get_arg_val<uint32_t>(2);
    const uint32_t Wt = get_arg_val<uint32_t>(3);
    const bool flip = get_arg_val<uint32_t>(4) == 1;

    const uint32_t tile_size = get_tile_size(cb_src);
    const DataFormat data_format = get_dataformat(cb_src);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_size, .data_format = data_format};

    for (uint32_t r = rows_offset; r < rows_offset + num_rows; r++) {
        for (uint32_t i = 0; i < Wt; i++) {
            const uint32_t tile_index = flip ? r * Wt + (Wt - i - 1) : r * Wt + i;
            cb_reserve_back(cb_src, onetile);
            const uint32_t cb_src_addr = get_write_ptr(cb_src);
            noc_async_read_tile(tile_index, s, cb_src_addr);
            noc_async_read_barrier();
            cb_push_back(cb_src, onetile);
        }
    }
}
