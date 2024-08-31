// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

constexpr uint32_t onetile = 1;

void kernel_main() {
    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr bool flip = get_compile_time_arg_val(3) == 1;

    constexpr uint32_t cb_dst = tt::CB::c_out0;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_cols = get_arg_val<uint32_t>(1);
    const uint32_t cols_offset = get_arg_val<uint32_t>(2);

    const uint32_t tile_size = get_tile_size(cb_dst);
    const DataFormat data_format = get_dataformat(cb_dst);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_size, .data_format = data_format};

    for (uint32_t c = cols_offset; c < cols_offset + num_cols; c++) {
        for (uint32_t i = 0; i < Ht; i++) {
            const uint32_t j = (c / Wt) * Ht * Wt + (c % Wt);
            const uint32_t tile_index = flip ? (Ht - i - 1) * Wt + j : i * Wt + j;
            cb_wait_front(cb_dst, onetile);
            const uint32_t cb_dst_addr = get_read_ptr(cb_dst);
            noc_async_write_tile(tile_index, s, cb_dst_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_dst, onetile);
        }
    }
}
