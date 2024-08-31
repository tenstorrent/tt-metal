// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

constexpr uint32_t onetile = 1;

void kernel_main() {
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr bool flip = get_compile_time_arg_val(3) == 1;

    constexpr uint32_t cb_src = tt::CB::c_in0;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_cols = get_arg_val<uint32_t>(1);
    const uint32_t cols_offset = get_arg_val<uint32_t>(2);

    const uint32_t tile_size = get_tile_size(cb_src);
    const DataFormat data_format = get_dataformat(cb_src);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_size, .data_format = data_format};

    for (uint32_t c = cols_offset; c < cols_offset + num_cols; c++) {
        for (uint32_t i = 0; i < Ht; i++) {
            const uint32_t j = (c / Wt) * Ht * Wt + (c % Wt);
            const uint32_t tile_index = flip ? (Ht - i - 1) * Wt + j : i * Wt + j;
            cb_reserve_back(cb_src, onetile);
            const uint32_t cb_src_addr = get_write_ptr(cb_src);
            noc_async_read_tile(tile_index, s, cb_src_addr);
            noc_async_read_barrier();
            cb_push_back(cb_src, onetile);
        }
    }
}
