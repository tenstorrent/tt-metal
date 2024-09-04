// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {

    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t start_row_idx = get_arg_val<uint32_t>(1); // Index correctly in the for loop

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t num_rows_per_core =  get_compile_time_arg_val(2);
    constexpr uint32_t num_sin_cos_rows_per_core = get_compile_time_arg_val(3);
    constexpr uint32_t Wt =  get_compile_time_arg_val(4);
    constexpr uint32_t Ht =  get_compile_time_arg_val(5);


    // single-tile ublocks
    constexpr uint32_t onetile = 1;

    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    uint32_t output_row_cnt = 0;

    uint32_t tile_idx = start_row_idx * Wt; // start index in tiles, instead of rows
    for (uint32_t i = 0; i < num_rows_per_core; i++) {
        cb_wait_front(cb_id_out, Wt);

        // Write a row
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        for (uint32_t j = 0; j < Wt; j++) {
            noc_async_write_tile(tile_idx, s, l1_read_addr);
            l1_read_addr += tile_bytes;
            tile_idx++;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, Wt);
        output_row_cnt++;

        if (output_row_cnt % num_sin_cos_rows_per_core == 0) {
            tile_idx += (Ht - num_sin_cos_rows_per_core) * Wt; // Increment by stride
        }
    }
}
