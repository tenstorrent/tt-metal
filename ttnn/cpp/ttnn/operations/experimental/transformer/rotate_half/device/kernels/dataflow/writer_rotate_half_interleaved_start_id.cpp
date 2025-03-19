// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// #include "debug/dprint.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_rows = get_arg_val<uint32_t>(1);
    uint32_t half_row_width = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_out_no_mul = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out_mul = get_compile_time_arg_val(1);
    constexpr bool dst_is_dram = get_compile_time_arg_val(2) == 1;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out_no_mul);
    const DataFormat data_format = get_dataformat(cb_id_out_no_mul);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t out_no_mul_curr_id = start_id + half_row_width;
    uint32_t out_mul_curr_id = start_id;
    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_rows; i++) {
        for (uint32_t j = 0; j < half_row_width; j++) {
            cb_wait_front(cb_id_out_no_mul, onetile);
            uint32_t out_no_mul_l1_read_addr = get_read_ptr(cb_id_out_no_mul);
            noc_async_write_tile(out_no_mul_curr_id, s, out_no_mul_l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_out_no_mul, onetile);
            out_no_mul_curr_id++;

            cb_wait_front(cb_id_out_mul, onetile);
            uint32_t out_mul_l1_read_addr = get_read_ptr(cb_id_out_mul);
            noc_async_write_tile(out_mul_curr_id, s, out_mul_l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_out_mul, onetile);
            out_mul_curr_id++;
        }
        out_no_mul_curr_id += half_row_width;
        out_mul_curr_id += half_row_width;
    }
}
