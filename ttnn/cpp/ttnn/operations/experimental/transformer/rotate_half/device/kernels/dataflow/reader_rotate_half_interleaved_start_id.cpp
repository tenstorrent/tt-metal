// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

//#include "debug/dprint.h"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_rows = get_arg_val<uint32_t>(1);
    uint32_t half_row_size = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in_no_mul = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in_mul = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_in_scalar = get_compile_time_arg_val(2);
    constexpr bool src_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr uint16_t scalar_value = get_compile_time_arg_val(4);



    // in_no_mul, in_mul are from same tensor, so same sizes
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in_no_mul);
    const DataFormat data_format = get_dataformat(cb_id_in_no_mul);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    // Fill tile with zeros
    cb_reserve_back(cb_id_in_scalar, onetile);
    uint32_t l1_zeros_addr_in_scalar = get_write_ptr(cb_id_in_scalar);
    volatile tt_l1_ptr uint16_t* scalar_buffer = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_zeros_addr_in_scalar);
    scalar_buffer[0] = scalar_value;
    cb_push_back(cb_id_in_scalar, onetile);

    uint32_t in_no_mul_curr_id = start_id;
    uint32_t in_mul_curr_id = start_id + half_row_size;
    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = 0; i<num_rows; i ++) {
        for (uint32_t j = 0; j < half_row_size; j++) {
            cb_reserve_back(cb_id_in_no_mul, onetile);
            uint32_t in_no_mul_l1_write_addr = get_write_ptr(cb_id_in_no_mul);
            noc_async_read_tile(in_no_mul_curr_id, s, in_no_mul_l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_in_no_mul, onetile);
            in_no_mul_curr_id++;

            cb_reserve_back(cb_id_in_mul, onetile);
            uint32_t in1_l1_write_addr = get_write_ptr(cb_id_in_mul);
            noc_async_read_tile(in_mul_curr_id, s, in1_l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_in_mul, onetile);
            in_mul_curr_id++;
        }
        in_no_mul_curr_id += half_row_size;
        in_mul_curr_id += half_row_size;
    }
}
