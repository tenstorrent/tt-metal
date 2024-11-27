// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t num_input_rows = get_arg_val<uint32_t>(0);
    const uint32_t input_width_bytes = get_arg_val<uint32_t>(1);
    const uint32_t input_block_size = get_arg_val<uint32_t>(2);
    const uint32_t num_padded_tiles_per_batch = get_arg_val<uint32_t>(3);
    const uint32_t num_padded_rows = get_arg_val<uint32_t>(4);
    const uint32_t num_batches = get_arg_val<uint32_t>(5);
    const uint32_t packed_pad_value = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t pad_cb = get_compile_time_arg_val(2);

    cb_reserve_back(cb_id_in0, num_input_rows);

    cb_reserve_back(cb_id_in1, num_padded_tiles_per_batch);

    cb_reserve_back(pad_cb, 1);

    uint64_t read_noc_addr = get_noc_addr(get_read_ptr(cb_id_in0));
    uint32_t write_addr = get_write_ptr(cb_id_in1);
    uint32_t pad_addr = get_write_ptr(pad_cb);
    uint64_t pad_noc_addr = get_noc_addr(pad_addr);

    noc_async_read(read_noc_addr, write_addr, input_block_size);
    read_noc_addr += input_block_size;
    write_addr += input_block_size;
    volatile tt_l1_ptr std::uint32_t* pad = (volatile tt_l1_ptr uint32_t*)(pad_addr);
    for (uint32_t i = 0; i < input_width_bytes >> 2; ++i) {
        pad[i] = packed_pad_value;
    }
    for (uint32_t i = 0; i < num_padded_rows; ++i) {
        noc_async_read(pad_noc_addr, write_addr, input_width_bytes);
        write_addr += input_width_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id_in1, num_padded_tiles_per_batch);

    for (uint32_t b = 1; b < num_batches; ++b) {
        cb_reserve_back(cb_id_in1, num_padded_tiles_per_batch);
        write_addr = get_write_ptr(cb_id_in1);
        noc_async_read(read_noc_addr, write_addr, input_block_size);
        read_noc_addr += input_block_size;
        write_addr += input_block_size;
        for (uint32_t i = 0; i < num_padded_rows; ++i) {
            noc_async_read(pad_noc_addr, write_addr, input_width_bytes);
            write_addr += input_width_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in1, num_padded_tiles_per_batch);
    }
}
