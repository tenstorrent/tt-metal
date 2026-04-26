// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"

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

    experimental::CircularBuffer cb_in0(cb_id_in0);
    experimental::CircularBuffer cb_in1(cb_id_in1);
    experimental::CircularBuffer cb_pad(pad_cb);

    cb_in0.reserve_back(num_input_rows);

    cb_in1.reserve_back(num_padded_tiles_per_batch);

    cb_pad.reserve_back(1);

    // Keep legacy NOC API for local L1 reads (sharded kernel)
    uint64_t read_noc_addr = get_noc_addr(cb_in0.get_read_ptr());
    uint32_t write_addr = cb_in1.get_write_ptr();
    uint32_t pad_addr = cb_pad.get_write_ptr();
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
    cb_in1.push_back(num_padded_tiles_per_batch);

    for (uint32_t b = 1; b < num_batches; ++b) {
        cb_in1.reserve_back(num_padded_tiles_per_batch);
        write_addr = cb_in1.get_write_ptr();
        noc_async_read(read_noc_addr, write_addr, input_block_size);
        read_noc_addr += input_block_size;
        write_addr += input_block_size;
        for (uint32_t i = 0; i < num_padded_rows; ++i) {
            noc_async_read(pad_noc_addr, write_addr, input_width_bytes);
            write_addr += input_width_bytes;
        }
        noc_async_read_barrier();
        cb_in1.push_back(num_padded_tiles_per_batch);
    }
}
