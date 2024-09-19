// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

template<uint32_t num_heads, uint32_t block_size_t, uint32_t Wt>
uint32_t virtual_seq_tile_id_to_physical_tile_id(uint32_t seq_tile_idx, volatile tt_l1_ptr const uint32_t* const page_table_ptr) {
    // Given some index in the sequence tiles in range [0, max_seq_len_t]
    // Return the physical tile id for that tile row
    constexpr uint32_t block_stride = num_heads * block_size_t * Wt;

    const uint32_t virtual_block = seq_tile_idx / block_size_t;
    const uint32_t physical_block = page_table_ptr[virtual_block];
    const uint32_t block_row_offset = seq_tile_idx % block_size_t;
    const uint32_t block_offset = block_row_offset * Wt;
    return physical_block * block_stride + block_offset;
}

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t page_table_addr = get_arg_val<uint32_t>(1);
    uint32_t start_row_num = get_arg_val<uint32_t>(2);
    uint32_t num_rows = get_arg_val<uint32_t>(3);
    uint32_t batch_idx = get_arg_val<uint32_t>(4);

    constexpr bool out_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool page_table_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t cb_id_in = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_page_table = get_compile_time_arg_val(3);
    constexpr uint32_t num_heads = get_compile_time_arg_val(4);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr uint32_t log2_page_table_stick_size = get_compile_time_arg_val(7);
    constexpr uint32_t page_table_stick_size = get_compile_time_arg_val(8);


    const uint32_t tile_bytes = get_tile_size(cb_id_in);
    const DataFormat data_format = get_dataformat(cb_id_in);

    const InterleavedAddrGenFast<out_is_dram> out_gen = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };


    const InterleavedAddrGen<page_table_is_dram> page_table_gen = {
        .bank_base_address = page_table_addr,
        .page_size = page_table_stick_size
    };
    cb_reserve_back(cb_id_page_table, 1);
    uint32_t page_table_cb_wr_ptr = get_write_ptr(cb_id_page_table);
    uint64_t page_table_noc_addr = get_noc_addr(batch_idx, page_table_gen);
    noc_async_read(page_table_noc_addr, page_table_cb_wr_ptr, page_table_stick_size);
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* page_table_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_cb_wr_ptr);

    for (uint32_t row_id = start_row_num; row_id < start_row_num + num_rows; ++row_id) {
        uint32_t physical_tile_id = virtual_seq_tile_id_to_physical_tile_id<num_heads, block_size_t, Wt>(row_id, page_table_ptr);
        cb_wait_front(cb_id_in, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_id_in);
        for (uint32_t w = 0; w < Wt; ++w) {
            noc_async_write_tile(physical_tile_id, out_gen, l1_read_addr);
            l1_read_addr += tile_bytes;
            physical_tile_id += 1;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_in, Wt);
    }

}
