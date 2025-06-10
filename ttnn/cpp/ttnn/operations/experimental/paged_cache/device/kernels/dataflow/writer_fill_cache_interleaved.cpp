// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// Define the sentinel value for a page table entry that indicates a skip.
constexpr uint32_t SKIP_PAGE_TABLE_ENTRY = (uint32_t)-1;

template <uint32_t num_heads, uint32_t block_size_t, uint32_t Wt>
uint32_t virtual_seq_tile_id_to_physical_tile_id(
    uint32_t seq_tile_idx, uint32_t cur_head, volatile tt_l1_ptr const uint32_t* const page_table_ptr) {
    // Given some index in the sequence tiles in range [0, max_seq_len_t]
    // Return the physical tile id for that tile row, or SKIP_PAGE_TABLE_ENTRY if block is skipped
    constexpr uint32_t block_stride = num_heads * block_size_t * Wt;
    const uint32_t head_offset = cur_head * block_size_t * Wt;

    const uint32_t virtual_block = seq_tile_idx / block_size_t;
    const uint32_t physical_block = page_table_ptr[virtual_block];

    if (physical_block == SKIP_PAGE_TABLE_ENTRY) {
        return SKIP_PAGE_TABLE_ENTRY;  // Return sentinel to indicate skip
    }

    const uint32_t block_row_offset = seq_tile_idx % block_size_t;
    const uint32_t block_offset = block_row_offset * Wt;
    return physical_block * block_stride + head_offset + block_offset;
}

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t page_table_addr = get_arg_val<uint32_t>(1);
    uint32_t start_row_num = get_arg_val<uint32_t>(2);
    uint32_t num_rows = get_arg_val<uint32_t>(3);
    // Arg 4 is either batch_idx_tensor_addr or batch_idx_fallback scalar
    uint32_t batch_arg = get_arg_val<uint32_t>(4);

    constexpr bool out_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool page_table_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t cb_id_in = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_page_table = get_compile_time_arg_val(3);
    constexpr uint32_t num_heads = get_compile_time_arg_val(4);
    constexpr uint32_t num_blocks_of_work_per_head = get_compile_time_arg_val(5);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t log2_page_table_stick_size = get_compile_time_arg_val(8);
    constexpr uint32_t page_table_stick_size = get_compile_time_arg_val(9);

    // New compile-time args for optional batch_idx_tensor
    constexpr bool use_batch_idx_tensor = get_compile_time_arg_val(10) == 1;
    constexpr uint32_t cb_id_batch_idx = get_compile_time_arg_val(11);  // CB for reading from batch_idx_tensor
    constexpr bool batch_idx_tensor_is_dram = get_compile_time_arg_val(12) == 1;
    constexpr uint32_t batch_idx_stick_size =
        get_compile_time_arg_val(13);  // Expected to be small (e.g., 4 for uint32)

    uint32_t batch_idx;
    if constexpr (use_batch_idx_tensor) {
        uint32_t batch_idx_tensor_addr = batch_arg;  // Arg 4 is the address

        const InterleavedAddrGen<batch_idx_tensor_is_dram> batch_idx_gen = {
            .bank_base_address = batch_idx_tensor_addr, .page_size = batch_idx_stick_size};
        cb_reserve_back(cb_id_batch_idx, 1);  // Expecting 1 element (the batch_idx)
        uint32_t batch_idx_cb_wr_ptr = get_write_ptr(cb_id_batch_idx);
        uint64_t batch_idx_noc_addr = get_noc_addr(0, batch_idx_gen);
        noc_async_read(batch_idx_noc_addr, batch_idx_cb_wr_ptr, batch_idx_stick_size);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* batch_idx_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_idx_cb_wr_ptr);
        batch_idx = batch_idx_ptr[0];
    } else {
        batch_idx = batch_arg;  // Arg 4 is the scalar fallback value
    }

    const uint32_t tile_bytes = get_tile_size(cb_id_in);
    const DataFormat data_format = get_dataformat(cb_id_in);

    const InterleavedAddrGenFast<out_is_dram> out_gen = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGen<page_table_is_dram> page_table_gen = {
        .bank_base_address = page_table_addr, .page_size = page_table_stick_size};
    cb_reserve_back(cb_id_page_table, 1);
    uint32_t page_table_cb_wr_ptr = get_write_ptr(cb_id_page_table);
    uint64_t page_table_noc_addr = get_noc_addr(batch_idx, page_table_gen);
    noc_async_read(page_table_noc_addr, page_table_cb_wr_ptr, page_table_stick_size);
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* page_table_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_cb_wr_ptr);

    for (uint32_t row_id = start_row_num; row_id < start_row_num + num_rows; ++row_id) {
        uint32_t cur_head = row_id / num_blocks_of_work_per_head;
        uint32_t seq_tile_id = row_id % num_blocks_of_work_per_head;
        uint32_t physical_tile_id =
            virtual_seq_tile_id_to_physical_tile_id<num_heads, block_size_t, Wt>(seq_tile_id, cur_head, page_table_ptr);

        if (physical_tile_id == SKIP_PAGE_TABLE_ENTRY) {
            // Block should be skipped. Consume the input tiles from the CB and discard.
            cb_wait_front(cb_id_in, Wt);
            cb_pop_front(cb_id_in, Wt);
        } else {
            // Valid block, proceed with writing.
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
}
