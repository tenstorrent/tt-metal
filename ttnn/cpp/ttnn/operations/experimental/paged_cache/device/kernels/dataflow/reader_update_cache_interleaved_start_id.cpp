// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cache_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);

    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t cache_start_id = get_arg_val<uint32_t>(1);
    const uint32_t index_tensor_addr = get_arg_val<uint32_t>(2);
    const uint32_t my_batch_idx = get_arg_val<uint32_t>(3);
    const uint32_t page_table_tensor_addr = get_arg_val<uint32_t>(4);
    const bool wait_to_start_signal = get_arg_val<uint32_t>(5) == 1;
    const uint32_t noop = get_arg_val<uint32_t>(6);

    if (noop == 1) {
        return;  // Early exit, no work done
    }

    constexpr bool use_index_tensor = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t cb_index_id = get_compile_time_arg_val(3);
    constexpr uint32_t cache_batch_num_tiles = get_compile_time_arg_val(4);
    const uint32_t log_base_2_of_page_size = get_compile_time_arg_val(6);
    const uint32_t index_stick_size_B = get_compile_time_arg_val(7);

    // paged_cache args
    constexpr bool is_paged_cache = get_compile_time_arg_val(8) == 1;
    constexpr uint32_t num_heads = get_compile_time_arg_val(9);
    constexpr uint32_t block_size = get_compile_time_arg_val(10);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(11);
    constexpr uint32_t max_blocks_per_seq = get_compile_time_arg_val(12);
    constexpr uint32_t log2_page_table_stick_size = get_compile_time_arg_val(13);
    constexpr uint32_t page_table_stick_size = get_compile_time_arg_val(14);
    constexpr uint32_t page_table_cb_id = get_compile_time_arg_val(15);

    const uint32_t St = get_compile_time_arg_val(16);
    uint32_t semaphore_addr = get_semaphore(get_compile_time_arg_val(17));  // semaphore for receiver

    constexpr auto s0_args = TensorAccessorArgs<18>();
    constexpr auto index_tensor_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<index_tensor_args.next_compile_time_args_offset()>();

    constexpr uint32_t head_offset_t = Wt * St;

    // Kick off compute
    cb_reserve_back(input_cb_id, Wt);
    cb_push_back(input_cb_id, Wt);

    const uint32_t cache_tile_bytes = get_tile_size(cache_cb_id);
    const DataFormat cache_data_format = get_dataformat(cache_cb_id);

    constexpr uint32_t TILE_HEIGHT = 32;

    uint32_t cache_id = cache_start_id;

    const auto s0 = TensorAccessor(s0_args, cache_addr, cache_tile_bytes);

    bool skip_update = false;

    if constexpr (use_index_tensor) {
        const auto addrg = TensorAccessor(index_tensor_args, index_tensor_addr, index_stick_size_B);

        cb_reserve_back(cb_index_id, 1);
        uint32_t index_cb_wr_ptr = get_write_ptr(cb_index_id);
        // index_tensor has one page to read
        uint64_t tensor_index_noc_addr = addrg.get_noc_addr(0);
        noc_async_read(tensor_index_noc_addr, index_cb_wr_ptr, index_stick_size_B);
        noc_async_read_barrier();
        cb_push_back(cb_index_id, 1);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr);

        const uint32_t update_idx = index_ptr[my_batch_idx];
        if (update_idx == (uint32_t)-1) {
            // Passing update_idx = -1 tells us to skip update for this user
            skip_update = true;
        } else {
            if constexpr (is_paged_cache) {
                const auto page_table_gen =
                    TensorAccessor(page_table_args, page_table_tensor_addr, page_table_stick_size);
                cb_reserve_back(page_table_cb_id, 1);
                uint32_t page_table_cb_wr_ptr = get_write_ptr(page_table_cb_id);
                uint64_t page_table_noc_addr = page_table_gen.get_noc_addr(my_batch_idx);
                noc_async_read(page_table_noc_addr, page_table_cb_wr_ptr, page_table_stick_size);
                noc_async_read_barrier();
                cb_push_back(page_table_cb_id, 1);
                volatile tt_l1_ptr uint32_t* page_table_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_cb_wr_ptr);

                const uint32_t virtual_block_id = update_idx / block_size;
                const uint32_t physical_block_id = page_table_ptr[virtual_block_id];
                const uint32_t block_start_id = physical_block_id * num_heads * block_size_t * Wt;
                const uint32_t block_row_tile = (update_idx % block_size) / TILE_HEIGHT;
                const uint32_t block_offset = block_row_tile * Wt;
                cache_id = block_start_id + block_offset;

            } else {
                const uint32_t cache_batch_tile_offset = my_batch_idx * cache_batch_num_tiles;
                const uint32_t cache_start_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
                cache_id = cache_start_id;
            }
        }
    }

    if (wait_to_start_signal) {
        // wait for signal to start pushing tensor
        volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);
        noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, 1);
        noc_semaphore_set(in0_receiver_semaphore_addr_ptr, 0);
    }

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        cb_reserve_back(cache_cb_id, Wt);
        if (!skip_update) {
            uint32_t cache_l1_write_addr = get_write_ptr(cache_cb_id);
            for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                noc_async_read_tile(curr_cache_id, s0, cache_l1_write_addr);
                cache_l1_write_addr += cache_tile_bytes;
            }

            noc_async_read_barrier();
        }
        cb_push_back(cache_cb_id, Wt);

        cache_id += head_offset_t;
    }
}
