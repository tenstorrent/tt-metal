// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t cache_start_id = get_arg_val<uint32_t>(1);
    uint32_t cache_tile_offset_B = get_arg_val<uint32_t>(2);
    const uint32_t my_batch_idx = get_arg_val<uint32_t>(3);
    const bool send_signal = get_arg_val<uint32_t>(4) == 1;
    const uint32_t send_core_x = get_arg_val<uint32_t>(5);
    const uint32_t send_core_y = get_arg_val<uint32_t>(6);

    constexpr uint32_t cache_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t untilized_cache_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t untilized_cache2_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t untilized_input_cb_id = get_compile_time_arg_val(3);
    constexpr bool use_index_tensor = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t cb_index_id = get_compile_time_arg_val(5);
    constexpr uint32_t cache_batch_num_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t Wbytes = get_compile_time_arg_val(8);

    // paged_cache args
    constexpr bool is_paged_cache = get_compile_time_arg_val(9) == 1;
    constexpr uint32_t num_heads = get_compile_time_arg_val(10);
    constexpr uint32_t block_size = get_compile_time_arg_val(11);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(12);
    constexpr uint32_t max_blocks_per_seq = get_compile_time_arg_val(13);
    constexpr uint32_t page_table_cb_id = get_compile_time_arg_val(14);

    constexpr uint32_t St = get_compile_time_arg_val(15);
    uint32_t semaphore_addr = get_semaphore(get_compile_time_arg_val(16));  // semaphore for receiver

    constexpr auto s0_args = TensorAccessorArgs<17>();

    constexpr uint32_t head_offset_t = Wt * St;

    const uint32_t cache_tile_bytes = get_tile_size(cache_cb_id);
    const DataFormat cache_data_format = get_dataformat(cache_cb_id);

    constexpr uint32_t TILE_HEIGHT = 32;

    const auto s0 = TensorAccessor(s0_args, cache_addr, cache_tile_bytes);

    uint32_t cache_id = cache_start_id;
    uint32_t update_idx = 0;

    bool skip_update = false;

    if constexpr (use_index_tensor) {
        cb_wait_front(cb_index_id, 1);
        uint32_t index_cb_ptr = get_read_ptr(cb_index_id);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_ptr);
        const uint32_t update_idx = index_ptr[my_batch_idx];

        if (update_idx == (uint32_t)-1) {
            // Passing update_idx = -1 tells us to skip update for this user
            skip_update = true;
        } else {
            if constexpr (is_paged_cache) {
                cb_wait_front(page_table_cb_id, 1);
                uint32_t page_table_cb_rd_ptr = get_read_ptr(page_table_cb_id);
                volatile tt_l1_ptr uint32_t* page_table_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_cb_rd_ptr);

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
            cache_tile_offset_B = update_idx % TILE_HEIGHT * Wbytes;
        }
    }

    cb_wait_front(untilized_input_cb_id, Wt);  // input tensor
    uint64_t input_l1_read_addr = get_noc_addr(get_read_ptr(untilized_input_cb_id));

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        // Wait on compute to untilize a block. Update that block in L1.
        cb_wait_front(untilized_cache_cb_id, Wt);
        cb_reserve_back(untilized_cache2_cb_id, Wt);

        uint32_t cache_l1_write_addr = get_read_ptr(untilized_cache_cb_id) + cache_tile_offset_B;
        noc_async_read(input_l1_read_addr, cache_l1_write_addr, Wbytes);
        noc_async_read_barrier();
        cb_push_back(untilized_cache2_cb_id, Wt);
        cb_pop_front(untilized_cache_cb_id, Wt);  // NEW

        // Wait on compute to tilize an updated block. Write that block to DRAM
        cb_wait_front(cache_cb_id, Wt);
        if (!skip_update) {
            uint32_t out_l1_read_addr = get_read_ptr(cache_cb_id);
            for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                noc_async_write_tile(curr_cache_id, s0, out_l1_read_addr);
                out_l1_read_addr += cache_tile_bytes;
            }

            noc_async_writes_flushed();
        }
        cb_pop_front(cache_cb_id, Wt);

        if (!skip_update) {
            // Delay syncing the writes to maximize perf.
            noc_async_write_barrier();
        }

        // read from next head
        input_l1_read_addr += Wbytes;
        cache_id += head_offset_t;
    }

    cb_pop_front(untilized_input_cb_id, Wt);

    if (send_signal) {
        // send signal to start compute
        const uint64_t in0_sender_semaphore_noc_addr = get_noc_addr(send_core_x, send_core_y, semaphore_addr);
        noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
        noc_async_atomic_barrier();
    }
}
