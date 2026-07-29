// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    uint32_t rt_args_idx = 0;
    const bool has_work = get_arg_val<uint32_t>(rt_args_idx++);
    if (!has_work) {
        return;
    }
    const bool is_input1 = get_arg_val<uint32_t>(rt_args_idx++);

    const uint32_t cache_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t cache_start_id = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t index_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t my_batch_idx = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t page_table_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const bool wait_to_start_signal = get_arg_val<uint32_t>(rt_args_idx++) == 1;

    constexpr uint32_t input1_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t input2_cb_id = get_compile_time_arg_val(1);
    uint32_t input_cb_id = input1_cb_id;
    if (!is_input1) {
        input_cb_id = input2_cb_id;
    }

    constexpr uint32_t cache_cb_id = get_compile_time_arg_val(2);
    constexpr bool use_index_tensor = get_compile_time_arg_val(3) == 1;
    constexpr bool index_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t cb_index_id = get_compile_time_arg_val(5);
    constexpr uint32_t cache_batch_num_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    const uint32_t log_base_2_of_page_size = get_compile_time_arg_val(8);
    const uint32_t index_stick_size_B = get_compile_time_arg_val(9);

    // paged_cache args
    constexpr bool is_paged_cache = get_compile_time_arg_val(10) == 1;
    constexpr uint32_t num_heads = get_compile_time_arg_val(11);
    constexpr uint32_t block_size = get_compile_time_arg_val(12);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(13);
    constexpr uint32_t max_blocks_per_seq = get_compile_time_arg_val(14);
    constexpr uint32_t log2_page_table_stick_size = get_compile_time_arg_val(15);
    constexpr uint32_t page_table_stick_size = get_compile_time_arg_val(16);
    constexpr uint32_t page_table_is_dram = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t page_table_cb_id = get_compile_time_arg_val(18);

    const uint32_t St = get_compile_time_arg_val(19);
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(20);  // semaphore for receiver
    constexpr uint32_t batch_size = get_compile_time_arg_val(21);

    constexpr auto s0_args = TensorAccessorArgs<22>();
    constexpr auto index_tensor_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<index_tensor_args.next_compile_time_args_offset()>();

    constexpr uint32_t head_offset_t = Wt * St;

    CircularBuffer cb_input(input_cb_id);
    CircularBuffer cb_cache(cache_cb_id);
    CircularBuffer cb_index(cb_index_id);
    CircularBuffer cb_page_table(page_table_cb_id);

    // Kick off compute
    cb_input.reserve_back(Wt);
    cb_input.push_back(Wt);

    const uint32_t cache_tile_bytes = get_tile_size(cache_cb_id);
    const DataFormat cache_data_format = get_dataformat(cache_cb_id);

    constexpr uint32_t TILE_HEIGHT = 32;

    uint32_t cache_id = cache_start_id;

    const auto s0 = TensorAccessor(s0_args, cache_addr);

    bool skip_update = false;

    if constexpr (use_index_tensor) {
        const auto addrg = TensorAccessor(index_tensor_args, index_tensor_addr);
        cb_index.reserve_back(1);
        uint32_t index_cb_wr_ptr = cb_index.get_write_ptr();
        if constexpr (index_is_dram) {
            // index_tensor has one page to read
            noc.async_read(addrg, CoreLocalMem<uint32_t>(index_cb_wr_ptr), index_stick_size_B, {.page_id = 0}, {});
            noc.async_read_barrier();
        }
        cb_index.push_back(1);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr);

        const uint32_t update_idx = index_ptr[my_batch_idx];
        if (update_idx == (uint32_t)-1) {
            // Passing update_idx = -1 tells us to skip update for this user
            skip_update = true;
        } else {
            if constexpr (is_paged_cache) {
                uint32_t num_pages_to_read = page_table_is_dram ? 1 : batch_size;
                cb_page_table.reserve_back(num_pages_to_read);
                uint32_t page_table_cb_wr_ptr = cb_page_table.get_write_ptr();

                if constexpr (page_table_is_dram) {
                    const auto page_table_gen = TensorAccessor(page_table_args, page_table_tensor_addr);
                    noc.async_read(
                        page_table_gen,
                        CoreLocalMem<uint32_t>(page_table_cb_wr_ptr),
                        page_table_stick_size,
                        {.page_id = my_batch_idx},
                        {});
                    noc.async_read_barrier();
                } else {
                    page_table_cb_wr_ptr += my_batch_idx * page_table_stick_size;
                }

                cb_page_table.push_back(num_pages_to_read);
                // DRAM uses uint32 entries; L1-sharded page table uses uint16 entries
                volatile tt_l1_ptr uint32_t* page_table_ptr_u32 = nullptr;
                volatile tt_l1_ptr uint16_t* page_table_ptr_u16 = nullptr;

                if constexpr (page_table_is_dram) {
                    page_table_ptr_u32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_cb_wr_ptr);
                } else {
                    page_table_ptr_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(page_table_cb_wr_ptr);
                }

                const uint32_t virtual_block_id = update_idx / block_size;
                const uint32_t physical_block_id = (page_table_is_dram)
                                                       ? page_table_ptr_u32[virtual_block_id]
                                                       : static_cast<uint32_t>(page_table_ptr_u16[virtual_block_id]);

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
        // wait for signal from writer that it has finished using the input CB
        Semaphore<> receiver_sem(receiver_sem_id);
        receiver_sem.wait(1);
        receiver_sem.set(0);
    }

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        cb_cache.reserve_back(Wt);
        if (!skip_update) {
            uint32_t cache_l1_write_addr = cb_cache.get_write_ptr();
            for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                noc.async_read(
                    s0, CoreLocalMem<uint32_t>(cache_l1_write_addr), cache_tile_bytes, {.page_id = curr_cache_id}, {});
                cache_l1_write_addr += cache_tile_bytes;
            }

            noc.async_read_barrier();
        }
        cb_cache.push_back(Wt);

        cache_id += head_offset_t;
    }
}
