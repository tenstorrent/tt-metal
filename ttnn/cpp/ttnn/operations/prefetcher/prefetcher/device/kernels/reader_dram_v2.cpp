// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

#include "debug/dprint.h"

FORCE_INLINE uint32_t get_fifo_start_address(uint32_t cb_id) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    uint32_t fifo_size = local_cb.fifo_size;
    uint32_t fifo_limit = local_cb.fifo_limit;
    uint32_t fifo_start_addr = fifo_limit - fifo_size;
    return fifo_start_addr;
}

FORCE_INLINE uint32_t get_fifo_start_size(uint32_t cb_id) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    uint32_t fifo_size = local_cb.fifo_size;
    return fifo_size;
}

FORCE_INLINE void resize_local_cb_interface(
    uint32_t cb_id, uint32_t page_size, uint32_t fifo_start_addr, uint32_t fifo_start_size) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);

    uint32_t fifo_limit = local_cb.fifo_limit;

    uint32_t fifo_wr_ptr = local_cb.fifo_wr_ptr;
    uint32_t fifo_rd_ptr = local_cb.fifo_rd_ptr;
    uint32_t fifo_size_page_aligned = fifo_start_size - fifo_start_size % page_size;
    uint32_t fifo_limit_page_aligned = fifo_start_addr + fifo_size_page_aligned;
    uint32_t fifo_num_pages = fifo_size_page_aligned / page_size;

    uint32_t next_fifo_wr_ptr = fifo_start_addr + align(fifo_wr_ptr - fifo_start_addr, page_size);
    if (next_fifo_wr_ptr >= fifo_limit_page_aligned) {
        next_fifo_wr_ptr = fifo_start_addr;
    }
    uint32_t next_fifo_rd_ptr = fifo_start_addr + align(fifo_rd_ptr - fifo_start_addr, page_size);
    if (next_fifo_rd_ptr >= fifo_limit_page_aligned) {
        next_fifo_rd_ptr = fifo_start_addr;
    }
    local_cb.fifo_limit = fifo_limit_page_aligned;
    local_cb.fifo_size = fifo_size_page_aligned;
    local_cb.fifo_num_pages = fifo_num_pages;
    local_cb.fifo_page_size = page_size;
    local_cb.fifo_wr_ptr = next_fifo_wr_ptr;
    local_cb.fifo_rd_ptr = next_fifo_rd_ptr;
}

FORCE_INLINE uint32_t get_local_cb_rd_ptr(uint32_t cb_id, uint32_t fifo_start_addr) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    return local_cb.fifo_rd_ptr - fifo_start_addr;
}
FORCE_INLINE uint32_t get_local_cb_wr_ptr(uint32_t cb_id, uint32_t fifo_start_addr) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    return local_cb.fifo_wr_ptr - fifo_start_addr;
}

void kernel_main() {
    constexpr uint32_t num_layers = get_compile_time_arg_val(0);
    constexpr uint32_t num_tensors = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t read_cb_size = get_compile_time_arg_val(3);
    constexpr uint32_t max_block_num_tiles = get_compile_time_arg_val(4);

    constexpr uint32_t cb_id = 0;        // Reader cb
    constexpr uint32_t addrs_cb_id = 1;  // Tensor specs
    constexpr uint32_t sync_cb = 2;

    uint32_t fifo_start_address = get_fifo_start_address(cb_id);
    uint32_t fifo_start_size = get_fifo_start_size(cb_id);

    // TODO: Take noc as input?
    uint32_t rt_args_idx = 0;
    const uint32_t bank_id = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t vc = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t total_num_blocks_in_buffer = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t* page_sizes = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* block_num_pages = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* block_num_tiles = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));

    uint32_t l1_buffer_start_addr = get_write_ptr(cb_id);
    uint32_t l1_buffer_end_addr = get_write_ptr(cb_id) + read_cb_size;

    volatile tt_l1_ptr uint32_t* tensor_addrs_l1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(addrs_cb_id));

    for (uint32_t layer = 0; layer < num_layers; layer++) {
        DeviceZoneScopedN("layers");
        for (uint32_t t = 0; t < num_tensors; t++) {
            DeviceZoneScopedN("tensors");
            uint32_t curr_page_size = page_sizes[t];
            uint32_t curr_block_num_pages = block_num_pages[t];
            uint32_t curr_block_num_tiles = block_num_tiles[t];
            uint32_t curr_block_size_bytes = curr_block_num_pages * curr_page_size;

            // Address setup
            uint32_t tensor_base_address = tensor_addrs_l1[layer * num_tensors + t];  // tensor_addrs_l1[t][layer];
            uint32_t src_base_addr =
                noc_async_read_tile_dram_sharded_set_state<true>(tensor_base_address, curr_page_size, bank_id, vc);
            uint32_t src_read_addr = 0;
            {
                // DeviceZoneScopedN("tensor_read");
                for (uint32_t block = 0; block < num_blocks; ++block) {
                    cb_reserve_back(cb_id, max_block_num_tiles);
                    auto l1_write_addr = get_write_ptr(cb_id);

                    DPRINT << "reader max_block_num_tiles " << max_block_num_tiles << ENDL();

                    for (uint32_t h = 0; h < curr_block_num_pages; ++h) {
                        noc_async_read_tile_dram_sharded_with_state(src_base_addr, src_read_addr, l1_write_addr);
                        src_read_addr += curr_page_size;
                        l1_write_addr += curr_page_size;
                    }

                    noc_async_read_barrier();

                    cb_push_back(cb_id, max_block_num_tiles);
                }
            }
        }
    }
}
