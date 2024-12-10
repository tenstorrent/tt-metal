// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

#include "debug/dprint.h"

#define MAX_NUM_TENSORS 8

void get_tensor_addrs_l1(
    const uint32_t num_layers,
    const uint32_t num_tensors,
    const uint32_t addrs_cb_id,
    volatile tt_l1_ptr uint32_t** tensor_addrs_l1) {
    volatile tt_l1_ptr uint32_t* l1_cb_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(addrs_cb_id));

    for (uint32_t t = 0; t < num_tensors; t++) {
        tensor_addrs_l1[t] = l1_cb_addr;
        l1_cb_addr += num_layers;
    }
}

// [t1_1, t1_2, ..., t1_80, t2_1, t2_2, ..., t2_80, ..., t5_1, t5_2, ..., t5_80]

void kernel_main() {
    constexpr uint32_t num_layers = get_compile_time_arg_val(0);
    constexpr uint32_t num_tensors = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t read_cb_size = get_compile_time_arg_val(3);

    DPRINT << "num_layers: " << num_layers << ENDL();
    DPRINT << "num_tensors: " << num_tensors << ENDL();
    DPRINT << "num_blocks: " << num_blocks << ENDL();
    DPRINT << "read_cb_size: " << read_cb_size << ENDL();

    constexpr uint32_t cb_id = 0;        // Reader cb
    constexpr uint32_t addrs_cb_id = 1;  // Tensor specs
    constexpr uint32_t out_cb_id = 2;    // Output cb

    uint32_t rt_args_idx = 0;
    const uint32_t bank_id = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t vc = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t total_num_blocks_in_buffer = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t* page_sizes = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* block_num_tiles = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));

    DPRINT << "bank_id: " << bank_id << ENDL();
    DPRINT << "vc: " << vc << ENDL();
    DPRINT << "total_num_blocks_in_buffer: " << total_num_blocks_in_buffer << ENDL();

    uint32_t l1_buffer_start_addr = get_write_ptr(cb_id);
    uint32_t l1_buffer_end_addr = get_write_ptr(cb_id) + read_cb_size;

    volatile tt_l1_ptr uint32_t* tensor_addrs_l1[MAX_NUM_TENSORS];
    get_tensor_addrs_l1(num_layers, num_tensors, addrs_cb_id, tensor_addrs_l1);

    for (uint32_t layer = 0; layer < num_layers; layer++) {
        for (uint32_t t = 0; t < num_tensors; t++) {
            uint32_t curr_page_size = page_sizes[t];
            uint32_t curr_block_num_tiles = block_num_tiles[t];
            uint32_t curr_block_size_bytes = curr_block_num_tiles * curr_page_size;

            // // DPRINT some info
            // DPRINT << "curr_page_size: " << curr_page_size << ENDL();
            // DPRINT << "curr_block_num_tiles: " << curr_block_num_tiles << ENDL();
            // DPRINT << "curr_block_size_bytes: " << curr_block_size_bytes << ENDL();

            // Address setup
            uint32_t tensor_base_address = tensor_addrs_l1[t][layer];
            DPRINT << "tensor_base_address: " << tensor_base_address << ENDL();
            uint32_t src_base_addr =
                noc_async_read_tile_dram_sharded_set_state<true>(tensor_base_address, curr_page_size, bank_id, vc);
            uint32_t src_read_addr = 0;

            uint32_t num_free_blocks_in_buffer = total_num_blocks_in_buffer;
            uint32_t curr_block_trid = 1;
            uint32_t block_trid_to_wait = 1;

            // TODO: In what case does reserve back pass, but then the write ptr
            // is past the end of the buffer?? Why is this check needed here?
            // cb_reserve_back(cb_id, curr_block_num_tiles);
            uint32_t l1_write_addr_offset = 0;
            uint32_t l1_write_addr_start = get_write_ptr(cb_id);
            if (l1_write_addr_start >= l1_buffer_end_addr) {
                l1_write_addr_start = l1_buffer_start_addr;
            }
            uint32_t l1_write_addr = l1_write_addr_start;
            DPRINT << "l1_write_addr: " << l1_write_addr << ENDL();

            for (uint32_t block = 0; block < num_blocks; ++block) {
                cb_reserve_back(cb_id, curr_block_num_tiles);
                for (uint32_t h = 0; h < curr_block_num_tiles; ++h) {
                    noc_async_read_tile_dram_sharded_with_state(src_base_addr, src_read_addr, l1_write_addr);
                    src_read_addr += curr_page_size;
                    l1_write_addr += curr_page_size;
                }

                noc_async_read_barrier();
                cb_push_back(cb_id, curr_block_num_tiles);

                // // Dprint a tile from the buffer
                // for (uint32_t i = 0; i < 4; i++) {
                //     uint32_t value = *(volatile uint32_t*)(l1_write_addr_start + i);
                //     DPRINT << "tile[" << i << "]: " << value << ENDL();
                // }
            }

            // Copy from reader to output
            auto l1_read_addr = l1_write_addr_start;
            DPRINT << "out_write_addr: " << get_write_ptr(out_cb_id) << ENDL();
            DPRINT << "l1_read_addr: " << l1_read_addr << ENDL();
            for (uint32_t block = 0; block < num_blocks; ++block) {
                cb_reserve_back(out_cb_id, curr_block_num_tiles);
                auto l1_out_write_addr = get_noc_addr(get_write_ptr(out_cb_id));

                noc_async_write(l1_read_addr, l1_out_write_addr, curr_block_size_bytes);
                l1_read_addr += curr_block_size_bytes;
                cb_push_back(out_cb_id, curr_block_num_tiles);
            }

            // for (uint32_t block = 0; block < num_blocks; block++) {
            //     noc_async_read_tile_dram_sharded_set_trid(curr_block_trid);

            //     uint32_t temp_l1_write_addr = l1_write_addr;
            //     for (uint32_t h = 0; h < curr_block_num_tiles; ++h) {
            //         noc_async_read_tile_dram_sharded_with_state_with_trid(
            //             src_base_addr, src_read_addr, temp_l1_write_addr, curr_block_trid);
            //         src_read_addr += curr_page_size;
            //         temp_l1_write_addr += curr_page_size;
            //     }

            //     // TODO: Write comments here explaining what's going on
            //     if (num_free_blocks_in_buffer == 2) {
            //         noc_async_read_barrier_with_trid(block_trid_to_wait);
            //         cb_push_back(cb_id, curr_block_num_tiles);
            //         // wait for next block trid
            //         block_trid_to_wait = block_trid_to_wait == 3 ? 1 : (block_trid_to_wait + 1);
            //         // reserve for next block
            //         cb_reserve_back(cb_id, curr_block_num_tiles * 2);
            //     } else {
            //         num_free_blocks_in_buffer -= 1;
            //     }

            //     if (curr_block_trid == total_num_blocks_in_buffer) {
            //         curr_block_trid = 1;
            //     } else {
            //         curr_block_trid += 1;
            //     }

            //     // Is this needed? shouldn't the CB internally handle overflow?
            //     l1_write_addr += curr_block_size_bytes;
            //     if (l1_write_addr >= l1_buffer_end_addr) {
            //         l1_write_addr = l1_buffer_start_addr;
            //     }
            // }

            // // last block to wait
            // noc_async_read_barrier_with_trid(block_trid_to_wait);
            // cb_push_back(cb_id, curr_block_num_tiles);
        }
    }
}
