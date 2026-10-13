// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

#include "api/debug/dprint.h"

// Zeroes [l1_addr, l1_addr + size) with loopback NoC reads of the zeroed L1 range [zeros_addr, +zeros_size), in
// chunks of min(size, zeros_size); the last chunk is end-aligned and may overlap the previous one. Sets the read cmd
// buf state. The reads carry the trid of the last noc_async_read_set_trid (the trid argument only feeds the issue
// throttle), so the caller's trid barrier covers them. l1_addr, size: multiples of NOC_L1_READ_ALIGNMENT_BYTES.
template <bool skip_ptr_update, uint32_t zeros_size>
FORCE_INLINE void zero_fill_with_trid(
    uint32_t zeros_addr, uint32_t l1_addr, uint32_t size, uint32_t vc, uint32_t trid) {
    static_assert(zeros_size <= NOC_MAX_BURST_SIZE, "a zero-fill chunk must be a single NoC packet");
    ASSERT(size != 0);
    const uint32_t chunk = size < zeros_size ? size : zeros_size;
    noc_async_read_one_packet_set_state<true>(get_noc_addr(zeros_addr), chunk, vc);
    const uint32_t last_chunk_addr = l1_addr + size - chunk;
    for (; l1_addr < last_chunk_addr; l1_addr += chunk) {
        noc_async_read_one_packet_with_state_with_trid<skip_ptr_update>(zeros_addr, 0, l1_addr, trid);
    }
    noc_async_read_one_packet_with_state_with_trid<skip_ptr_update>(zeros_addr, 0, last_chunk_addr, trid);
}

void kernel_main() {
    // Compile time args
    constexpr uint32_t num_layers = get_compile_time_arg_val(0);
    constexpr uint32_t num_tensors = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t read_cb_size = get_compile_time_arg_val(3);
    constexpr uint32_t max_block_num_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t max_block_size = get_compile_time_arg_val(5);
    constexpr uint32_t cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t addrs_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t sync_cb_id = get_compile_time_arg_val(8);
    constexpr bool skip_ptr_update = get_compile_time_arg_val(9);
    constexpr uint32_t zero_cb_id = get_compile_time_arg_val(10);
    constexpr uint32_t zero_cb_size = get_compile_time_arg_val(11);
    constexpr bool any_tensor_padded = zero_cb_size != 0;

    // Runtime args
    uint32_t rt_args_idx = 0;
    const uint32_t bank_id = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t vc = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t total_num_blocks_in_buffer = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t* page_sizes = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* block_num_pages = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* block_num_tiles = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    // Padding contract: a tensor's shard height H (tiles) is rounded up to a multiple of num_blocks, and every block
    // is staged and sent at the padded size. Blocks [0, num_full_blocks) lie entirely inside the DRAM shard and are
    // read exactly as for an unpadded tensor. Block num_full_blocks reads its remaining real rows (if any, with
    // partial_block_page_sizes / partial_block_num_pages), and nothing at or beyond row H is read from DRAM: those
    // pad rows are delivered as zeros, copied from the zero CB. When H is a multiple of num_blocks,
    // num_full_blocks == num_blocks.
    const uint32_t* num_full_blocks = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* partial_block_page_sizes = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));
    const uint32_t* partial_block_num_pages = (uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors)));

    // Zero the zero CB once, from MEM_ZEROS, before any block is staged.
    uint32_t zero_cb_addr = 0;
    if constexpr (any_tensor_padded) {
        zero_cb_addr = get_write_ptr(zero_cb_id);
        constexpr uint32_t zero_init_trid = 1;  // drained here, before block 0 uses it again
        noc_async_read_set_trid(zero_init_trid);
        zero_fill_with_trid<skip_ptr_update, MEM_ZEROS_SIZE>(
            MEM_ZEROS_BASE, zero_cb_addr, zero_cb_size, vc, zero_init_trid);
        noc_async_read_barrier_with_trid(zero_init_trid);
    }

    uint32_t l1_buffer_start_addr = get_write_ptr(cb_id);
    uint32_t l1_buffer_end_addr = get_write_ptr(cb_id) + read_cb_size;

    volatile tt_l1_ptr uint32_t* tensor_addrs_l1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(addrs_cb_id));

    for (uint32_t layer = 0; layer < num_layers; layer++) {
        for (uint32_t t = 0; t < num_tensors; t++) {
            uint32_t curr_page_size = page_sizes[t];
            uint32_t curr_block_num_pages = block_num_pages[t];
            uint32_t curr_block_num_tiles = block_num_tiles[t];
            uint32_t curr_num_full_blocks = num_full_blocks[t];

            // Address setup
            uint32_t tensor_base_address = tensor_addrs_l1[layer * num_tensors + t];
            uint64_t src_base_addr = get_noc_addr_from_bank_id<true>(bank_id, tensor_base_address);
            noc_async_read_one_packet_set_state<true>(src_base_addr, curr_page_size, vc);

            uint32_t src_read_addr = 0;

            uint32_t num_free_blocks_in_buffer = total_num_blocks_in_buffer;
            uint32_t curr_block_trid = 1;
            uint32_t block_trid_to_wait = 1;

            cb_reserve_back(cb_id, max_block_num_tiles);

            uint32_t l1_write_addr_offset = 0;
            uint32_t l1_write_addr_start = get_write_ptr(cb_id);
            // Wrap around l1_write_addr if it reaches l1_buffer_end_addr
            if (l1_write_addr_start >= l1_buffer_end_addr) {
                l1_write_addr_start = l1_buffer_start_addr;
            }
            uint32_t l1_write_addr = l1_write_addr_start;

            for (uint32_t block = 0; block < num_blocks; block++) {
                // Set trid for current block
                noc_async_read_set_trid(curr_block_trid);

                // Issue noc async read commands for current block
                uint32_t temp_l1_write_addr = l1_write_addr;
                if (!any_tensor_padded || block < curr_num_full_blocks) {
                    for (uint32_t h = 0; h < curr_block_num_pages; ++h) {
                        noc_async_read_one_packet_with_state_with_trid<skip_ptr_update>(
                            src_base_addr, src_read_addr, temp_l1_write_addr, curr_block_trid);
                        src_read_addr += curr_page_size;
                        temp_l1_write_addr += curr_page_size;
                    }
                } else {
                    // Block reaches past the end of the DRAM shard (see the padding contract above). Only the first
                    // such block can hold real rows. Re-setting the read state while earlier reads are in flight is
                    // safe: set_state waits for the cmd buf to be free, and issued requests do not depend on it. The
                    // next tensor sets the DRAM read state again.
                    if (block == curr_num_full_blocks) {
                        const uint32_t curr_partial_block_num_pages = partial_block_num_pages[t];
                        if (curr_partial_block_num_pages != 0) {
                            const uint32_t curr_partial_block_page_size = partial_block_page_sizes[t];
                            noc_async_read_one_packet_set_state<true>(src_base_addr, curr_partial_block_page_size, vc);
                            for (uint32_t h = 0; h < curr_partial_block_num_pages; ++h) {
                                noc_async_read_one_packet_with_state_with_trid<skip_ptr_update>(
                                    src_base_addr, src_read_addr, temp_l1_write_addr, curr_block_trid);
                                src_read_addr += curr_partial_block_page_size;
                                temp_l1_write_addr += curr_partial_block_page_size;
                            }
                        }
                    }

                    // Zero-fill the pad rows [temp_l1_write_addr, end of block) from the zero CB, under the block's
                    // trid so the barrier below covers them.
                    const uint32_t block_end_addr = l1_write_addr + curr_page_size * curr_block_num_pages;
                    zero_fill_with_trid<skip_ptr_update, zero_cb_size>(
                        zero_cb_addr, temp_l1_write_addr, block_end_addr - temp_l1_write_addr, vc, curr_block_trid);
                }

                if (num_free_blocks_in_buffer == 3) {  // After first block we don't block but continue issuing reads
                    num_free_blocks_in_buffer -= 1;
                } else {
                    noc_async_read_barrier_with_trid(block_trid_to_wait);
                    cb_push_back(cb_id, max_block_num_tiles);
                    block_trid_to_wait =
                        block_trid_to_wait == total_num_blocks_in_buffer ? 1 : (block_trid_to_wait + 1);
                }

                // We still have blocks to read
                if (block != num_blocks - 1) {
                    // Increment block_trid, wrap around to 1 if it reaches total_num_blocks_in_buffer
                    curr_block_trid = curr_block_trid == total_num_blocks_in_buffer ? 1 : (curr_block_trid + 1);

                    // Wrap around l1_write_addr if it reaches l1_buffer_end_addr
                    l1_write_addr += max_block_size;
                    if (l1_write_addr >= l1_buffer_end_addr) {
                        l1_write_addr = l1_buffer_start_addr;
                    }

                    cb_reserve_back(
                        cb_id,
                        max_block_num_tiles *
                            2);  // Reserve two blocks of space to issue multiple block reads in parallel
                }
            }

            // last block to wait
            noc_async_read_barrier_with_trid(block_trid_to_wait);
            cb_push_back(cb_id, max_block_num_tiles);
        }
    }

    // wait for signal to exit, since reader cannot exit early due to the ongoing traffic on the same noc.
    cb_wait_front(sync_cb_id, 1);
    cb_pop_front(sync_cb_id, 1);

    // reset noc counters here because we didn't properly update ptrs for better perf.
    if (noc_mode == DM_DEDICATED_NOC) {
        ncrisc_noc_counters_init();
    } else {
        dynamic_noc_local_state_init();
    }
}
