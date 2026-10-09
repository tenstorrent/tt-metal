// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_dram.cpp, which lives beside it and serves the GlobalCircularBuffer path. This
// fork serves the PrefetcherPipe path (DramPrefetcherPipeSpecFactory); the read algorithm is the same, so a
// change to either likely belongs in the other too.
//
// Reads this core's DRAM bank of every weight tensor, layer by layer, one block at a time into the staging
// buffer, keeping up to two blocks of reads in flight (one transaction id per staging slot).
//
// Bindings:
//   dfb::staging - PRODUCER: the staging ring, num_staging_blocks slots of max_block_num_tiles entries
//   dfb::addrs   - PRODUCER + CONSUMER: the address tensor's shard, [layer][tensor] weight base addresses
//   dfb::sync    - CONSUMER: the writer's one-entry exit signal
// Common runtime varargs: page_size[num_tensors], then block_num_pages[num_tensors].

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_layers = get_arg(args::num_layers);
    constexpr uint32_t num_tensors = get_arg(args::num_tensors);
    constexpr uint32_t num_blocks = get_arg(args::num_blocks);
    constexpr uint32_t staging_size_bytes = get_arg(args::staging_size_bytes);
    constexpr uint32_t num_staging_blocks = get_arg(args::num_staging_blocks);
    constexpr uint32_t max_block_num_tiles = get_arg(args::max_block_num_tiles);
    constexpr uint32_t max_block_size = get_arg(args::max_block_size);
    constexpr bool skip_ptr_update = get_arg(args::skip_ptr_update);

    const uint32_t bank_id = get_arg(args::bank_id);
    const uint32_t vc = get_arg(args::vc);

    DataflowBuffer staging(dfb::staging);
    DataflowBuffer addrs(dfb::addrs);
    DataflowBuffer sync(dfb::sync);

    const uint32_t l1_buffer_start_addr = staging.get_write_ptr();
    const uint32_t l1_buffer_end_addr = l1_buffer_start_addr + staging_size_bytes;

    volatile tt_l1_ptr uint32_t* tensor_addrs_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addrs.get_read_ptr());

    for (uint32_t layer = 0; layer < num_layers; layer++) {
        for (uint32_t t = 0; t < num_tensors; t++) {
            const uint32_t curr_page_size = get_common_vararg(t);
            const uint32_t curr_block_num_pages = get_common_vararg(num_tensors + t);

            // Address setup
            uint32_t tensor_base_address = tensor_addrs_l1[layer * num_tensors + t];
            uint64_t src_base_addr = get_noc_addr_from_bank_id<true>(bank_id, tensor_base_address);
            noc_async_read_one_packet_set_state<true>(src_base_addr, curr_page_size, vc);

            uint32_t src_read_addr = 0;

            uint32_t num_free_blocks_in_buffer = num_staging_blocks;
            uint32_t curr_block_trid = 1;
            uint32_t block_trid_to_wait = 1;

            staging.reserve_back(max_block_num_tiles);

            uint32_t l1_write_addr_start = staging.get_write_ptr();
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
                for (uint32_t h = 0; h < curr_block_num_pages; ++h) {
                    noc_async_read_one_packet_with_state_with_trid<skip_ptr_update>(
                        src_base_addr, src_read_addr, temp_l1_write_addr, curr_block_trid);
                    src_read_addr += curr_page_size;
                    temp_l1_write_addr += curr_page_size;
                }

                if (num_free_blocks_in_buffer == num_staging_blocks) {
                    // After the first block, keep issuing reads rather than wait for it.
                    num_free_blocks_in_buffer -= 1;
                } else {
                    noc_async_read_barrier_with_trid(block_trid_to_wait);
                    staging.push_back(max_block_num_tiles);
                    block_trid_to_wait = block_trid_to_wait == num_staging_blocks ? 1 : (block_trid_to_wait + 1);
                }

                // We still have blocks to read
                if (block != num_blocks - 1) {
                    // Increment block_trid, wrap around to 1 if it reaches num_staging_blocks
                    curr_block_trid = curr_block_trid == num_staging_blocks ? 1 : (curr_block_trid + 1);

                    // Wrap around l1_write_addr if it reaches l1_buffer_end_addr
                    l1_write_addr += max_block_size;
                    if (l1_write_addr >= l1_buffer_end_addr) {
                        l1_write_addr = l1_buffer_start_addr;
                    }

                    // Reserve two blocks of space to issue multiple block reads in parallel
                    staging.reserve_back(max_block_num_tiles * 2);
                }
            }

            // last block to wait
            noc_async_read_barrier_with_trid(block_trid_to_wait);
            staging.push_back(max_block_num_tiles);
        }
    }

    // Leave the read command buffer untagged: in dynamic-NOC mode it is also an atomic buffer, whose
    // packet tag the firmware requires to be zero before the next kernel.
    noc_async_read_set_trid(0);

    // In performance mode the reads skipped the NoC counters; re-sync them before exit. The writer shares
    // this NOC (dynamic-NOC mode shares the counters too), so wait until it has no traffic in flight.
    sync.wait_front(1);
    sync.pop_front(1);
    if (noc_mode == DM_DEDICATED_NOC) {
        ncrisc_noc_counters_init();
    } else {
        dynamic_noc_local_state_init();
    }
}
