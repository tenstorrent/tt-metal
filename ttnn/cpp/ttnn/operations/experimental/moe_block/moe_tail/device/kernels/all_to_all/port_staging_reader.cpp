// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// port_staging_reader: an all-to-all port's reader - its runs of every destination from staging into the staging CB
// Runs on: all-to-all port cores (plan.port_union; idle on a chip where the core is no port), RISCV_1 (NCRISC), NoC 0
// Responsibilities:
//  - each row block: zeroes the block's z-ready words (CB_Z_READY, L1, a row tile's 16 B apart - the first CB on the
//    workers and the ports, so one address on all of them), then releases every worker (+1 on its ZERO_GATE_SEM)
//    before any of them counts a row tile in; the staging buffer is reused, so this happens only once the previous
//    block's reads are done
//  - waits for a row tile's z-ready word to reach the worker count (every worker wrote it), then reads the row tile's
//    runs from the staging buffer (DRAM) over NoC 0
//  - pushes whole groups (pages_per_run x runs_per_group pages, a partial last group padded) into the staging CB for
//    port_fabric_sender, so a group's pages stay contiguous block after block
// Note: NoC 0 because worker_output_writer writes the staging buffer on NoC 0: reading a DRAM page through the other
//       NoC endpoint can return stale data after the write's ack (BH DRAM cross-port stale read)

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_tail/device/kernels/all_to_all/port_common.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_tail/device/kernels/all_to_all/padded_row_tiles.hpp"

void kernel_main() {
    moe_block_args::load();  // (my args: this chip's block of the DRAM table, common/args_table.hpp)
    constexpr auto staging_args = TensorAccessorArgs<PORT_ACCESSOR_ARGS>();
    constexpr uint32_t row_blocks = get_compile_time_arg_val(staging_args.next_compile_time_args_offset());
    size_t arg_index = 0;
    const uint32_t staging_address = moe_block_args::arg(arg_index++), num_workers = moe_block_args::arg(arg_index++);
    const uint32_t z_ready_address = get_write_ptr(CB_Z_READY), owned_row_tiles = moe_block_args::arg(arg_index++);
    const uint32_t hidden_tiles = moe_block_args::arg(arg_index++), block_tiles = moe_block_args::arg(arg_index++);
    const uint32_t link = moe_block_args::arg(arg_index++), num_links = moe_block_args::arg(arg_index++);
    const uint32_t num_destinations = moe_block_args::arg(arg_index++);
    const uint32_t padding_args = arg_index;  // (the padding args, at a fixed place)
    if constexpr (skip_padding) {
        arg_index += 7;
    }
    // (my destinations: position | part << 8, then my workers' x | y << 16, num_workers of them)
    const uint32_t destinations = arg_index, workers = destinations + num_destinations;
    const auto staging = TensorAccessor(staging_args, staging_address, page_bytes);
    volatile tt_l1_ptr uint32_t* z_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(z_ready_address);
    if (num_destinations == 0) {
        return;  // (not a carrying port: no worker counts in here)
    }
    constexpr uint32_t words_per_row_tile = 16 / sizeof(uint32_t);  // (a row tile's z-ready word: 16 B apart)
    for (uint32_t block = 0; block < row_blocks; ++block) {
        for (uint32_t row_tile = 0; row_tile < owned_row_tiles; ++row_tile) {
            z_ready[words_per_row_tile * row_tile] = 0;
        }
        for (uint32_t worker = 0; worker < num_workers; ++worker) {
            const uint32_t worker_xy = moe_block_args::arg(workers + worker);
            noc_semaphore_inc(get_noc_addr(worker_xy & 0xFFFF, worker_xy >> 16, get_semaphore(ZERO_GATE_SEM)), 1);
        }
        uint64_t live_mask = 0;
        if constexpr (skip_padding) {
            live_mask =
                read_live_row_tile_mask(padding_args, get_write_ptr(CB_PADDING_SCRATCH), owned_row_tiles, block);
        }
        uint32_t runs_in_group = 0, group_l1 = 0;
        for (uint32_t row_tile = 0; row_tile < owned_row_tiles; ++row_tile) {
            if (!row_tile_is_live<skip_padding>(live_mask, row_tile)) {
                continue;  // (its workers neither wrote nor counted it)
            }
            noc_semaphore_wait_min(
                z_ready + words_per_row_tile * row_tile, num_workers);  // (every worker wrote this row tile)
            for (uint32_t destination_index = 0; destination_index < num_destinations; ++destination_index) {
                const uint32_t entry = moe_block_args::arg(destinations + destination_index), position = entry & 0xFF,
                               part = (entry >> 8) & 3;
                const uint32_t row_tile_first_page = row_tile * hidden_tiles + position * block_tiles;
                for_each_run(block_tiles, link, num_links, part, [&](uint32_t run_first_page, uint32_t pages) {
                    if (runs_in_group == 0) {
                        cb_reserve_back(staging_cb, pages_per_run * runs_per_group);
                        group_l1 = get_write_ptr(staging_cb);
                    }
                    for (uint32_t i = 0; i < pages;
                         ++i) {  // (staging's pages run_first_page, + banks, ..: one bank, consecutive)
                        noc_async_read(
                            staging.get_noc_addr(row_tile_first_page + run_first_page + i * num_dram_banks),
                            group_l1 + (runs_in_group * pages_per_run + i) * page_bytes,
                            page_bytes);
                    }
                    if (++runs_in_group == runs_per_group) {
                        noc_async_read_barrier();
                        cb_push_back(staging_cb, pages_per_run * runs_per_group);
                        runs_in_group = 0;
                    }
                });
            }
        }
        if (runs_in_group) {  // (a whole group: the sender pops a whole group too)
            noc_async_read_barrier();
            cb_push_back(staging_cb, pages_per_run * runs_per_group);
        }
        noc_async_atomic_barrier();
    }
}
