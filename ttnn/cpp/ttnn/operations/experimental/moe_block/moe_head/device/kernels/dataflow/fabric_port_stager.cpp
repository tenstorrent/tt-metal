// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fabric_port_stager: stages what a fabric port core sends into CB 0, for fabric_port_sender on the same core
// Runs on: port cores (one per X direction and fabric link), RISCV_0 (BRISC), NoC 1 (RISCV_1_default)
// Responsibilities:
//  - x all-gather: NoC-reads this chip's own x shard from the matmul cores' x-own L1 (once every matmul core counted
//    its load) and, relaying, the upstream shards the fabric landed there (arrival counter) into CB 0
//  - reduce-scatter: per matmul core's hand-off (mailbox word) reads its sub-block from its FINB into CB 0, then +1 on
//    its FINB-free word; latent all-to-all: the cores' latent rows from LAT the same way
//  - skips padding row tiles; pushes whole groups (packets_per_group x tiles_per_packet tiles) so CB 0 stays in step
//    with the sender
// Note: in this op the reading side runs on BRISC (RISCV_0) and the sending side on NCRISC (RISCV_1), opposite to
//  tt-metal's Reader/Writer config convention, on purpose; the fabric-landed shards are read on NoC 1, the fabric
//  routers' NoC
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/row_block.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/x_gather.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/latent_all_to_all.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/reduce_scatter.hpp"

constexpr auto x_accessor_args = TensorAccessorArgs<x_gather_num_compile_time_args>();

// An entry of my x gather: a shard this port sends on (entry 0: my chip's own; then the relayed ones), in place - 4
// words each (head_gather_program.cpp ChipProgram::gather, `reader_entries`)
struct StagerEntries {
    volatile tt_l1_ptr uint32_t* words;
    uint32_t shard(uint32_t entry) const { return words[4 * entry]; }
    uint32_t part(uint32_t entry) const { return words[4 * entry + 1]; }  // 0: whole; 1 / 2: a ring half
    // (relayed) the upstream's arrival increments before this entry's within a chunk
    uint32_t upstream_base(uint32_t entry) const { return words[4 * entry + 2]; }
    uint32_t packets_per_chunk(uint32_t entry) const { return words[4 * entry + 3]; }
};

// This kernel's runtime args (head_gather_program.cpp ChipProgram::gather, `reader_args`), read once in kernel_main
struct FabricPortStagerArgs {
    uint32_t x_address;                           // [0] x in DRAM (source_arg(SOURCE_X))
    uint32_t x_own_address;                       // [1] the matmul cores' x-own L1
    uint32_t gather_link;                         // [2] my fabric link
    uint32_t arrival_counter_address;             // [3] my arrival counter (the upstream port's fabric increments)
    uint32_t num_entries;                         // [4]
    uint32_t upstream_increments_per_chunk;       // [5] the upstream's arrival increments a chunk (pl.up_ipc)
    StagerEntries entries;                        // [6 ..] num_entries x 4 words
    XGatherDestinationTables destination_tables;  // then: my chip's (HeadModel::tables)
    uint32_t x_loaded_counter_address;            // then: the matmul cores' "own shard loaded" counter here
    uint32_t x_loaded_count;                      // its full count (every matmul core)
    ReduceScatterArgs reduce_scatter;             // then: reduce_scatter_block
    LatentAllToAllArgs latent_all_to_all;         // then: latent_block
#ifdef SKIP_PADDING_CHUNKS
    XGatherLiveChunkArgs live_chunks;  // (common args)
#endif
};

FORCE_INLINE FabricPortStagerArgs read_fabric_port_stager_args() {
    FabricPortStagerArgs args;
    uint32_t arg_index = 0;
    args.x_address = moe_block_args::arg(arg_index++);
    args.x_own_address = moe_block_args::arg(arg_index++);
    args.gather_link = moe_block_args::arg(arg_index++);
    args.arrival_counter_address = moe_block_args::arg(arg_index++);
    args.num_entries = moe_block_args::arg(arg_index++);
    args.upstream_increments_per_chunk = moe_block_args::arg(arg_index++);
    args.entries.words = moe_block_args::arg_pointer(arg_index);
    arg_index += 4 * args.num_entries;
    args.destination_tables.words = moe_block_args::arg_pointer(arg_index);
    arg_index += XGatherDestinationTables::num_words;
    args.x_loaded_counter_address = moe_block_args::arg(arg_index++);
    args.x_loaded_count = moe_block_args::arg(arg_index++);
    args.reduce_scatter = read_reduce_scatter_args(arg_index);
    args.latent_all_to_all = read_latent_all_to_all_args(arg_index);
#ifdef SKIP_PADDING_CHUNKS
    args.live_chunks = read_x_gather_live_chunk_args();
#endif
    return args;
}

// One row block of this kernel's work: kernel_main below runs it once per block (row_block.hpp).
void process_row_block(uint32_t block, const FabricPortStagerArgs& args) {
#ifdef SKIP_PADDING_CHUNKS
    x_gather_find_live_chunks(args.live_chunks);  // (a padded forward: the row chunks up to the last with a real row)
#endif
    volatile tt_l1_ptr uint32_t* x_loaded =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.x_loaded_counter_address);
    bool x_loaded_seen = false;
    const auto x = TensorAccessor(x_accessor_args, args.x_address, moe_block_tiles::kBf16TileBytes);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.arrival_counter_address);
    uint32_t batch_packets = 0, write_address = 0;
    for (uint32_t chunk = 0; chunk < x_gather_num_live_chunks; ++chunk) {
        for (uint32_t entry = 0; entry < args.num_entries; ++entry) {
            const uint32_t upstream_base = args.entries.upstream_base(entry),
                           packets_per_chunk = args.entries.packets_per_chunk(entry);
            uint32_t packets_staged = 0;
            const auto real_row_tiles = x_gather_mask(chunk);  // (chunk's real row tiles)
            x_gather_walk(
                chunk,
                args.entries.shard(entry),
                args.entries.part(entry),
                args.gather_link,
                [&](uint32_t row_tile,
                    uint32_t k_step,
                    uint32_t,
                    uint32_t step_tile,
                    uint32_t packet_tiles,
                    uint32_t packet_index) {
                    if (batch_packets == 0) {
                        cb_reserve_back(0, tiles_per_packet * packets_per_group);
                        write_address = get_write_ptr(0);
                    }
                    const uint32_t cb_address =
                        write_address + batch_packets * tiles_per_packet * moe_block_tiles::kBf16TileBytes;
                    if (entry == 0) {  // my own shard: from the engine's x-own L1 once every engine core loaded it (no
                                       // DRAM here: the DRAM is the engine's weight load's)
                        if (!x_loaded_seen) {
                            noc_semaphore_wait_min(x_loaded, args.x_loaded_count);
                            x_loaded_seen = true;
                        }
                        if ((real_row_tiles >> (row_tile - chunk * chunk_row_tiles)) & 1) {  // (padding: no read)
                            const XGatherDestination source = args.destination_tables.destination(
                                args.x_own_address, chunk, row_tile, k_step, step_tile);
                            noc_async_read(
                                get_noc_addr(source.noc_x, source.noc_y, source.address),
                                cb_address,
                                packet_tiles * moe_block_tiles::kBf16TileBytes);
                        }
                    } else {  // relay: the upstream's packet packet_index of (this chunk, its entry - 1) landed here
                        noc_semaphore_wait_min(
                            arrived,
                            chunk * args.upstream_increments_per_chunk + upstream_base +
                                packet_index / packets_per_arrival_increment + 1);
                        if ((real_row_tiles >> (row_tile - chunk * chunk_row_tiles)) & 1) {
                            const XGatherDestination source = args.destination_tables.destination(
                                args.x_own_address, chunk, row_tile, k_step, step_tile);
                            noc_async_read(
                                get_noc_addr(source.noc_x, source.noc_y, source.address),
                                cb_address,
                                packet_tiles * moe_block_tiles::kBf16TileBytes);
                        }
                    }
                    ++batch_packets;
                    ++packets_staged;
                    if (batch_packets == packets_per_group || packets_staged == packets_per_chunk) {
                        noc_async_read_barrier();
                        cb_push_back(0, tiles_per_packet * packets_per_group);
                        batch_packets = 0;
                    }
                });
        }
    }
    noc_semaphore_wait_min(x_loaded, args.x_loaded_count);  // (re-arm: every engine core counted itself on every port)
    noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(x_loaded)), 0u - args.x_loaded_count);
    noc_async_atomic_barrier();
    uint32_t packets_pushed = 0;  // (the gather left CB 0 batch-aligned; the phases after it share one count)
    uint32_t reduce_scatter_handoffs_seen[128], latent_handoffs_seen[128];
    for (uint32_t chunk = 0; chunk < x_gather_num_live_chunks; ++chunk) {
        latent_all_to_all_gather(args.latent_all_to_all, packets_pushed, chunk, latent_handoffs_seen);
        reduce_scatter_gather(args.reduce_scatter, packets_pushed, chunk, reduce_scatter_handoffs_seen);
    }
    if constexpr (row_block::BLOCKS > 1) {
        if (packets_pushed % packets_per_group) {  // (row blocks: the last group padded - the next block's gather
                                                   // writes whole groups from here)
            cb_reserve_back(0, (packets_per_group - packets_pushed % packets_per_group) * tiles_per_packet);
            cb_push_back(0, (packets_per_group - packets_pushed % packets_per_group) * tiles_per_packet);
        }
    }
}

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table; a compute kernel waits for its page)
    const FabricPortStagerArgs args = read_fabric_port_stager_args();
    for (uint32_t block = 0; block < row_block::BLOCKS; ++block) {
        row_block::begin(block);
        process_row_block(block, args);
        row_block::end(block);
    }
}
