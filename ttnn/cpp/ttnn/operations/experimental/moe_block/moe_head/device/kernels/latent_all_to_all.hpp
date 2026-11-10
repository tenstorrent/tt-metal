// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// K3's latent all-to-all on a fabric port core: its args (LatentAllToAllArgs), the stager's half
// (latent_all_to_all_gather: the matmul cores' latent rows into CB 0) and the sender's half (latent_all_to_all_send:
// CB 0 into the destination chips' LATZ over the fabric, multi-hop).
#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
// (no public device-side fabric API: the packet header's command types)
#include "hostdev/fabric_edm_packet_header.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/x_gather.hpp"

// A unit: one destination chip of my direction
struct LatentAllToAllUnit {
    uint32_t word;     // its block | my slot in its LATZ << 8 | its fabric chip id << 16
    uint32_t mesh_id;  // its mesh id
    uint32_t block() const { return word & 0xFF; }
    uint32_t zone_slot() const { return (word >> 8) & 0xFF; }
    uint16_t chip_id() const { return static_cast<uint16_t>(word >> 16); }
};
// A matmul core of my link
struct LatentAllToAllCore {
    uint32_t word;       // NoC x | y << 8 | matmul core index << 16
    uint32_t rows_word;  // its rows of each block: 4 bits a block (first row | rows << 2)
    uint32_t noc_x() const { return word & 0xFF; }
    uint32_t noc_y() const { return (word >> 8) & 0xFF; }
    uint32_t index() const { return word >> 16; }
    uint32_t first_row(uint32_t block) const { return (rows_word >> (4 * block)) & 3; }
    uint32_t num_rows(uint32_t block) const { return ((rows_word >> (4 * block)) & 0xF) >> 2; }
};

// A fabric port's latent all-to-all block, the same in the stager's and the sender's args (head_gather_program.cpp
// ChipProgram::latent_block): num_units, num_units x (unit word, mesh id), num_cores, num_cores x (core word, rows
// word), the 8 words from latent_address on, then the chips told "started". Read once in kernel_main
// (read_latent_all_to_all_args).
struct LatentAllToAllArgs {
    uint32_t num_units;
    volatile tt_l1_ptr uint32_t* units;  // in place: num_units x (word, mesh id)
    uint32_t num_cores;
    volatile tt_l1_ptr uint32_t* cores;    // in place: num_cores x (word, rows word)
    uint32_t latent_address;               // "lat": a core's latent block [row_tiles_per_core][max_column_tiles]
    uint32_t handoff_words_address;        // my hand-off words: X_GATHER_HANDOFF_BYTES apart, by matmul core index
    uint32_t latent_zone_address;          // "latz": the destination's LATZ (a slot a source chip)
    uint32_t arrival_counter_address;      // a destination core's latent arrival counter
    uint32_t max_column_tiles;             // a latent row's tiles (LWMAX)
    uint32_t fence_address;                // my end-to-end fence word: the destinations' "started"
    uint32_t fence_count;                  // the "started"s my sends need (one a destination chip)
    FabricStartedTargets started_targets;  // the chips I tell "started" (their latent ports toward me)

    LatentAllToAllUnit unit(uint32_t i) const { return {units[2 * i], units[2 * i + 1]}; }
    LatentAllToAllCore core(uint32_t i) const { return {cores[2 * i], cores[2 * i + 1]}; }
};
// (kernel_main) the block at arg `arg_index`, which it leaves past the block
FORCE_INLINE LatentAllToAllArgs read_latent_all_to_all_args(uint32_t& arg_index) {
    LatentAllToAllArgs args;
    args.num_units = moe_block_args::arg(arg_index);
    args.units = moe_block_args::arg_pointer(arg_index + 1);
    arg_index += 1 + 2 * args.num_units;
    args.num_cores = moe_block_args::arg(arg_index);
    args.cores = moe_block_args::arg_pointer(arg_index + 1);
    arg_index += 1 + 2 * args.num_cores;
    args.latent_address = moe_block_args::arg(arg_index++);
    args.handoff_words_address = moe_block_args::arg(arg_index++);
    args.latent_zone_address = moe_block_args::arg(arg_index++);
    args.arrival_counter_address = moe_block_args::arg(arg_index++);
    args.max_column_tiles = moe_block_args::arg(arg_index++);
    args.fence_address = moe_block_args::arg(arg_index++);
    args.fence_count = moe_block_args::arg(arg_index++);
    args.started_targets.count = moe_block_args::arg(arg_index++);
    args.started_targets.words = moe_block_args::arg_pointer(arg_index);
    arg_index += 2 * args.started_targets.count;
    return args;
}

// (the stager) per destination block of my direction, per core of my link holding rows of it: wait its hand-off on my
// latent word of it, NoC-read its rows [first row, first row + rows) of its latent block in tiles_per_packet-tile
// packets into CB 0 (the batch rules of reduce_scatter_gather). Chunk m: its LAT at + m LATENT_CHUNK_BYTES bytes; the
// words' targets count up (handoffs_seen, the caller's), re-armed after the last live chunk.
FORCE_INLINE void latent_all_to_all_gather(
    const LatentAllToAllArgs& args, uint32_t& packets_pushed, uint32_t chunk, uint32_t* handoffs_seen) {
    if (args.num_units == 0) {
        return;
    }
    const uint32_t latent_address = args.latent_address + chunk * LATENT_CHUNK_BYTES;
    const uint32_t tiles_per_row = args.max_column_tiles;
    if (chunk == 0) {
        for (uint32_t i = 0; i < args.num_cores; ++i) {
            handoffs_seen[i] = 0;
        }
    }
    uint32_t batch_packets = 0, batch_capacity = 0, write_address = 0;
    auto flush = [&]() {
        noc_async_read_barrier();
        cb_push_back(0, batch_packets * tiles_per_packet);
        packets_pushed += batch_packets;
        batch_packets = 0;
    };
    for (uint32_t unit_index = 0; unit_index < args.num_units; ++unit_index) {
        const uint32_t block = args.unit(unit_index).block();
        for (uint32_t core_index = 0; core_index < args.num_cores; ++core_index) {
            const LatentAllToAllCore core = args.core(core_index);
            const uint32_t first_row = core.first_row(block), tiles = core.num_rows(block) * tiles_per_row;
            if (tiles == 0) {
                continue;
            }
            volatile tt_l1_ptr uint32_t* handoff = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                args.handoff_words_address + X_GATHER_HANDOFF_BYTES * core.index());
            const uint32_t target = ++handoffs_seen[core_index];
            invalidate_l1_cache();
            if (*handoff < target && batch_packets > 0) {
                flush();
            }
            noc_semaphore_wait_min(handoff, target);
            const uint64_t source_noc_address = get_noc_addr(
                core.noc_x(),
                core.noc_y(),
                latent_address + first_row * tiles_per_row * moe_block_tiles::kBf16TileBytes);
            for (uint32_t first_tile = 0; first_tile < tiles; first_tile += tiles_per_packet) {
                if (batch_packets == 0) {
                    batch_capacity = packets_per_group - packets_pushed % packets_per_group;
                    cb_reserve_back(0, batch_capacity * tiles_per_packet);
                    write_address = get_write_ptr(0);
                }
                const uint32_t packet_tiles =
                    tiles - first_tile < tiles_per_packet ? tiles - first_tile : tiles_per_packet;
                noc_async_read(
                    source_noc_address + first_tile * moe_block_tiles::kBf16TileBytes,
                    write_address + batch_packets * tiles_per_packet * moe_block_tiles::kBf16TileBytes,
                    packet_tiles * moe_block_tiles::kBf16TileBytes);
                if (++batch_packets == batch_capacity) {
                    flush();
                }
            }
        }
        if (batch_packets > 0) {
            flush();
        }
    }
    if (chunk + 1 < x_gather_num_live_chunks) {
        return;
    }
    for (uint32_t i = 0; i < args.num_cores; ++i) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            args.handoff_words_address + X_GATHER_HANDOFF_BYTES * args.core(i).index()) = 0;
    }
}

// (the sender) the stager's walk: packets into destination chip b's same core (multi-hop fabric unicast: the unit's
// header, routed to b once - packet_headers[packets_per_group + 1 + unit]), the LATZ slot of my chip, its rows; a
// core's last packet with the fused increment of its arrival counter. First the end-to-end fence: every destination
// chip started this call (its LATZ is free; my neighbour's "ready" only fences the first hop).
template <typename Connection, typename Header, typename Poll>
FORCE_INLINE void latent_all_to_all_send(
    const LatentAllToAllArgs& args,
    Connection& connection,
    Header* packet_headers,
    Poll&& poll_interleaved_sends,
    uint32_t& packets_popped,
    uint32_t chunk) {
    if (args.num_units == 0) {
        return;
    }
    const uint32_t latent_zone_address = args.latent_zone_address + chunk * LATENT_CHUNK_BYTES;
    const uint32_t tiles_per_row = args.max_column_tiles;
    while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.fence_address) < args.fence_count) {
        invalidate_l1_cache();
        poll_interleaved_sends();
    }
    uint32_t batch_packets = 0, batch_capacity = packets_per_group - packets_popped % packets_per_group;
    auto pop_batch = [&]() {
        noc_async_writes_flushed();
        cb_pop_front(0, batch_packets * tiles_per_packet);
        packets_popped += batch_packets;
        batch_packets = 0;
        batch_capacity = packets_per_group - packets_popped % packets_per_group;
    };
    for (uint32_t unit_index = 0; unit_index < args.num_units; ++unit_index) {
        const LatentAllToAllUnit unit = args.unit(unit_index);
        const uint32_t block = unit.block(), zone_slot = unit.zone_slot();
        Header header = packet_headers[packets_per_group + 1 + unit_index];
        for (uint32_t core_index = 0; core_index < args.num_cores; ++core_index) {
            const LatentAllToAllCore core = args.core(core_index);
            const uint32_t first_row = core.first_row(block), tiles = core.num_rows(block) * tiles_per_row;
            if (tiles == 0) {
                continue;
            }
            const uint32_t noc_x = core.noc_x(), noc_y = core.noc_y();
            const uint32_t destination_address = latent_zone_address + (zone_slot * row_tiles_per_core + first_row) *
                                                                           tiles_per_row *
                                                                           moe_block_tiles::kBf16TileBytes;
            for (uint32_t first_tile = 0; first_tile < tiles; first_tile += tiles_per_packet) {
                poll_interleaved_sends();
                while (!cb_pages_available_at_front(0, tiles_per_packet * (batch_packets + 1))) {
                    poll_interleaved_sends();
                }
                cb_wait_front(0, tiles_per_packet * (batch_packets + 1));
                const uint32_t source_address =
                    get_read_ptr(0) + batch_packets * tiles_per_packet * moe_block_tiles::kBf16TileBytes;
                const uint32_t packet_tiles =
                    tiles - first_tile < tiles_per_packet ? tiles - first_tile : tiles_per_packet;
                const uint64_t destination_noc_address =
                    get_noc_addr(noc_x, noc_y, destination_address + first_tile * moe_block_tiles::kBf16TileBytes);
                noc_async_writes_flushed();  // (this unit's one header: its previous packet has left)
                if (first_tile + tiles_per_packet >= tiles) {
                    header->to_noc_fused_unicast_write_atomic_inc(
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            destination_noc_address, get_noc_addr(noc_x, noc_y, args.arrival_counter_address), 1, true},
                        packet_tiles * moe_block_tiles::kBf16TileBytes);
                } else {
                    header->to_noc_unicast_write(
                        tt::tt_fabric::NocUnicastCommandHeader{destination_noc_address},
                        packet_tiles * moe_block_tiles::kBf16TileBytes);
                }
                connection.wait_for_empty_write_slot();
                connection.send_current_slot_non_blocking(
                    source_address, packet_tiles * moe_block_tiles::kBf16TileBytes, reinterpret_cast<uint32_t>(header));
                if (++batch_packets == batch_capacity) {
                    pop_batch();
                }
            }
        }
        if (batch_packets > 0) {
            pop_batch();
        }
    }
}
