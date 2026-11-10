// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The reduce-scatter (down + reduce-scatter) on a fabric port core: its args (ReduceScatterArgs), the stager's half
// (reduce_scatter_gather: the matmul cores' sub-blocks into CB 0) and the sender's half (reduce_scatter_send: CB 0 into
// the downstream chip's zones over the fabric).
#pragma once

#include <cstdint>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
// (no public device-side fabric API: the packet header's command types)
#include "hostdev/fabric_edm_packet_header.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/x_gather.hpp"

// A unit: one entry's block for my direction, from the cores of the halves in its mask
struct ReduceScatterUnit {
    // entry | block << 8 | half mask << 16 | zone prefix masks << 20 (4 bits a half) | pieces << 28
    uint32_t word;
    uint32_t slots_word;  // the zone slot of half 0 | of half 1 << 16
    uint32_t entry() const { return word & 0xFF; }
    uint32_t block() const { return (word >> 8) & 0xFF; }
    uint32_t half_mask() const { return (word >> 16) & 3; }
    // (compact zones) the blocks before this unit's in the zone of the half: a core's slot is its tiles of them
    uint32_t zone_prefix_mask(uint32_t half) const { return (word >> (20 + 4 * half)) & 0xF; }
    // the cores hand the unit over pass by pass (REDUCE_SCATTER_PASSES pieces a core), else each core whole
    bool in_pieces() const { return (word >> 28) & 1; }
    uint32_t zone_slot(uint32_t half) const { return half ? (slots_word >> 16) : (slots_word & 0xFFFF); }
};
// A matmul core of my link
struct ReduceScatterCore {
    uint32_t word;    // NoC x | y << 8 | half << 16 | matmul core index << 20 | its pass columns << 27
    uint32_t counts;  // its tiles of each block (tiles_of_block)
    uint32_t noc_x() const { return word & 0xFF; }
    uint32_t noc_y() const { return (word >> 8) & 0xFF; }
    uint32_t half() const { return (word >> 16) & 1; }
    uint32_t index() const { return (word >> 20) & 0x7F; }
    uint32_t pass_columns() const { return word >> 27; }
};

// A fabric port's reduce-scatter block, the same in the stager's and the sender's args (head_gather_program.cpp
// ChipProgram::reduce_scatter_block): num_units, num_units x (unit word, slots word), num_cores, num_cores x (core
// word, counts word), the chip's entry order [tp_size], then the 12 words from final_block_address on. Read once in
// kernel_main (read_reduce_scatter_args).
struct ReduceScatterArgs {
    uint32_t num_units;
    volatile tt_l1_ptr uint32_t* units;  // in place: num_units x (word, slots word)
    uint32_t num_cores;
    volatile tt_l1_ptr uint32_t* cores;        // in place: num_cores x (word, counts)
    volatile tt_l1_ptr uint32_t* entry_order;  // in place: the chip's entries [tp_size]
    uint32_t final_block_address;              // "finb": a matmul core's final sub-blocks (compact, in entry order)
    uint32_t mailbox_address;                  // my hand-off words: X_GATHER_HANDOFF_BYTES apart, by matmul core index
    uint32_t zone_address;                     // the downstream chip's zones of this direction ("arrf" / "arrb")
    uint32_t arrival_counter_address;          // a downstream core's arrival counter of this direction
    uint32_t zone_slot_tiles;                  // a zone slot's tiles; 0: compact zones
    uint32_t final_free_address;               // a matmul core's FINB-free word
    uint32_t zone_credit_address;              // my zone-credit word (the peer port's credits: chunk m may go)
    uint32_t zone_free_address;                // my zone-free word (my chip's cores read a chunk's zones)
    uint32_t zone_free_signals_per_chunk;      // its signals a chunk
    uint32_t zone_credits_expected;            // the credits a call expects (every chunk live; 0: none)
    uint32_t router_done_address;              // my "router done" word (0: no router buffers over the zones)
    // the matmul cores counting on it | it gates chunk 0's zone credit << 16 | it gates the latent's "started" << 17
    uint32_t router_done_word;

    ReduceScatterUnit unit(uint32_t i) const { return {units[2 * i], units[2 * i + 1]}; }
    ReduceScatterCore core(uint32_t i) const { return {cores[2 * i], cores[2 * i + 1]}; }
    uint32_t router_done_count() const { return router_done_word & 0xFFFF; }
    bool router_done_gates_zone() const { return (router_done_word >> 16) & 1; }
    bool router_done_gates_latent() const { return (router_done_word >> 17) & 1; }
};
// (kernel_main) the block at arg `arg_index`, which it leaves past the block
FORCE_INLINE ReduceScatterArgs read_reduce_scatter_args(uint32_t& arg_index) {
    ReduceScatterArgs args;
    args.num_units = moe_block_args::arg(arg_index);
    args.units = moe_block_args::arg_pointer(arg_index + 1);
    arg_index += 1 + 2 * args.num_units;
    args.num_cores = moe_block_args::arg(arg_index);
    args.cores = moe_block_args::arg_pointer(arg_index + 1);
    arg_index += 1 + 2 * args.num_cores;
    args.entry_order = moe_block_args::arg_pointer(arg_index);
    arg_index += tp_size;
    args.final_block_address = moe_block_args::arg(arg_index++);
    args.mailbox_address = moe_block_args::arg(arg_index++);
    args.zone_address = moe_block_args::arg(arg_index++);
    args.arrival_counter_address = moe_block_args::arg(arg_index++);
    args.zone_slot_tiles = moe_block_args::arg(arg_index++);
    args.final_free_address = moe_block_args::arg(arg_index++);
    args.zone_credit_address = moe_block_args::arg(arg_index++);
    args.zone_free_address = moe_block_args::arg(arg_index++);
    args.zone_free_signals_per_chunk = moe_block_args::arg(arg_index++);
    args.zone_credits_expected = moe_block_args::arg(arg_index++);
    args.router_done_address = moe_block_args::arg(arg_index++);
    args.router_done_word = moe_block_args::arg(arg_index++);
    return args;
}

// (pieces) pass `pass`'s piece of a core's sub-block of `tiles` tiles: the column's widths are
// split_even(pass_columns, REDUCE_SCATTER_PASSES); the piece's rows x its width, at rows x (the earlier passes' widths)
// in its pass-major sub-block
struct ReduceScatterPiece {
    uint32_t offset, tiles;
};
FORCE_INLINE ReduceScatterPiece reduce_scatter_piece(uint32_t pass, uint32_t pass_columns, uint32_t tiles) {
    const uint32_t rows = tiles / pass_columns, base_width = pass_columns / REDUCE_SCATTER_PASSES,
                   remainder = pass_columns % REDUCE_SCATTER_PASSES;
    return {
        rows * (pass * base_width + (pass < remainder ? pass : remainder)),
        rows * (base_width + (pass < remainder ? 1 : 0))};
}

// (the stager) gather my units of chunk `chunk`: per unit, per core of my link in the unit's halves holding tiles of
// its block: wait its hand-off on my mailbox word of it, NoC-read its bf16 sub-block from its FINB (compact in the
// chip's entry order) in packets of tiles_per_packet tiles into CB 0; a batch ends at a multiple of packets_per_group
// packets (packets_pushed: since the phases after the gather began - CB 0's batch alignment), at the end of a unit and
// before a blocking wait. The mailbox targets count up over the chunks (handoffs_seen, a word a core: the caller's).
// After chunk m (not the last live one) +1 on the FINB-free word of every core read - its FINB is free for chunk m + 1;
// after the last the mailbox words re-armed.
FORCE_INLINE void reduce_scatter_gather(
    const ReduceScatterArgs& args, uint32_t& packets_pushed, uint32_t chunk, uint32_t* handoffs_seen) {
    if (args.num_units == 0) {
        return;
    }
    if (chunk == 0) {
        for (uint32_t i = 0; i < args.num_cores; ++i) {
            handoffs_seen[i] = 0;
        }
    }
    uint8_t core_read[128];
    for (uint32_t i = 0; i < args.num_cores; ++i) {
        core_read[i] = 0;
    }
    uint32_t batch_packets = 0, batch_capacity = 0, write_address = 0;
    auto flush = [&]() {
        noc_async_read_barrier();
        cb_push_back(0, batch_packets * tiles_per_packet);
        packets_pushed += batch_packets;
        batch_packets = 0;
    };
    for (uint32_t unit_index = 0; unit_index < args.num_units; ++unit_index) {
        const ReduceScatterUnit unit = args.unit(unit_index);
        const uint32_t block = unit.block();
        // (pieces: the cores hand it pass by pass - walk pass-major over my cores; else core by core whole)
        const uint32_t passes = unit.in_pieces() ? REDUCE_SCATTER_PASSES : 1;
        for (uint32_t pass = 0; pass < passes; ++pass) {
            for (uint32_t core_index = 0; core_index < args.num_cores; ++core_index) {
                const ReduceScatterCore core = args.core(core_index);
                uint32_t tiles = tiles_of_block(core.counts, block);
                if (!((unit.half_mask() >> core.half()) & 1) || tiles == 0) {
                    continue;
                }
                uint32_t final_block_offset = 0;  // (the sub-block's FINB offset: the core's tiles of earlier entries)
                for (uint32_t entry = 0; entry < unit.entry(); ++entry) {
                    final_block_offset += tiles_of_block(core.counts, args.entry_order[entry]);
                }
                if (passes > 1) {
                    const ReduceScatterPiece piece = reduce_scatter_piece(pass, core.pass_columns(), tiles);
                    final_block_offset += piece.offset;
                    tiles = piece.tiles;
                }
                volatile tt_l1_ptr uint32_t* mailbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    args.mailbox_address + X_GATHER_HANDOFF_BYTES * core.index());
                const uint32_t target = ++handoffs_seen[core_index];
                invalidate_l1_cache();
                if (*mailbox < target && batch_packets > 0) {
                    flush();  // (never hold reads while blocked on a producer)
                }
                noc_semaphore_wait_min(mailbox, target);
                core_read[core_index] = 1;
                const uint64_t source_noc_address = get_noc_addr(
                    core.noc_x(),
                    core.noc_y(),
                    args.final_block_address + final_block_offset * moe_block_tiles::kBf16TileBytes);
                const bool live = x_gather_block_real(chunk, core.index(), block);  // (padding rows only: not read)
                for (uint32_t first_tile = 0; first_tile < tiles; first_tile += tiles_per_packet) {
                    if (batch_packets == 0) {
                        batch_capacity = packets_per_group - packets_pushed % packets_per_group;
                        cb_reserve_back(0, batch_capacity * tiles_per_packet);
                        write_address = get_write_ptr(0);
                    }
                    const uint32_t packet_tiles =
                        tiles - first_tile < tiles_per_packet ? tiles - first_tile : tiles_per_packet;
                    if (live) {
                        noc_async_read(
                            source_noc_address + first_tile * moe_block_tiles::kBf16TileBytes,
                            write_address + batch_packets * tiles_per_packet * moe_block_tiles::kBf16TileBytes,
                            packet_tiles * moe_block_tiles::kBf16TileBytes);
                    }
                    if (++batch_packets == batch_capacity) {
                        flush();
                    }
                }
            }
        }
        if (batch_packets > 0) {
            flush();
        }
    }
    if (chunk + 1 < x_gather_num_live_chunks) {  // (every read of the chunk landed - each unit ends in a flush)
        for (uint32_t i = 0; i < args.num_cores; ++i) {
            if (core_read[i]) {
                const ReduceScatterCore core = args.core(i);
                noc_semaphore_inc(get_noc_addr(core.noc_x(), core.noc_y(), args.final_free_address), 1);
            }
        }
        noc_async_atomic_barrier();
        return;
    }
    for (uint32_t i = 0; i < args.num_cores; ++i) {  // (re-arm: every hand-off of the call was waited for)
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            args.mailbox_address + X_GATHER_HANDOFF_BYTES * args.core(i).index()) = 0;
    }
}

// (the sender) my units' packets of chunk `chunk`, in the stager's walk: one fabric packet each into the downstream
// chip's same core, its zone slot of this direction (by the core's half; compact zones: at the sum of its tiles of the
// blocks in the unit's prefix mask for its half); a core's last packet carries the fused increment of its arrival
// counter. Pops CB 0 at the stager's batch boundaries (packets_popped). Chunk m goes into the zones once the
// downstream's cores read chunk m - 1's (the peer port's credits on my zone-credit word); with the router's buffers
// over the zones chunk 0 too, once its router phase is over: credits m + 1. poll_interleaved_sends: the logits /
// routing sends that go between my packets.
template <typename Connection, typename Header, typename Poll>
FORCE_INLINE void reduce_scatter_send(
    const ReduceScatterArgs& args,
    Connection& connection,
    Header* packet_headers,
    Poll&& poll_interleaved_sends,
    uint32_t& packets_popped,
    uint32_t chunk) {
    using PacketHeader = std::remove_cv_t<std::remove_pointer_t<Header>>;
    if (args.num_units == 0) {
        return;
    }
    const uint32_t credits_before_chunk_0 = args.router_done_gates_zone() ? 1 : 0;
    if (chunk + credits_before_chunk_0 > 0) {
        volatile tt_l1_ptr uint32_t* zone_credits =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.zone_credit_address);
        while (*zone_credits < chunk + credits_before_chunk_0) {
            invalidate_l1_cache();
            poll_interleaved_sends();
        }
    }
    uint32_t batch_packets = 0, batch_capacity = packets_per_group - packets_popped % packets_per_group,
             next_header = 0;
    auto pop_batch = [&]() {
        noc_async_writes_flushed();
        cb_pop_front(0, batch_packets * tiles_per_packet);
        packets_popped += batch_packets;
        batch_packets = 0;
        batch_capacity = packets_per_group - packets_popped % packets_per_group;
    };
    for (uint32_t unit_index = 0; unit_index < args.num_units; ++unit_index) {
        const ReduceScatterUnit unit = args.unit(unit_index);
        const uint32_t block = unit.block();
        const uint32_t passes =
            unit.in_pieces() ? REDUCE_SCATTER_PASSES : 1;  // (pieces: the stager's walk - pass-major over my cores)
        for (uint32_t pass = 0; pass < passes; ++pass) {
            for (uint32_t core_index = 0; core_index < args.num_cores; ++core_index) {
                const ReduceScatterCore core = args.core(core_index);
                uint32_t tiles = tiles_of_block(core.counts, block);
                const uint32_t half = core.half();
                if (!((unit.half_mask() >> half) & 1) || tiles == 0) {
                    continue;
                }
                uint32_t piece_offset = 0;  // (the piece's tile offset in the sub-block)
                if (passes > 1) {
                    const ReduceScatterPiece piece = reduce_scatter_piece(pass, core.pass_columns(), tiles);
                    piece_offset = piece.offset;
                    tiles = piece.tiles;
                }
                const bool last_piece = pass + 1 == passes;  // (the core's last piece carries its arrival increment)
                const uint32_t noc_x = core.noc_x(), noc_y = core.noc_y();
                uint32_t zone_offset;
                if (args.zone_slot_tiles) {
                    zone_offset = unit.zone_slot(half) * args.zone_slot_tiles;
                } else {
                    const uint32_t prefix_mask = unit.zone_prefix_mask(half);
                    zone_offset = 0;
                    for (uint32_t j = 0; j < 4; ++j) {
                        if ((prefix_mask >> j) & 1) {
                            zone_offset += (core.counts >> (8 * j)) & 0xFF;
                        }
                    }
                }
                const uint32_t destination_address =
                    args.zone_address + (zone_offset + piece_offset) * moe_block_tiles::kBf16TileBytes;
                const uint64_t arrival_counter_noc_address = get_noc_addr(noc_x, noc_y, args.arrival_counter_address);
                const bool live = x_gather_block_real(chunk, core.index(), block);  // (padding rows only: no payload)
                for (uint32_t first_tile = 0; first_tile < tiles; first_tile += tiles_per_packet) {
                    poll_interleaved_sends();  // (the logits / routing go between my packets: ahead of my next one)
                    while (!cb_pages_available_at_front(0, tiles_per_packet * (batch_packets + 1))) {
                        poll_interleaved_sends();
                    }
                    cb_wait_front(0, tiles_per_packet * (batch_packets + 1));
                    const uint32_t source_address =
                        get_read_ptr(0) + batch_packets * tiles_per_packet * moe_block_tiles::kBf16TileBytes;
                    const uint32_t packet_tiles =
                        tiles - first_tile < tiles_per_packet ? tiles - first_tile : tiles_per_packet;
                    const bool carries_increment = last_piece && first_tile + tiles_per_packet >= tiles;
                    if (live) {
                        const uint64_t destination_noc_address = get_noc_addr(
                            noc_x, noc_y, destination_address + first_tile * moe_block_tiles::kBf16TileBytes);
                        Header header = packet_headers[next_header];
                        next_header = next_header + 1 == packets_per_group ? 0 : next_header + 1;
                        if (carries_increment) {
                            header->to_noc_fused_unicast_write_atomic_inc(
                                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                                    destination_noc_address, arrival_counter_noc_address, 1, true},
                                packet_tiles * moe_block_tiles::kBf16TileBytes);
                        } else {
                            header->to_noc_unicast_write(
                                tt::tt_fabric::NocUnicastCommandHeader{destination_noc_address},
                                packet_tiles * moe_block_tiles::kBf16TileBytes);
                        }
                        connection.wait_for_empty_write_slot();
                        connection.send_current_slot_non_blocking(
                            source_address,
                            packet_tiles * moe_block_tiles::kBf16TileBytes,
                            reinterpret_cast<uint32_t>(header));
                    } else if (carries_increment) {  // (only the arrival count)
                        packet_headers[packets_per_group]->to_noc_unicast_atomic_inc(
                            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{arrival_counter_noc_address, 1, true});
                        connection.wait_for_empty_write_slot();
                        connection.send_payload_flush_blocking_from_address(
                            reinterpret_cast<uint32_t>(packet_headers[packets_per_group]), sizeof(PacketHeader));
                    }
                    if (++batch_packets == batch_capacity) {
                        pop_batch();
                    }
                }
            }
        }
        if (batch_packets > 0) {
            pop_batch();
        }
    }
}
