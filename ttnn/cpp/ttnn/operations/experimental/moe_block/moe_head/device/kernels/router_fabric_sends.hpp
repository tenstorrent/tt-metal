// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// What the router side sends over the fabric from the port and watcher cores: the router's logits partials to their
// owner chips (RouterLogitsStream, fabric_port_sender) and the routing exchange's packets into the destination chips'
// DRAM slots (RoutingExchangeSend: fabric_port_sender's X directions, routing_exchange_y_sender's first Y direction,
// x_gather_watcher's second). Each one's args are read once in kernel_main (read_*_args) and the sends poll them.
#pragma once

#include <cstdint>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/routing_planner.hpp"
// (no public device-side fabric API: the packet header's command types; the fabric's DRAM endpoint for its NoC)
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"

// ---- the router's logits stream ----

// A port's logits block (head_gather_program.cpp ChipProgram::port_block): num_units, num_units x (unit word, mesh id),
// num_cores, num_cores x core NoC word, the 9 words from partials_address on, then the router iterations' 3 words.
// Read once in kernel_main (read_router_logits_stream_args).
struct RouterLogitsStreamArgs {
    uint32_t num_units;  // the owner chips of my direction
    // in place: per owner chip (its fabric chip id | my slot in its arrival zone << 16, its mesh id)
    volatile tt_l1_ptr uint32_t* units;
    uint32_t num_cores;                  // the router cores of my link
    volatile tt_l1_ptr uint32_t* cores;  // in place: per router core its NoC x | y << 8
    uint32_t partials_address;           // the partials' sub-block slots (fp32)
    uint32_t handoff_words_address;      // a sub-block's hand-off word: 16 B apart, unit-major
    uint32_t arrival_zone_address;       // the owner's router cores' arrival zone ("arr")
    uint32_t arrival_counter_address;    // their arrival counters: 16 B a slot
    uint32_t tiles_per_sub_block;        // a sub-block's fp32 tiles
    uint32_t fp32_tiles_per_packet;      // a packet's fp32 tiles
    uint32_t handoff_target;             // a sub-block's hand-off word when it is in
    uint32_t sub_block_slots;            // sub-block slots a unit: core i in slot i % slots, then core i + slots told
    uint32_t slot_free_address;          // a router core's slot-free word of this direction: 16 B a unit
    // (router iterations) my "rit" word (my count of the chip's router cores done at + 16), router core 0's NoC x | y
    // << 16, the router cores' counts an iteration (host: router_row_tiles_per_iteration)
    uint32_t iteration_word_address, router_core_0_noc, router_counts_per_iteration;

    bool active() const { return num_units > 0 && num_cores > 0; }
};
// (kernel_main) the block at arg `arg_index`, which it leaves past the block
FORCE_INLINE RouterLogitsStreamArgs read_router_logits_stream_args(uint32_t& arg_index) {
    RouterLogitsStreamArgs args;
    args.num_units = moe_block_args::arg(arg_index);
    args.units = moe_block_args::arg_pointer(arg_index + 1);
    arg_index += 1 + 2 * args.num_units;
    args.num_cores = moe_block_args::arg(arg_index);
    args.cores = moe_block_args::arg_pointer(arg_index + 1);
    arg_index += 1 + args.num_cores;
    args.partials_address = moe_block_args::arg(arg_index++);
    args.handoff_words_address = moe_block_args::arg(arg_index++);
    args.arrival_zone_address = moe_block_args::arg(arg_index++);
    args.arrival_counter_address = moe_block_args::arg(arg_index++);
    args.tiles_per_sub_block = moe_block_args::arg(arg_index++);
    args.fp32_tiles_per_packet = moe_block_args::arg(arg_index++);
    args.handoff_target = moe_block_args::arg(arg_index++);
    args.sub_block_slots = moe_block_args::arg(arg_index++);
    args.slot_free_address = moe_block_args::arg(arg_index++);
    args.iteration_word_address = moe_block_args::arg(arg_index++);
    args.router_core_0_noc = moe_block_args::arg(arg_index++);
    args.router_counts_per_iteration = moe_block_args::arg(arg_index++);
    return args;
}

// The stream's progress through a call: unit (an owner chip, multi-hop) by unit, core (of my link: its partial of that
// owner's rows) by core
struct RouterLogitsStream {
    const RouterLogitsStreamArgs& args;
    uint32_t unit = 0, core = 0;

    bool done() const { return unit >= args.num_units; }
    // send what is ready; blocking: until everything is sent
    template <typename Connection, typename Header>
    void poll(Connection& connection, Header header, bool blocking) {
        while (unit < args.num_units) {
            if (core >= args.num_cores) {
                ++unit;
                core = 0;
                continue;
            }
            const uint32_t unit_word = args.units[2 * unit], mesh_id = args.units[2 * unit + 1];
            volatile tt_l1_ptr uint32_t* handoff = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                args.handoff_words_address + 16 * (unit * args.num_cores + core));
            invalidate_l1_cache();
            if (*handoff < args.handoff_target) {
                if (!blocking) {
                    return;
                }
                noc_semaphore_wait_min(handoff, args.handoff_target);
            }
            const uint32_t core_noc = args.cores[core], noc_x = core_noc & 0xFF, noc_y = (core_noc >> 8) & 0xFF;
            const uint32_t tiles = args.tiles_per_sub_block, packet_tiles_max = args.fp32_tiles_per_packet;
            const uint32_t source_address =
                args.partials_address +
                ((unit * args.sub_block_slots + core % args.sub_block_slots) * tiles) * moe_block_tiles::kFp32TileBytes;
            const uint32_t destination_address =
                args.arrival_zone_address + (unit_word >> 16) * tiles * moe_block_tiles::kFp32TileBytes;
            for (uint32_t first_tile = 0; first_tile < tiles; first_tile += packet_tiles_max) {
                const uint32_t packet_tiles =
                    tiles - first_tile < packet_tiles_max ? tiles - first_tile : packet_tiles_max;
                const uint64_t destination_noc_address =
                    get_noc_addr(noc_x, noc_y, destination_address + first_tile * moe_block_tiles::kFp32TileBytes);
                noc_async_writes_flushed();  // (the header's previous packet has left)
                (void)fabric_set_unicast_route(
                    header, static_cast<uint16_t>(unit_word & 0xFFFF), static_cast<uint16_t>(mesh_id));
                if (first_tile + packet_tiles_max >= tiles) {
                    header->to_noc_fused_unicast_write_atomic_inc(
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            destination_noc_address,
                            get_noc_addr(noc_x, noc_y, args.arrival_counter_address + 16 * (unit_word >> 16)),
                            1,
                            true},
                        packet_tiles * moe_block_tiles::kFp32TileBytes);
                } else {
                    header->to_noc_unicast_write(
                        tt::tt_fabric::NocUnicastCommandHeader{destination_noc_address},
                        packet_tiles * moe_block_tiles::kFp32TileBytes);
                }
                connection.wait_for_empty_write_slot();
                connection.send_current_slot_non_blocking(
                    source_address + first_tile * moe_block_tiles::kFp32TileBytes,
                    packet_tiles * moe_block_tiles::kFp32TileBytes,
                    reinterpret_cast<uint32_t>(header));
            }
            if (core + args.sub_block_slots < args.num_cores) {  // (the slot's next core may write it once it left)
                noc_async_writes_flushed();
                const uint32_t next_core_noc = args.cores[core + args.sub_block_slots];
                noc_semaphore_inc(
                    get_noc_addr(next_core_noc & 0xFF, (next_core_noc >> 8) & 0xFF, args.slot_free_address + 16 * unit),
                    1);
            }
            ++core;
        }
    }
    void rearm() {
        for (uint32_t word = 0; word < args.num_units * args.num_cores; ++word) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.handoff_words_address + 16 * word) = 0;
        }
    }
    // the next router iteration's sub-blocks (its slots and words reused: the matmul cores write them only once router
    // core 0 told them go, past this iteration's count of mine)
    void next_iteration() {
        rearm();
        unit = 0;
        core = 0;
    }
};

// ---- the routing exchange's send ----

// A routing exchange send block (head_router_program.cpp ChipProgram::send_block): num_destinations (0: nothing to send
// - the block is this one word), then the 12 words from num_remote on, then a fabric chip id a destination. Read once
// in kernel_main (read_routing_exchange_send_args).
struct RoutingExchangeSendArgs {
    uint32_t num_destinations;
    uint32_t num_remote;                 // the destinations sent over the fabric; the rest (the last) is my own chip
    uint32_t includes_self;              // the last destination is my own chip
    uint32_t packets_address;            // the destinations' packets in L1, slot_bytes apart
    uint32_t slot_bytes;                 // a DRAM slot's (and an L1 packet's) capacity
    uint32_t slots_address;              // the DRAM slots tensor
    uint32_t arrival_counter_0_address;  // planner core 0's arrival counters: a call of parity 0
    uint32_t arrival_counter_1_address;  // ... of parity 1
    uint32_t planner_core_noc_x, planner_core_noc_y;  // planner core 0
    uint32_t calls_address;  // my call counter (every port and planner core counts its calls: they stay in step)
    uint32_t ready_address;  // the planner instances' "ready" count
    uint32_t mesh_id;        // my mesh id (every destination's)
    volatile tt_l1_ptr uint32_t* destination_chip_ids;  // in place: [num_destinations] (my own: 0)
};
// (kernel_main) the block at arg `arg_index`, which it leaves past the block
FORCE_INLINE RoutingExchangeSendArgs read_routing_exchange_send_args(uint32_t& arg_index) {
    RoutingExchangeSendArgs args{};
    args.num_destinations = moe_block_args::arg(arg_index++);
    if (args.num_destinations == 0) {
        return args;
    }
    args.num_remote = moe_block_args::arg(arg_index++);
    args.includes_self = moe_block_args::arg(arg_index++);
    args.packets_address = moe_block_args::arg(arg_index++);
    args.slot_bytes = moe_block_args::arg(arg_index++);
    args.slots_address = moe_block_args::arg(arg_index++);
    args.arrival_counter_0_address = moe_block_args::arg(arg_index++);
    args.arrival_counter_1_address = moe_block_args::arg(arg_index++);
    args.planner_core_noc_x = moe_block_args::arg(arg_index++);
    args.planner_core_noc_y = moe_block_args::arg(arg_index++);
    args.calls_address = moe_block_args::arg(arg_index++);
    args.ready_address = moe_block_args::arg(arg_index++);
    args.mesh_id = moe_block_args::arg(arg_index++);
    args.destination_chip_ids = moe_block_args::arg_pointer(arg_index);
    arg_index += args.num_destinations;
    return args;
}

// one packet straight to its destination, the fabric routing every hop (Y then X, its turn included): into the
// destination's DRAM slot page parity N + source chip, max_packet_payload_bytes at a time, the last a fused
// increment of its arrival counter (by parity) on planner core 0. No relay on the way: no chip re-reads and re-sends
// another chip's packet, so a packet's parity is always its sender's own call count.
template <typename Connection, typename Header>
FORCE_INLINE void routing_exchange_hop(
    Connection& connection,
    Header header,
    const RoutingExchangeSendArgs& args,
    uint32_t source_address,
    uint32_t bytes,
    uint32_t chip_id,
    uint32_t parity,
    uint32_t source_chip,
    uint32_t num_chips,
    uint32_t max_packet_payload_bytes) {
    // fabric router writes on NOC1: resolve the DRAM endpoint for its NoC (SYS-1419)
    const uint64_t destination_noc_address = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
        TensorAccessor(routing_accessors::slots, args.slots_address, args.slot_bytes),
        parity * num_chips + source_chip);
    const uint64_t arrival_counter_noc_address = get_noc_addr(
        args.planner_core_noc_x,
        args.planner_core_noc_y,
        parity ? args.arrival_counter_1_address : args.arrival_counter_0_address);
    for (uint32_t offset = 0; offset < bytes; offset += max_packet_payload_bytes) {
        const uint32_t packet_bytes =
            bytes - offset < max_packet_payload_bytes ? bytes - offset : max_packet_payload_bytes;
        noc_async_writes_flushed();  // (the header's previous packet has left)
        (void)fabric_set_unicast_route(header, static_cast<uint16_t>(chip_id), static_cast<uint16_t>(args.mesh_id));
        if (offset + packet_bytes >= bytes) {
            header->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                    destination_noc_address + offset, arrival_counter_noc_address, 1, true},
                packet_bytes);
        } else {
            header->to_noc_unicast_write(
                tt::tt_fabric::NocUnicastCommandHeader{destination_noc_address + offset}, packet_bytes);
        }
        connection.wait_for_empty_write_slot();
        connection.send_current_slot_non_blocking(
            source_address + offset, packet_bytes, reinterpret_cast<uint32_t>(header));
    }
}
// a packet's bytes: its header, then a word a routed token of each planner instance (their counts: its first words)
FORCE_INLINE uint32_t
routing_exchange_bytes(uint32_t packet_address, uint32_t planner_instances, uint32_t header_bytes) {
    const uint32_t* instance_counts = reinterpret_cast<const uint32_t*>(packet_address);
    uint32_t tokens = 0;
    for (uint32_t instance = 0; instance < planner_instances; ++instance) {
        tokens += instance_counts[instance];
    }
    return (header_bytes + 4 * tokens + 63) / 64 * 64;
}
// The send's progress through a call: once the planner instances are ready, every destination's packet at once
struct RoutingExchangeSend {
    const RoutingExchangeSendArgs& args;
    uint32_t done = args.num_destinations == 0 ? 1 : 0;

    template <typename Connection, typename Header>
    void poll(
        Connection& connection,
        Header header,
        bool blocking,
        uint32_t num_chips,
        uint32_t planner_instances,
        uint32_t experts_per_chip,
        uint32_t my_chip,
        uint32_t header_bytes,
        uint32_t expert_count_block_bytes,
        uint32_t instance_block_bytes,
        uint32_t max_packet_payload_bytes) {
        if (done) {
            return;
        }
        volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ready_address);
        invalidate_l1_cache();
        if (*ready < planner_instances) {
            if (!blocking) {
                return;
            }
            noc_semaphore_wait_min(ready, planner_instances);
        }
        DeviceZoneScopedN("SE-RASEND");
        noc_semaphore_inc(get_noc_addr(args.ready_address), 0u - planner_instances);
        noc_async_atomic_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* calls = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.calls_address);
        const uint32_t parity = calls[0] & 1;  // (every port core and planner core counts its calls: in step)
        calls[0] = calls[0] + 1;
        for (uint32_t destination = 0; destination < args.num_destinations; ++destination) {
            const uint32_t packet_address = args.packets_address + destination * args.slot_bytes;
            // (the per-expert totals: the planners' - the destination's owner instance wrote them into the header)
            reinterpret_cast<uint32_t*>(packet_address)[14] = parity;   // (the call's parity and the source: a hang
            reinterpret_cast<uint32_t*>(packet_address)[15] = my_chip;  // dump reads them)
            const uint32_t bytes = routing_exchange_bytes(packet_address, planner_instances, header_bytes);
            if (destination < args.num_remote) {
                routing_exchange_hop(
                    connection,
                    header,
                    args,
                    packet_address,
                    bytes,
                    args.destination_chip_ids[destination],
                    parity,
                    my_chip,
                    num_chips,
                    max_packet_payload_bytes);
            } else {  // (SELF: my own chip's packet)
                noc_async_write(
                    packet_address,
                    TensorAccessor(routing_accessors::slots, args.slots_address, args.slot_bytes)
                        .get_noc_addr(parity * num_chips + my_chip),
                    bytes);
                noc_async_write_barrier();
                noc_semaphore_inc(
                    get_noc_addr(
                        args.planner_core_noc_x,
                        args.planner_core_noc_y,
                        parity ? args.arrival_counter_1_address : args.arrival_counter_0_address),
                    1);
            }
        }
        noc_async_writes_flushed();
        done = 1;
    }
};
