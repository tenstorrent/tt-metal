// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"

using namespace tt::tt_fabric;
using namespace tt::tt_fabric::linear::experimental;

constexpr uint32_t cb_stage = get_compile_time_arg_val(0);
constexpr uint32_t cb_eout = get_compile_time_arg_val(1);  // at the same address on the relay cores
constexpr uint32_t sp_size = get_compile_time_arg_val(2);
constexpr uint32_t local_rows = get_compile_time_arg_val(3);
constexpr uint32_t unit_bytes = get_compile_time_arg_val(4);   // one tile column of a head's state
constexpr uint32_t chunk_bytes = get_compile_time_arg_val(5);  // one fabric packet
constexpr uint32_t kt = get_compile_time_arg_val(6);           // tile columns per state
constexpr uint32_t slots = get_compile_time_arg_val(7);        // unit buffers
// On the relay cores, per head slot: the releases of the head.
constexpr uint32_t first_ack_semaphore_id = get_compile_time_arg_val(8);
constexpr uint32_t relay_heads = get_compile_time_arg_val(9);  // a relay core's head slots
constexpr auto start_args = TensorAccessorArgs<10>();
// On a 2D fabric whose SP axis turns, a multicast cannot follow the axis: the carry goes to each rank this sender
// serves as a 2D unicast (per rank after the connection args: the connection its route leaves on, its chip and mesh),
// carry_range ranks in all.
constexpr bool unicast_peers = get_compile_time_arg_val(start_args.next_compile_time_args_offset());
constexpr uint32_t chunks = (unit_bytes + chunk_bytes - 1) / chunk_bytes;
constexpr uint32_t state_bytes = kt * unit_bytes;
// One packet header per packet of each slot, out of this RISC's share of the pool.
static_assert(
    slots * chunks <= MEM_PACKET_HEADER_POOL_SIZE / MaxDMProcessorsPerCoreType / PACKET_HEADER_MAX_SIZE,
    "the sender's packet headers exceed the packet header pool");

// Sender core: the relay's sender of a link reads each finished tile column out of its heads' relay cores and forwards
// it into the head's slot of the successor's landing buffer (the first rank's final-state slot, from the last rank)
// over the fabric. On the chronologically first rank, every sender then multicasts the final carry the relay cores
// form, carry_range ranks along its one connection (or unicasts it to each of them), into the head's final-state slot
// of every rank's relay core, which writes it out; together a link's senders reach every other rank once.
void kernel_main() {
    size_t arg = 0;
    const uint32_t start_address = get_common_arg_val<uint32_t>(0);
    // The landing buffer (same on every relay core): a slot per head of the successor's entry states, then of the
    // final states (the first rank's relayed final state, every other rank's carry).
    const uint32_t ein_landing = get_common_arg_val<uint32_t>(1);
    const uint32_t final_landing = ein_landing + relay_heads * state_bytes;
    const uint32_t eout_address = get_write_ptr(cb_eout);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg++));
    const uint32_t arrival_semaphore = get_arg_val<uint32_t>(arg++);
    const uint32_t final_semaphore = get_arg_val<uint32_t>(arg++);
    const uint32_t sp_rank = get_arg_val<uint32_t>(arg++);
    // 1D routing: hops to the successor, 1 forward, or sp_size - 1 backward from the last rank of a line.
    const uint32_t successor_hops = get_arg_val<uint32_t>(arg++);
    const uint32_t num_heads = get_arg_val<uint32_t>(arg++);  // entries: the head's relay core NoC x | y << 16
    const size_t head_args = arg;
    arg += num_heads;
    const size_t notify_args = arg;  // then each head's count semaphore id on this core
    arg += num_heads;
    const size_t slot_args = arg;  // then each head's slot on its relay core
    arg += num_heads;
    const size_t carries_args = arg;  // then whether this sender carries each head's final carry
    arg += num_heads;
    // The relay's own sender, else the link's second carry sender, which sends towards the predecessor.
    const bool relay = get_arg_val<uint32_t>(arg++);
    const uint8_t carry_range = get_arg_val<uint32_t>(arg++);
    // A second carry sender without ranks behind its rank holds no connection.
    const uint32_t connections = get_arg_val<uint32_t>(arg++);

    cb_reserve_back(cb_stage, 1);
    const uint32_t stage = get_write_ptr(cb_stage);
    noc_async_read(TensorAccessor(start_args, start_address).get_noc_addr(0), stage, sizeof(uint32_t));
    noc_async_read_barrier();
    const auto topology =
        kda_chronology::derive(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[0], sp_rank, sp_size, local_rows);
    const uint32_t position = (topology.rank + sp_size - topology.first_rank) % sp_size;
    const bool last = position == sp_size - 1;

    RoutingPlaneConnectionManager fabric;
    open_connections(fabric, connections, arg);
    const size_t peer_args = arg;
    // A unit is one tile column of a head's state (column-major storage): a relay core computes that column of the
    // next state as soon as it lands. Units go through `slots` buffers: unit u + slots - 1 is read out of its relay
    // core while earlier units' packets are on the link, and a slot (and its packet headers) is reused only after its
    // packets have left L1.
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[slots][chunks];
    for (uint32_t b = 0; b < slots; ++b) {
        for (uint32_t c = 0; c < chunks; ++c) {
            hdrs[b][c] = PacketHeaderPool::allocate_header();
        }
    }
    const uint32_t units = num_heads * kt;
    const auto head_xy = [&](uint32_t u) { return get_arg_val<uint32_t>(head_args + u / kt); };
    const auto head_slot = [&](uint32_t u) { return get_arg_val<uint32_t>(slot_args + u / kt); };
    // Unit u's column in its head's slot (of cb_eout, and of a landing buffer's states).
    const auto column_offset = [&](uint32_t u) { return head_slot(u) * state_bytes + (u % kt) * unit_bytes; };
    const auto release = [&](uint32_t u) {
        const uint32_t xy = head_xy(u);
        noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, get_semaphore(first_ack_semaphore_id + head_slot(u))), 1);
    };
    // Streams the units of the heads `selected` picks through the slots: each once its relay core's writer has counted
    // it (its column, after `counted` earlier ones, in the head's count semaphore on this core), read under its slot's
    // transaction id so a unit waits only for its own read; send(u, buffer) puts it on the fabric. A head's slot is
    // released once its last column has been read.
    const auto stream = [&](uint32_t counted, const auto& selected, const auto& send) {
        uint32_t picked[16];  // a sender serves at most relay_chunk_scan_max_heads_per_link heads
        uint32_t picks = 0;
        for (uint32_t h = 0; h < num_heads; ++h) {
            if (selected(h)) {
                picked[picks++] = h;
            }
        }
        const uint32_t units = picks * kt;
        // The k-th streamed unit is column k % kt of the picked head k / kt.
        const auto unit = [&](uint32_t k) { return picked[k / kt] * kt + k % kt; };
        // Blackhole's data-movement RISCs cache L1: invalidate before each poll, as noc_semaphore_wait_min does.
        const auto ready = [&](uint32_t u) {
            invalidate_l1_cache();
            return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                       get_semaphore(get_arg_val<uint32_t>(notify_args + u / kt)))[0] >= counted + u % kt + 1;
        };
        uint32_t fetched = 0;
        for (uint32_t k = 0; k < units; ++k) {
            // Issue every read the free slots allow, waiting only for unit k itself.
            while (fetched < units && fetched < k + slots && (fetched == k || ready(unit(fetched)))) {
                while (!ready(unit(fetched))) {
                }
                if (fetched >= slots) {
                    // Unit fetched - slots last used this slot: its packets must have left L1.
                    noc_async_writes_flushed();
                }
                const uint32_t u = unit(fetched);
                const uint32_t xy = head_xy(u);
                noc_async_read_set_trid(1 + fetched % slots);
                noc_async_read(
                    get_noc_addr(xy & 0xFFFF, xy >> 16, eout_address + column_offset(u)),
                    stage + (fetched % slots) * unit_bytes,
                    unit_bytes);
                ++fetched;
            }
            noc_async_read_barrier_with_trid(1 + k % slots);
            const uint32_t u = unit(k);
            if (u % kt == kt - 1) {
                release(u);
            }
            send(u, k % slots, stage + (k % slots) * unit_bytes);
        }
        noc_async_read_set_trid(0);
        noc_async_writes_flushed();
    };
    // A unit's packets, each `chunk_bytes` but the last.
    const auto chunk_size = [](uint32_t c) { return c + 1 == chunks ? unit_bytes - c * chunk_bytes : chunk_bytes; };

    if (relay) {
        auto* to_successor = &fabric.get(0).sender;
        for (uint32_t b = 0; b < slots; ++b) {
            for (uint32_t c = 0; c < chunks; ++c) {
#if defined(FABRIC_2D)
                // 2D routing: the hand-off from the last rank to the first crosses the mesh when the fabric does not
                // wrap the SP axis.
                (void)fabric_set_unicast_route(hdrs[b][c], dst_chip_id, dst_mesh_id);
#else
                (void)dst_chip_id;
                (void)dst_mesh_id;
#endif
            }
        }
        const uint32_t landing = last ? final_landing : ein_landing;
        const uint32_t landed_semaphore = last ? final_semaphore : arrival_semaphore;
        stream(
            0,
            [](uint32_t) { return true; },
            [&](uint32_t u, uint32_t slot, uint32_t buffer) {
                const uint32_t xy = head_xy(u);
                const uint32_t column = landing + column_offset(u);
                for (uint32_t c = 0; c < chunks; ++c) {
                    const uint64_t dst = get_noc_addr(xy & 0xFFFF, xy >> 16, column + c * chunk_bytes);
                    // 1D routing: the connection leaves towards the successor's side of the line (2D ignores the hops).
                    if (c + 1 == chunks) {
                        fabric_unicast_noc_fused_unicast_with_atomic_inc(
                            to_successor,
                            hdrs[slot][c],
                            buffer + c * chunk_bytes,
                            chunk_size(c),
                            NocUnicastAtomicIncFusedCommandHeader{
                                dst, get_noc_addr(xy & 0xFFFF, xy >> 16, landed_semaphore), 1, true},
                            successor_hops);
                    } else {
                        fabric_unicast_noc_unicast_write(
                            to_successor,
                            hdrs[slot][c],
                            buffer + c * chunk_bytes,
                            chunk_size(c),
                            NocUnicastCommandHeader{dst},
                            successor_hops);
                    }
                }
            });
    }

    const auto carries = [&](uint32_t h) { return get_arg_val<uint32_t>(carries_args + h) != 0; };
    if (position == 0 && carry_range != 0) {
        // The final carry, a head's columns after its relay's on the relay's sender: each column into the head's
        // final-state slot of its relay core on every rank this sender reaches (the same core everywhere), counted
        // there in the final semaphore with the column's last packet.
        const auto destination = [&](uint32_t u, uint32_t c) {
            const uint32_t xy = head_xy(u);
            return NocUnicastAtomicIncFusedCommandHeader{
                get_noc_addr(xy & 0xFFFF, xy >> 16, final_landing + column_offset(u) + c * chunk_bytes),
                get_noc_addr(xy & 0xFFFF, xy >> 16, final_semaphore),
                1,
                true};
        };
        if constexpr (unicast_peers) {
            // Unicast: a slot's headers per unit and rank in flight, rerouted to the next rank once flushed.
            uint32_t group = 0;
            stream(relay ? kt : 0, carries, [&](uint32_t u, uint32_t slot, uint32_t buffer) {
                for (uint32_t peer = 0; peer < carry_range; ++peer) {
                    if (group != 0 && group % slots == 0) {
                        noc_async_writes_flushed();
                    }
                    auto* sender = &fabric.get(get_arg_val<uint32_t>(peer_args + 3 * peer)).sender;
                    for (uint32_t c = 0; c < chunks; ++c) {
                        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[group % slots][c];
#if defined(FABRIC_2D)
                        fabric_set_unicast_route(
                            hdr,
                            static_cast<uint16_t>(get_arg_val<uint32_t>(peer_args + 3 * peer + 1)),
                            static_cast<uint16_t>(get_arg_val<uint32_t>(peer_args + 3 * peer + 2)));
#endif
                        const auto command = destination(u, c);
                        if (c + 1 == chunks) {
                            fabric_unicast_noc_fused_unicast_with_atomic_inc(
                                sender, hdr, buffer + c * chunk_bytes, chunk_size(c), command, 1);
                        } else {
                            fabric_unicast_noc_unicast_write(
                                sender,
                                hdr,
                                buffer + c * chunk_bytes,
                                chunk_size(c),
                                NocUnicastCommandHeader{command.noc_address},
                                1);
                        }
                    }
                    ++group;
                }
            });
        } else {
            uint8_t range[1] = {carry_range};
            for (uint32_t b = 0; b < slots; ++b) {
                for (uint32_t c = 0; c < chunks; ++c) {
                    fabric_set_mcast_route(fabric, hdrs[b][c], range, 0);
                    if (c + 1 == chunks) {
                        fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state<
                            UnicastFusedAtomicIncUpdateMask::Val | UnicastFusedAtomicIncUpdateMask::Flush |
                            UnicastFusedAtomicIncUpdateMask::PayloadSize>(
                            hdrs[b][c],
                            1,
                            range[0],
                            NocUnicastAtomicIncFusedCommandHeader{0, 0, 1, true},
                            chunk_size(c));
                    } else {
                        fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
                            hdrs[b][c], 1, range[0], nullptr, chunk_size(c));
                    }
                }
            }
            auto* sender = &fabric.get(0).sender;
            stream(relay ? kt : 0, carries, [&](uint32_t u, uint32_t slot, uint32_t buffer) {
                for (uint32_t c = 0; c < chunks; ++c) {
                    const auto command = destination(u, c);
                    if (c + 1 == chunks) {
                        fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<
                            UnicastFusedAtomicIncUpdateMask::WriteDstAddr |
                            UnicastFusedAtomicIncUpdateMask::SemaphoreAddr>(
                            sender, hdrs[slot][c], buffer + c * chunk_bytes, command);
                    } else {
                        fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                            sender,
                            hdrs[slot][c],
                            buffer + c * chunk_bytes,
                            NocUnicastCommandHeader{command.noc_address});
                    }
                }
            });
        }
    }
    close_connections(fabric);
    noc_async_atomic_barrier();
}
