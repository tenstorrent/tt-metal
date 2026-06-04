// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dedicated fabric "gather" kernel for the multi-device all_gather_rms_norm (designated-gather, no mux).
//
// A few (= routing planes, e.g. 2) dedicated cores own the direct fabric connections. Worker cores compute
// their per-row local stat partial, copy it into their OWN gathered-stats slot, then NoC-write (partial +
// the destination metadata) into this gather core's per-worker relay slot and bump its `relay_ready`
// semaphore. This kernel waits for all its workers, then for each relay slot line-multicasts the partial to
// the SAME gathered-stats slot on every ring peer (the peer worker sits at the same core coords, so the
// worker-supplied NoC address routes there) and atomic-incs that peer worker's out-ready semaphore.
//
// The relay buffer lives at the unreserved L1 base on this (CB-less) gather core, a host-known address the
// worker writes to. Per-worker relay slot layout (bytes), written by the worker, read here:
//   [0 .. stat_tile_bytes)      : the partial tile (fabric write payload, sourced from THIS core's L1)
//   [stat_tile_bytes   .. +8)   : dst gathered-slot NoC address (u64, write-cmd form; lo word first)
//   [stat_tile_bytes+8 .. +16)  : peer out-ready semaphore NoC address (u64, atomic-inc packet form)
//
// Fabric mechanics mirror the historical per-worker direct-fabric writer (commit b2c0875e67f) and
// ccl/broadcast/device/kernels/broadcast_tile_writer.cpp.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
using namespace tt::tt_fabric::linear::experimental;

void kernel_main() {
    // Compile-time args (must match gather_ct_args in the program factory).
    constexpr uint32_t stat_tile_bytes = get_compile_time_arg_val(0);     // bytes of one partial tile
    constexpr uint32_t slot_stride_bytes = get_compile_time_arg_val(1);   // relay slot stride (tile + metadata)
    constexpr uint32_t relay_ready_sem_id = get_compile_time_arg_val(2);  // our local sem workers bump per round
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(3);         // worker-local back-pressure sem we inc
    constexpr uint32_t start_hops_forward = get_compile_time_arg_val(4);
    constexpr uint32_t range_hops_forward = get_compile_time_arg_val(5);
    constexpr uint32_t start_hops_backward = get_compile_time_arg_val(6);
    constexpr uint32_t range_hops_backward = get_compile_time_arg_val(7);

    // Runtime args.
    uint32_t ai = 0;
    const uint32_t relay_base = get_arg_val<uint32_t>(ai++);        // L1 base of the relay buffer on this core
    const uint32_t num_workers = get_arg_val<uint32_t>(ai++);       // total workers this gather core serves
    const uint32_t num_super_chunks = get_arg_val<uint32_t>(ai++);  // rows-per-worker rounds (round-robin)
    const uint32_t num_connections = get_arg_val<uint32_t>(ai++);
    // active_count[sc] = workers participating in round sc (a 0..K-1 prefix, since the contiguous row split is
    // monotonic in worker index). num_super_chunks values.
    const uint32_t active_count_arg_base = ai;
    ai += num_super_chunks;
    // Per-worker virtual coords (x then y interleaved): 2 * num_workers values, for the done-sem inc.
    const uint32_t worker_coords_arg_base = ai;
    ai += 2 * num_workers;
    size_t arg_for_fab = ai;  // remaining args belong to the fabric connection manager

    volatile tt_l1_ptr uint32_t* relay_ready_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(relay_ready_sem_id));

    // Open the direct fabric connection(s) on this core's routing plane(s) and pre-configure the
    // line-multicast route for the write + the atomic-inc once.
    auto write_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    auto sem_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, num_connections, arg_for_fab);

    uint8_t starts[] = {static_cast<uint8_t>(start_hops_forward), static_cast<uint8_t>(start_hops_backward)};
    uint8_t ranges[] = {static_cast<uint8_t>(range_hops_forward), static_cast<uint8_t>(range_hops_backward)};
    if (ranges[0] == 0) {  // no forward hops -> fold the backward route into slot 0
        starts[0] = starts[1];
        ranges[0] = ranges[1];
    }
    fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        fabric_connection, write_route_id, starts, ranges, nullptr, stat_tile_bytes);
    fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        fabric_connection, sem_route_id, starts, ranges, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1});

    for (uint32_t sc = 0; sc < num_super_chunks; sc++) {
        const uint32_t active = get_arg_val<uint32_t>(active_count_arg_base + sc);  // workers active this round
        // Wait until every active worker has staged its partial + metadata for this round.
        noc_semaphore_wait(relay_ready_ptr, active);
        noc_semaphore_set(relay_ready_ptr, 0);

        for (uint32_t w = 0; w < active; w++) {
            const uint32_t slot = relay_base + w * slot_stride_bytes;
            volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + stat_tile_bytes);
            const uint64_t dst_noc_addr = (static_cast<uint64_t>(meta[1]) << 32) | meta[0];
            const uint64_t out_ready_noc_addr_pkt = (static_cast<uint64_t>(meta[3]) << 32) | meta[2];

            // Line-multicast this worker's partial to the same gathered-stats slot on every ring peer,
            // then signal that peer worker's out-ready semaphore.
            fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                fabric_connection, write_route_id, slot, tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr});
            fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric_connection,
                sem_route_id,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_noc_addr_pkt, 0});
        }
        noc_async_writes_flushed();

        // Back-pressure: tell each active worker that this round's relay slot is free to reuse (so it can
        // produce its next row without racing our read of the current one).
        for (uint32_t w = 0; w < active; w++) {
            const uint32_t wx = get_arg_val<uint32_t>(worker_coords_arg_base + 2 * w);
            const uint32_t wy = get_arg_val<uint32_t>(worker_coords_arg_base + 2 * w + 1);
            const uint64_t done_noc_addr = safe_get_noc_addr(wx, wy, get_semaphore(done_sem_id));
            noc_semaphore_inc(done_noc_addr, 1);
        }
        noc_async_atomic_barrier();
    }

    close_connections(fabric_connection);
    noc_async_write_barrier();
}
