// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/api_common.h"

#include <cstdint>

#include "multicast_common.hpp"
#include "all_gather_common.hpp"

using address_t = uint32_t;

// Multicast writer: CB consumer, owns all fabric. Each packet goes once over every connection (one per direction with
// a neighbor), and together the connections reach every other device of the collective once. The init and done credits
// ride the same routes, so every device counts one of each from every other device, and done lands after our data.
void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t unit_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t packet_cap = get_compile_time_arg_val(1);
    constexpr uint32_t packets_per_slot = get_compile_time_arg_val(2);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(4);
    constexpr uint32_t num_connections = get_compile_time_arg_val(5);

    // Headers come from the pool, so the slot holds none.
    constexpr ag::SlotLayout lay = ag::slot_offsets(packets_per_slot, unit_bytes, 0, packet_cap);
    static_assert(lay.slot_bytes == cb_page_size, "host and kernel disagree on the slot layout");
    static_assert(ag::kSegmentsPerPacket <= NOC_SCATTER_WRITE_MAX_CHUNKS, "a packet has more segments than a scatter");
    // E, W, N and S. The two header sets below need 2 * num_connections of the pool's headers.
    constexpr uint32_t max_connections = 4;
    static_assert(num_connections >= 1 && num_connections <= max_connections, "no route to the other devices");

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    const uint32_t slice_count = get_arg_val<uint32_t>(arg_idx++);
    const address_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const address_t done_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_noc_x = get_arg_val<uint32_t>(arg_idx++);  // mirror core (semaphore location)
    const uint32_t sem_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t barrier_wait_value = get_arg_val<uint32_t>(arg_idx++);  // init credits to wait for
    const uint8_t e_hops = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t w_hops = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t n_hops = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t s_hops = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t e_dir = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t w_dir = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t n_dir = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t s_dir = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;

    Noc noc;
    CircularBuffer cb(cb0_id);

    ///////////////////////////////////////////////////
    // FABRIC INIT
    ///////////////////////////////////////////////////

    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    tt::tt_fabric::common::experimental::open_connections(fabric_connection, num_connections, arg_for_fab);

    // One connection per direction with a neighbor, in the host's order: the E and W lines, then the S and N spines,
    // which branch E and W at every hop. Each hop count is placed at its physical slot; an absent branch (hop == 0) is
    // skipped so it can't clobber a live slot.
    FabricRange ranges[max_connections] = {};
    uint32_t conn = 0;
    auto add_connection = [&](uint8_t spine_hops, uint8_t spine_dir, bool branches) {
        if (spine_hops == 0) {
            return;
        }
        uint8_t hops[4] = {};
        if (branches && e_hops > 0) {
            hops[e_dir] = e_hops;
        }
        if (branches && w_hops > 0) {
            hops[w_dir] = w_hops;
        }
        hops[spine_dir] = spine_hops;
        ranges[conn++] = make_fabric_range(hops[0], hops[1], hops[2], hops[3]);
    };
    add_connection(e_hops, e_dir, false);
    add_connection(w_hops, w_dir, false);
    add_connection(s_hops, s_dir, true);
    add_connection(n_hops, n_dir, true);
    ASSERT(conn == num_connections);
#ifndef FABRIC_2D
    uint8_t starts[max_connections] = {1, 1, 1, 1};  // every connection starts one hop out
#endif

    // One unicast-write route (it also carries the init and done credits) and one scatter-write route, each a header
    // per connection. Headers are reused, so every re-encode waits for the previous send to leave L1.
    const uint8_t unicast_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    const uint8_t scatter_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    // Sends one credit to the mirror core on every other device.
    auto signal_all = [&](address_t sem) {
        fabric_api::fabric_multicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            fabric_connection,
            unicast_route_id,
#ifndef FABRIC_2D
            starts,
#endif
            ranges,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{ag::fabric_frame_addr(sem_noc_x, sem_noc_y, sem), 1});
        fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::None>(
            fabric_connection, unicast_route_id);
        noc.async_writes_flushed();
    };

    // Initialization barrier: tell every other device we are alive (our output exists, and we are done reading the
    // previous launch's), then set up the data routes.
#ifndef AG_TEST_NO_INIT  // TEST ONLY
    signal_all(barrier_sem);
#endif
    fabric_api::fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
        fabric_connection,
        unicast_route_id,
#ifndef FABRIC_2D
        starts,
#endif
        ranges);
    fabric_api::fabric_multicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::None>(
        fabric_connection,
        scatter_route_id,
#ifndef FABRIC_2D
        starts,
#endif
        ranges);
    // Our data goes into every other device's output: wait until all are alive (each sent its init the same way). Our
    // reader fills the slots and does the local copy meanwhile.
#ifndef AG_TEST_NO_INIT  // TEST ONLY
    {
        AG_TEST_ZONE("AG-WR-INIT");
        ag::wait_and_consume(barrier_sem, barrier_wait_value);
    }
#endif

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    {
        AG_TEST_ZONE("AG-WR-LOOP");  // TEST ONLY
        for (bool iter_end = slice_count == 0; !iter_end;) {
            cb.wait_front(1);
            const uint32_t slot = cb.get_read_ptr();
            ag::slot_acquire();
            const ag::SlotInfo info = *reinterpret_cast<const ag::SlotInfo*>(slot);
            for (uint32_t j = 0; j < info.packets; ++j) {
                const auto& rec = *reinterpret_cast<const ag::PacketRecord*>(lay.record(slot, j));
                const uint32_t payload = lay.payload(slot, j);
                noc.async_writes_flushed();
                if (rec.segments == 1) {
                    fabric_api::fabric_multicast_noc_unicast_write_with_state<
                        UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                        fabric_connection,
                        unicast_route_id,
                        payload,
                        tt::tt_fabric::NocUnicastCommandHeader{rec.dst[0]},
                        rec.payload_bytes);
                } else {
                    fabric_api::fabric_multicast_noc_scatter_write_with_state<
                        UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                        UnicastScatterWriteUpdateMask::PayloadSize>(
                        fabric_connection,
                        scatter_route_id,
                        payload,
                        tt::tt_fabric::NocUnicastScatterCommandHeader(
                            rec.dst, rec.bytes, static_cast<uint8_t>(rec.segments)),
                        rec.payload_bytes);
                }
            }
            // Wait for Fabric writes to be sent out before popping CB entry
            noc.async_writes_flushed();
            cb.pop_front(1);
            iter_end = info.iter_end != 0;
        }
    }

    ///////////////////////////////////////////////////
    // CLEANUP
    ///////////////////////////////////////////////////

    // Completion barrier: after every data packet on the same routes, so each receiver's done count implies our data
    // has landed. Sent even with nothing to send: every reader waits for all of them.
    AG_TEST_ZONE("AG-WR-TAIL");  // TEST ONLY
    signal_all(done_sem);
    tt::tt_fabric::common::experimental::close_connections(fabric_connection);
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
