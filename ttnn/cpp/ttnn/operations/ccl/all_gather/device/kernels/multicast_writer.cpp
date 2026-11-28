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

#ifdef FABRIC_2D
#include "tt_metal/fabric/hw/inc/mesh/api.h"
namespace fabric_api = tt::tt_fabric::mesh::experimental;
using FabricRange = tt::tt_fabric::mesh::experimental::MeshMcastRange;
inline FabricRange make_fabric_range(uint8_t e, uint8_t w, uint8_t n, uint8_t s) { return FabricRange{e, w, n, s}; }
#else
#include "tt_metal/fabric/hw/inc/linear/api.h"
namespace fabric_api = tt::tt_fabric::linear::experimental;
using FabricRange = uint8_t;  // under 1D each connection carries a single hop count
// 1D has a single active direction, so exactly one direction's hop count is nonzero
inline FabricRange make_fabric_range(uint8_t e, uint8_t w, uint8_t n, uint8_t s) { return e + w + n + s; }
#endif

#include "all_gather_common.hpp"

using address_t = uint32_t;
namespace ag = ttnn::operations::ccl::all_gather_walk;

// Multicast writer: CB consumer, owns all fabric. Each packet goes once over every connection (one per direction with
// a neighbor), and together the connections reach every other device of the collective once.
//
// Routes. On an even ring the forward route reaches N/2 devices and the backward route N/2-1, so with
// load_balance_across_alt_routes the packets alternate between route 0 and route 1, which swaps the two hop counts and
// so also reaches every other device once. Init rides route 0 only. Done rides every route, each after that route's
// data on the same connections, so it lands after the data it covers: a device reached by route 0 forward and route 1
// backward (the antipode) would otherwise count route 0's done before route 1's data. Every device counts one done per
// route from every other device.
void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t unit_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t packet_cap = get_compile_time_arg_val(1);
    constexpr uint32_t packets_per_slot = get_compile_time_arg_val(2);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(3);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t num_connections = get_compile_time_arg_val(5);
    constexpr bool load_balance_across_alt_routes = get_compile_time_arg_val(6) != 0;

    // Headers come from the pool, so the slot holds none.
    constexpr ag::SlotLayout lay = ag::checked_slot_layout<packets_per_slot, unit_bytes, 0, packet_cap, slot_bytes>();
    // E, W, N and S.
    constexpr uint32_t max_connections = 4;
    static_assert(num_connections >= 1 && num_connections <= max_connections, "no route to the other devices");
    // Both directions of one ring axis.
    static_assert(
        !load_balance_across_alt_routes || num_connections == 2, "alternate routes need exactly one ring axis");
    constexpr uint32_t num_routes = load_balance_across_alt_routes ? 2 : 1;
    // A unicast-write and a scatter-write header set per route, a header per connection in each.
    static_assert(
        2 * num_routes * num_connections <= NUM_PACKET_HEADERS / MaxDMProcessorsPerCoreType,
        "the routes need more packet headers than the pool gives this RISC");

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    const uint32_t slice_count = get_arg_val<uint32_t>(arg_idx++);
    const address_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const address_t done_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);  // mirror core (semaphore location)
    const uint32_t sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_init_credits = get_arg_val<uint32_t>(arg_idx++);  // one from every other device
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
    // which branch E and W at every hop. Each hop count is placed at its direction's index; an absent branch (hop == 0)
    // is skipped so it can't clobber a live direction's hop count.
    FabricRange ranges[num_routes][max_connections] = {};
    auto build_ranges = [&](FabricRange* route, uint8_t e, uint8_t w, uint8_t n, uint8_t s) {
        uint32_t conn = 0;
        auto add_connection = [&](uint8_t spine_hops, uint8_t spine_dir, bool branches) {
            if (spine_hops == 0) {
                return;
            }
            uint8_t hops[4] = {};
            if (branches && e > 0) {
                hops[e_dir] = e;
            }
            if (branches && w > 0) {
                hops[w_dir] = w;
            }
            hops[spine_dir] = spine_hops;
            route[conn++] = make_fabric_range(hops[0], hops[1], hops[2], hops[3]);
        };
        add_connection(e, e_dir, false);
        add_connection(w, w_dir, false);
        add_connection(s, s_dir, true);
        add_connection(n, n_dir, true);
        ASSERT(conn == num_connections);
    };
    build_ranges(ranges[0], e_hops, w_hops, n_hops, s_hops);
    if constexpr (load_balance_across_alt_routes) {
        // The ring axis's two hop counts swapped (the host enables this only with both nonzero, so the connections
        // keep their order); the inactive axis's are zero either way.
        build_ranges(ranges[1], w_hops, e_hops, s_hops, n_hops);
    }
#ifndef FABRIC_2D
    uint8_t starts[max_connections] = {1, 1, 1, 1};  // every connection starts one hop out
#endif

    // Per route, a unicast-write header set (it also carries the route's init and done credits) and a scatter-write
    // header set, a header per connection in each. Headers are reused, so every re-encode waits for the previous send
    // to leave L1.
    uint8_t unicast_route_id[num_routes];
    uint8_t scatter_route_id[num_routes];
    for (uint32_t r = 0; r < num_routes; ++r) {
        unicast_route_id[r] = PacketHeaderPool::allocate_header_n(num_connections);
        scatter_route_id[r] = PacketHeaderPool::allocate_header_n(num_connections);
    }
    // Sends one credit to the mirror core on every other device, over route r.
    auto signal_all = [&](address_t sem, uint32_t r) {
        fabric_api::fabric_multicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            fabric_connection,
            unicast_route_id[r],
#ifndef FABRIC_2D
            starts,
#endif
            ranges[r],
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{ag::noc0_addr(sem_noc0_x, sem_noc0_y, sem), 1});
        fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::None>(
            fabric_connection, unicast_route_id[r]);
        noc.async_writes_flushed();
    };

    // Init: tell every other device we are alive (our output exists, and we are done reading the
    // previous launch's), then set up the data routes.
    signal_all(barrier_sem, 0);
    for (uint32_t r = 0; r < num_routes; ++r) {
        fabric_api::fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
            fabric_connection,
            unicast_route_id[r],
#ifndef FABRIC_2D
            starts,
#endif
            ranges[r]);
        fabric_api::fabric_multicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::None>(
            fabric_connection,
            scatter_route_id[r],
#ifndef FABRIC_2D
            starts,
#endif
            ranges[r]);
    }
    // Our data goes into every other device's output: wait until all are alive (each sent its init the same way). Our
    // reader fills the slots and does the local copy meanwhile.
    ag::wait_and_consume(barrier_sem, num_init_credits);

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    uint32_t route = 0;  // toggled per packet under load_balance_across_alt_routes
    for (bool iter_end = slice_count == 0; !iter_end;) {
        cb.wait_front(1);
        const uint32_t slot = cb.get_read_ptr();
        ag::slot_acquire();
        const ag::SlotInfo info = *reinterpret_cast<const ag::SlotInfo*>(slot);
        for (uint32_t j = 0; j < info.packets; ++j) {
            const auto& rec = *reinterpret_cast<const ag::PacketRecord*>(lay.record(slot, j));
            const uint32_t payload = lay.payload(slot, j);
            noc.async_writes_flushed();
            if (rec.num_chunks == 1) {
                fabric_api::fabric_multicast_noc_unicast_write_with_state<
                    UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                    fabric_connection,
                    unicast_route_id[route],
                    payload,
                    tt::tt_fabric::NocUnicastCommandHeader{rec.dst[0]},
                    rec.payload_bytes);
            } else {
                fabric_api::fabric_multicast_noc_scatter_write_with_state<
                    UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                    UnicastScatterWriteUpdateMask::PayloadSize>(
                    fabric_connection,
                    scatter_route_id[route],
                    payload,
                    tt::tt_fabric::NocUnicastScatterCommandHeader(
                        rec.dst, rec.chunk_bytes, static_cast<uint8_t>(rec.num_chunks)),
                    rec.payload_bytes);
            }
            if constexpr (load_balance_across_alt_routes) {
                route ^= 1;
            }
        }
        // Wait for the fabric writes to leave L1 before popping the slot
        noc.async_writes_flushed();
        cb.pop_front(1);
        iter_end = info.iter_end != 0;
    }

    ///////////////////////////////////////////////////
    // CLEANUP
    ///////////////////////////////////////////////////

    // Done: on every route, after every data packet on the same connections, so each receiver's done count implies our
    // data has landed. Sent even with nothing to send: every reader waits for all of them.
    for (uint32_t r = 0; r < num_routes; ++r) {
        signal_all(done_sem, r);
    }
    tt::tt_fabric::common::experimental::close_connections(fabric_connection);
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
