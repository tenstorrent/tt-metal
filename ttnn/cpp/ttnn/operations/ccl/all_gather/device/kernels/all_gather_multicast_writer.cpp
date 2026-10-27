// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// all_gather multicast writer: the slot consumer. Each packet goes once over every connection (one per direction
// with a neighbor), and together the connections reach every other device of the collective once. Init and done ride
// the same routes, so every device counts one of each from every other device, and done lands after our data.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/api_common.h"

#ifdef FABRIC_2D
#include "tt_metal/fabric/hw/inc/mesh/api.h"
namespace fabric_api = tt::tt_fabric::mesh::experimental;
#else
#include "tt_metal/fabric/hw/inc/linear/api.h"
namespace fabric_api = tt::tt_fabric::linear::experimental;
#endif

#include "ttnn/operations/ccl/all_gather/shared_with_host/all_gather_walk.hpp"
#include "all_gather_packet.hpp"
#include "all_gather_sync.hpp"

namespace {

// Directions a device can have neighbors in (E, W, N, S).
constexpr uint32_t kMaxConnections = 4;

// Each connection's route: hop counts packed one byte per physical direction (E, W, N, S).
struct Routes {
#ifdef FABRIC_2D
    tt::tt_fabric::mesh::experimental::MeshMcastRange range[kMaxConnections];
    void set(uint32_t c, uint32_t packed) {
        range[c] = {
            static_cast<uint8_t>(packed),
            static_cast<uint8_t>(packed >> 8),
            static_cast<uint8_t>(packed >> 16),
            static_cast<uint8_t>(packed >> 24)};
    }
#else
    // 1D: a connection runs along one direction, so only one byte is set; start one hop out.
    uint8_t start[kMaxConnections];
    uint8_t range[kMaxConnections];
    void set(uint32_t c, uint32_t packed) {
        start[c] = 1;
        range[c] = static_cast<uint8_t>(packed) + static_cast<uint8_t>(packed >> 8) +
                   static_cast<uint8_t>(packed >> 16) + static_cast<uint8_t>(packed >> 24);
    }
#endif
};

template <UnicastAtomicIncUpdateMask Mask>
FORCE_INLINE void atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& mgr, uint8_t route, Routes& r, uint64_t addr) {
    const tt::tt_fabric::NocUnicastAtomicIncCommandHeader cmd{addr, 1};
#ifdef FABRIC_2D
    fabric_api::fabric_multicast_noc_unicast_atomic_inc_set_state<Mask>(mgr, route, r.range, cmd);
#else
    fabric_api::fabric_multicast_noc_unicast_atomic_inc_set_state<Mask>(mgr, route, r.start, r.range, cmd);
#endif
}

FORCE_INLINE void unicast_write_set_state(tt::tt_fabric::RoutingPlaneConnectionManager& mgr, uint8_t route, Routes& r) {
#ifdef FABRIC_2D
    fabric_api::fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(mgr, route, r.range);
#else
    fabric_api::fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
        mgr, route, r.start, r.range);
#endif
}

FORCE_INLINE void scatter_write_set_state(tt::tt_fabric::RoutingPlaneConnectionManager& mgr, uint8_t route, Routes& r) {
#ifdef FABRIC_2D
    fabric_api::fabric_multicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::None>(mgr, route, r.range);
#else
    fabric_api::fabric_multicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::None>(
        mgr, route, r.start, r.range);
#endif
}

}  // namespace

void kernel_main() {
    constexpr uint32_t unit_bytes = get_named_compile_time_arg_val("unit_bytes");
    constexpr uint32_t packet_cap = get_named_compile_time_arg_val("packet_cap");
    constexpr uint32_t packets_per_slot = get_named_compile_time_arg_val("packets_per_slot");
    constexpr uint32_t slot_bytes = get_named_compile_time_arg_val("slot_bytes");
    constexpr uint32_t slot_cb = get_named_compile_time_arg_val("slot_cb");

    constexpr ag::SlotLayout lay =
        ag::slot_offsets(packets_per_slot, unit_bytes, sizeof(PACKET_HEADER_TYPE), packet_cap);
    static_assert(lay.slot_bytes == slot_bytes, "host and kernel disagree on the packet header size");

    size_t arg_idx = 0;
    const uint32_t take = get_arg_val<uint32_t>(arg_idx++);
    const bool init_send = get_arg_val<uint32_t>(arg_idx++) != 0;
    const uint32_t init_wait = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t init_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t done_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mirror_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mirror_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_connections = get_arg_val<uint32_t>(arg_idx++);
    ASSERT(num_connections >= 1 && num_connections <= kMaxConnections);
    Routes routes;
    for (uint32_t c = 0; c < num_connections; ++c) {
        routes.set(c, get_arg_val<uint32_t>(arg_idx++));
    }

    Noc noc;
    CircularBuffer cb(slot_cb);
    tt::tt_fabric::RoutingPlaneConnectionManager mgr;
    tt::tt_fabric::common::experimental::open_connections(mgr, num_connections, arg_idx);

    // One unicast-write route (also carries init and done) and one scatter-write route, each a header per
    // connection. Headers are reused, so every re-encode waits for the previous send to leave L1.
    const uint8_t unicast_route = PacketHeaderPool::allocate_header_n(num_connections);
    const uint8_t scatter_route = PacketHeaderPool::allocate_header_n(num_connections);
    // Sends one credit to the same core on every other device.
    auto signal_all = [&](uint32_t sem) {
        atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            mgr, unicast_route, routes, ag::fabric_frame_addr(mirror_x, mirror_y, sem));
        fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::None>(
            mgr, unicast_route);
        noc.async_writes_flushed();
    };

    if (init_send) {
        signal_all(init_sem);
    }
    unicast_write_set_state(mgr, unicast_route, routes);
    scatter_write_set_state(mgr, scatter_route, routes);
    // Our data goes into every other device's output: wait until all are alive (each sent its init the same way).
    // Our reader fills the slots and does the local copy meanwhile.
    ag::wait_and_consume(init_sem, init_wait);

    for (bool iter_end = take == 0; !iter_end;) {
        cb.wait_front(1);
        const uint32_t slot = cb.get_read_ptr();
        ag::slot_acquire();
        const ag::SlotInfo info = *reinterpret_cast<const ag::SlotInfo*>(slot);
        for (uint32_t j = 0; j < info.packets; ++j) {
            const auto& rec = *reinterpret_cast<const ag::PacketRecord*>(lay.record(slot, j));
            const uint32_t payload = lay.payload(slot, j);
            noc.async_writes_flushed();
            if (rec.cmd.chunk_count == 1) {
                fabric_api::fabric_multicast_noc_unicast_write_with_state<
                    UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                    mgr,
                    unicast_route,
                    payload,
                    tt::tt_fabric::NocUnicastCommandHeader{rec.cmd.noc_address[0]},
                    rec.payload_bytes);
            } else {
                fabric_api::fabric_multicast_noc_scatter_write_with_state<
                    UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                    UnicastScatterWriteUpdateMask::PayloadSize>(
                    mgr, scatter_route, payload, rec.cmd, rec.payload_bytes);
            }
        }
        noc.async_writes_flushed();
        cb.pop_front(1);
        iter_end = info.iter_end != 0;
    }

    // After every data packet on the same routes, so each receiver's done count implies our data has landed. Sent
    // even with nothing to send: every reader waits for all of them.
    signal_all(done_sem);
    tt::tt_fabric::common::experimental::close_connections(mgr);
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
