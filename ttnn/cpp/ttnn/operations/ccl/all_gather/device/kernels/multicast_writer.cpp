// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// all_gather multicast writer: the slot consumer. Each packet goes once over every connection (one per direction
// with a neighbor) and reaches every other device of the collective. On an even ring the two directions are
// unbalanced by one hop, so packets alternate between two route variants. Init and done ride every (connection,
// variant), so every device counts exactly VARIANTS credits from each other device whichever variant its data took.

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

#include "ag_walk.hpp"
#include "ag_packet.hpp"
#include "ag_sync.hpp"

namespace {

// Directions a device can have neighbors in (E, W, N, S).
constexpr uint32_t kMaxConnections = 4;

// A connection's route for one variant: hop counts packed one byte per physical direction (E, W, N, S).
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
    constexpr uint32_t A = get_named_compile_time_arg_val("A");
    constexpr uint32_t P_A = get_named_compile_time_arg_val("P_A");
    constexpr uint32_t K = get_named_compile_time_arg_val("K");
    constexpr uint32_t SLOT_BYTES = get_named_compile_time_arg_val("SLOT_BYTES");
    constexpr uint32_t CB_ID = get_named_compile_time_arg_val("CB_ID");
    constexpr uint32_t VARIANTS = get_named_compile_time_arg_val("VARIANTS");
    constexpr uint32_t WDELAY = get_named_compile_time_arg_val("WDELAY");
    static_assert(VARIANTS == 1 || VARIANTS == 2, "one route, or two alternating on an even ring");

    constexpr ag::SlotLayout lay = ag::slot_offsets(K, A, sizeof(PACKET_HEADER_TYPE), P_A);
    static_assert(lay.slot_bytes == SLOT_BYTES, "host and kernel disagree on the packet header size");

    size_t arg_idx = 0;
    const uint32_t take = get_arg_val<uint32_t>(arg_idx++);
    const bool init_send = get_arg_val<uint32_t>(arg_idx++) != 0;
    const uint32_t init_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t done_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mirror_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mirror_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_connections = get_arg_val<uint32_t>(arg_idx++);
    ASSERT(num_connections >= 1 && num_connections <= kMaxConnections);
    Routes routes[VARIANTS];
    for (uint32_t c = 0; c < num_connections; ++c) {
        const uint32_t v0 = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t v1 = get_arg_val<uint32_t>(arg_idx++);
        routes[0].set(c, v0);
        if constexpr (VARIANTS == 2) {
            routes[1].set(c, v1);
        }
    }

    Noc noc;
    CircularBuffer cb(CB_ID);
    tt::tt_fabric::RoutingPlaneConnectionManager mgr;
    tt::tt_fabric::common::experimental::open_connections(mgr, num_connections, arg_idx);

    // Per variant: one unicast-write route (also carries init and done) and one scatter-write route, each a header
    // per connection. Headers are reused, so every re-encode waits for the previous send to leave L1.
    uint8_t unicast_route[VARIANTS], scatter_route[VARIANTS];
    for (uint32_t v = 0; v < VARIANTS; ++v) {
        unicast_route[v] = PacketHeaderPool::allocate_header_n(num_connections);
        scatter_route[v] = PacketHeaderPool::allocate_header_n(num_connections);
    }
    // Sends one credit to the same core on every other device, once per variant.
    auto signal_all = [&](uint32_t sem) {
        const uint64_t addr = ag::fabric_frame_addr(mirror_x, mirror_y, sem);
        for (uint32_t v = 0; v < VARIANTS; ++v) {
            atomic_inc_set_state<
                UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val |
                UnicastAtomicIncUpdateMask::Flush>(mgr, unicast_route[v], routes[v], addr);
            fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::None>(
                mgr, unicast_route[v]);
        }
        noc.async_writes_flushed();
    };

    if (init_send) {
        signal_all(init_sem);
    }
    for (uint32_t v = 0; v < VARIANTS; ++v) {
        unicast_write_set_state(mgr, unicast_route[v], routes[v]);
        scatter_write_set_state(mgr, scatter_route[v], routes[v]);
    }

    uint32_t v = 0;
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
                    unicast_route[v],
                    payload,
                    tt::tt_fabric::NocUnicastCommandHeader{rec.cmd.noc_address[0]},
                    rec.payload_bytes);
            } else {
                fabric_api::fabric_multicast_noc_scatter_write_with_state<
                    UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                    UnicastScatterWriteUpdateMask::PayloadSize>(
                    mgr, scatter_route[v], payload, rec.cmd, rec.payload_bytes);
            }
            v = (v + 1) % VARIANTS;
        }
        noc.async_writes_flushed();
        cb.pop_front(1);
        iter_end = info.iter_end != 0;
        ag::test_delay<WDELAY>();
    }

    // After every data packet on every route, so each receiver's done count implies our data has landed. Sent even
    // with nothing to send: every reader waits for all of them.
    signal_all(done_sem);
    tt::tt_fabric::common::experimental::close_connections(mgr);
    noc.async_write_barrier();
}
