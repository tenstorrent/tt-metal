// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Device-only: the fabric ends of PacketPacker (gather_packer.hpp), one per all_gather algorithm.
//   HopSink<SenderT>          -- store-and-forward: every send is a 1-hop unicast to the neighbor.
//   McastSink<alternate>      -- multicast along the routes of a RoutingPlaneConnectionManager.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"

#include <cstdint>
#include <type_traits>

#include "gather_packer.hpp"

static_assert(gather_scatter_max_chunks == NOC_SCATTER_WRITE_MAX_CHUNKS, "gather_rules.hpp is out of date");
static_assert(gather_scatter_min_chunks == NOC_SCATTER_WRITE_MIN_CHUNKS, "gather_rules.hpp is out of date");

// The store-and-forward path always speaks the linear API (it runs on any effectively-1D topology, both
// Fabric 1D and 2D); multicast picks the API by FABRIC_2D.
namespace gather_hop_api = tt::tt_fabric::linear::experimental;

#ifdef FABRIC_2D
#include "tt_metal/fabric/hw/inc/mesh/api.h"
namespace gather_mcast_api = tt::tt_fabric::mesh::experimental;
using FabricRange = tt::tt_fabric::mesh::experimental::MeshMcastRange;
inline FabricRange make_fabric_range(uint8_t e, uint8_t w, uint8_t n, uint8_t s) { return FabricRange{e, w, n, s}; }
#else
namespace gather_mcast_api = tt::tt_fabric::linear::experimental;
using FabricRange = uint8_t;  // under 1D each connection carries a single hop count
// 1D has a single active direction, so exactly one slot is nonzero
inline FabricRange make_fabric_range(uint8_t e, uint8_t w, uint8_t n, uint8_t s) { return e + w + n + s; }
#endif

////////////////////////////////////////////////////////////////
// HopSink
////////////////////////////////////////////////////////////////

// Templated on the sender type (SenderT*) so the same writer drives either a direct WorkerToFabricEdmSender
// (one worker per direction) or a FabricMuxV2Sender (workers sharing a fabric mux). The send calls accept
// either (see CheckFabricSenderType in api_common.h), so no route-manager is needed -- which is also why this
// class routes its own headers.
template <typename SenderT>
class HopSink {
public:
    HopSink(const Noc& noc, SenderT* sender, uint16_t neighbor_chip_id, uint16_t neighbor_mesh_id) :
        noc{noc},
        sender{sender},
        scatter_packet_header{PacketHeaderPool::allocate_header(1)},
        unicast_packet_header{PacketHeaderPool::allocate_header(1)},
        sem_packet_header{PacketHeaderPool::allocate_header(1)} {
        constexpr uint8_t num_hops = 1;  // store-and-forward: always the immediate neighbor

        // Addresses and sizes both vary per packet, so set_state only fixes the route.
        gather_hop_api::fabric_unicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::None>(
            scatter_packet_header, num_hops);

        gather_hop_api::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
            unicast_packet_header, num_hops);

        // One atomic-inc header for the "alive" barrier inc + data_valid signals; Flush orders it after the
        // payload it announces.
        gather_hop_api::fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            sem_packet_header, num_hops, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0u, 1u});

        // For Fabric_2D, set_state() sets routes only in its RoutingPlaneConnectionManager overloads, so set
        // them here. Keyed on the header type since the FABRIC_2D define is absent on the mux path.
        if constexpr (std::is_base_of_v<tt::tt_fabric::HybridMeshPacketHeader, PACKET_HEADER_TYPE>) {
            using MeshHeader = volatile tt::tt_fabric::HybridMeshPacketHeader*;
            fabric_set_unicast_route(
                reinterpret_cast<MeshHeader>(scatter_packet_header), neighbor_chip_id, neighbor_mesh_id);
            fabric_set_unicast_route(
                reinterpret_cast<MeshHeader>(unicast_packet_header), neighbor_chip_id, neighbor_mesh_id);
            fabric_set_unicast_route(
                reinterpret_cast<MeshHeader>(sem_packet_header), neighbor_chip_id, neighbor_mesh_id);
        }
    }

    // Increment a semaphore on the neighbor.
    void atomic_inc(uint64_t addr, uint32_t val) {
        gather_hop_api::fabric_unicast_noc_unicast_atomic_inc_with_state<
            UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val>(
            sender, sem_packet_header, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{addr, val});
    }

    // PacketPacker sink.
    using ScatterSlots = NocUnicastScatterCommandHeader;
    FORCE_INLINE void unicast_write(uint32_t l1_addr, uint64_t dst, uint32_t bytes) {
        noc.async_writes_flushed();
        gather_hop_api::fabric_unicast_noc_unicast_write_with_state<
            UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
            sender, unicast_packet_header, l1_addr, tt::tt_fabric::NocUnicastCommandHeader{dst}, bytes);
    }

    FORCE_INLINE void scatter_write(uint32_t l1_addr, const ScatterSlots& slots, uint32_t payload) {
        noc.async_writes_flushed();
        gather_hop_api::fabric_unicast_noc_scatter_write_with_state<
            UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
            UnicastScatterWriteUpdateMask::PayloadSize>(sender, scatter_packet_header, l1_addr, slots, payload);
    }

    FORCE_INLINE void flush() { noc.async_writes_flushed(); }

private:
    const Noc& noc;
    SenderT* sender;  // direct or mux sender
    volatile tt_l1_ptr PACKET_HEADER_TYPE* scatter_packet_header;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* unicast_packet_header;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* sem_packet_header;
};

////////////////////////////////////////////////////////////////
// McastSink
////////////////////////////////////////////////////////////////

// The 12 runtime args that describe a worker's multicast routes, in host order: line then rect, each
// as (hops, hops_alt, dir).
struct McastRouteArgs {
    uint8_t line_hops, rect_e_hops, rect_w_hops, rect_spine_hops;
    uint8_t line_hops_alt, rect_e_hops_alt, rect_w_hops_alt, rect_spine_hops_alt;
    uint8_t line_dir, rect_e_dir, rect_w_dir, rect_spine_dir;

    static FORCE_INLINE McastRouteArgs from_args(size_t& arg_idx) {
        McastRouteArgs a;
        a.line_hops = get_arg_val<uint32_t>(arg_idx++);
        a.rect_e_hops = get_arg_val<uint32_t>(arg_idx++);
        a.rect_w_hops = get_arg_val<uint32_t>(arg_idx++);
        a.rect_spine_hops = get_arg_val<uint32_t>(arg_idx++);
        a.line_hops_alt = get_arg_val<uint32_t>(arg_idx++);
        a.rect_e_hops_alt = get_arg_val<uint32_t>(arg_idx++);
        a.rect_w_hops_alt = get_arg_val<uint32_t>(arg_idx++);
        a.rect_spine_hops_alt = get_arg_val<uint32_t>(arg_idx++);
        a.line_dir = get_arg_val<uint32_t>(arg_idx++);
        a.rect_e_dir = get_arg_val<uint32_t>(arg_idx++);
        a.rect_w_dir = get_arg_val<uint32_t>(arg_idx++);
        a.rect_spine_dir = get_arg_val<uint32_t>(arg_idx++);
        return a;
    }
};

// Build the line then rect ranges, in connection order (matches the host). Each hop count is placed at
// its physical slot; an absent branch (hop == 0) is skipped so it can't clobber a live slot.
FORCE_INLINE void build_mcast_ranges(const McastRouteArgs& a, FabricRange (&ranges)[2], FabricRange (&ranges_alt)[2]) {
    uint32_t conn = 0;
    if (a.line_hops > 0) {
        uint8_t hops[4] = {}, hops_alt[4] = {};
        hops[a.line_dir] = a.line_hops;
        hops_alt[a.line_dir] = a.line_hops_alt;
        ranges[conn] = make_fabric_range(hops[0], hops[1], hops[2], hops[3]);
        ranges_alt[conn] = make_fabric_range(hops_alt[0], hops_alt[1], hops_alt[2], hops_alt[3]);
        ++conn;
    }
    if (a.rect_spine_hops > 0) {
        uint8_t hops[4] = {}, hops_alt[4] = {};
        if (a.rect_e_hops > 0) {
            hops[a.rect_e_dir] = a.rect_e_hops;
            hops_alt[a.rect_e_dir] = a.rect_e_hops_alt;
        }
        if (a.rect_w_hops > 0) {
            hops[a.rect_w_dir] = a.rect_w_hops;
            hops_alt[a.rect_w_dir] = a.rect_w_hops_alt;
        }
        hops[a.rect_spine_dir] = a.rect_spine_hops;
        hops_alt[a.rect_spine_dir] = a.rect_spine_hops_alt;
        ranges[conn] = make_fabric_range(hops[0], hops[1], hops[2], hops[3]);
        ranges_alt[conn] = make_fabric_range(hops_alt[0], hops_alt[1], hops_alt[2], hops_alt[3]);
        ++conn;
    }
}

// Semaphore route: one header per connection, atomic inc of 1, flushed so it lands after the payload it
// follows. Allocate it after the data routes (McastSink), as before.
FORCE_INLINE uint8_t
mcast_sem_route(tt::tt_fabric::RoutingPlaneConnectionManager& manager, uint32_t num_connections, FabricRange* ranges) {
    const uint8_t route_id = PacketHeaderPool::allocate_header_n(num_connections);
    [[maybe_unused]] uint8_t starts[1] = {1};
    gather_mcast_api::fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        manager,
        route_id,
#ifndef FABRIC_2D
        starts,
#endif
        ranges,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0u,    // ignore
            1u});  // increment 1
    return route_id;
}

// Increment the barrier semaphore at (x, y, sem) on every device the sem route reaches.
FORCE_INLINE void mcast_sem_inc(
    tt::tt_fabric::RoutingPlaneConnectionManager& manager, uint8_t route_id, uint8_t x, uint8_t y, uint32_t sem) {
    const uint64_t sem_noc_addr_in_pkt = safe_get_noc_addr(x, y, sem, 0);
    gather_mcast_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        manager, route_id, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc_addr_in_pkt, 0});
}

template <bool alternate_routes>
class McastSink {
public:
    McastSink(
        const Noc& noc,
        tt::tt_fabric::RoutingPlaneConnectionManager& manager,
        uint32_t num_connections,
        FabricRange* ranges,
        FabricRange* ranges_alt = nullptr) :
        noc{noc},
        fabric_connection{manager},
        // PacketHeaderPool::allocate_header_n (vs allocate_header) allows sending the same packet along multiple
        // paths in a single API invocation.
        scatter_route_id_1{PacketHeaderPool::allocate_header_n(num_connections)},
        scatter_route_id_2{
            alternate_routes ? PacketHeaderPool::allocate_header_n(num_connections) : scatter_route_id_1},
        unicast_route_id_1{PacketHeaderPool::allocate_header_n(num_connections)},
        unicast_route_id_2{
            alternate_routes ? PacketHeaderPool::allocate_header_n(num_connections) : unicast_route_id_1},
        use_route_1{true} {
        set_routes(scatter_route_id_1, unicast_route_id_1, ranges);
        // Ring topology: create a second route to alternate with for load balancing.
        // Example for 8 device ring:
        //    forward worker alternates between 4 hops and 3 hops (in that order).
        //    backward worker alternates between 3 hops and 4 hops (in that order).
        if constexpr (alternate_routes) {
            set_routes(scatter_route_id_2, unicast_route_id_2, ranges_alt);
        }
    }

    // PacketPacker sink.
    using ScatterSlots = NocUnicastScatterCommandHeader;
    FORCE_INLINE void unicast_write(uint32_t l1_addr, uint64_t dst, uint32_t bytes) {
        noc.async_writes_flushed();
        gather_mcast_api::fabric_multicast_noc_unicast_write_with_state<
            UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
            fabric_connection,
            use_route_1 ? unicast_route_id_1 : unicast_route_id_2,
            l1_addr,
            tt::tt_fabric::NocUnicastCommandHeader{dst},
            bytes);
        next_route();
    }

    FORCE_INLINE void scatter_write(uint32_t l1_addr, const ScatterSlots& slots, uint32_t payload) {
        noc.async_writes_flushed();
        gather_mcast_api::fabric_multicast_noc_scatter_write_with_state<
            UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
            UnicastScatterWriteUpdateMask::PayloadSize>(
            fabric_connection, use_route_1 ? scatter_route_id_1 : scatter_route_id_2, l1_addr, slots, payload);
        next_route();
    }

    FORCE_INLINE void flush() { noc.async_writes_flushed(); }

private:
    FORCE_INLINE void set_routes(uint8_t scatter_route_id, uint8_t unicast_route_id, FabricRange* ranges) {
        [[maybe_unused]] uint8_t starts[1] = {1};
        // Addresses and sizes both vary per packet, so set_state only fixes the route.
        gather_mcast_api::fabric_multicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::None>(
            fabric_connection,
            scatter_route_id,
#ifndef FABRIC_2D
            starts,
#endif
            ranges);
        gather_mcast_api::fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
            fabric_connection,
            unicast_route_id,
#ifndef FABRIC_2D
            starts,
#endif
            ranges);
    }

    FORCE_INLINE void next_route() {
        if constexpr (alternate_routes) {
            use_route_1 = !use_route_1;  // alternate between routes for load balancing
        }
    }

    const Noc& noc;
    tt::tt_fabric::RoutingPlaneConnectionManager& fabric_connection;
    uint8_t scatter_route_id_1;
    uint8_t scatter_route_id_2;
    uint8_t unicast_route_id_1;
    uint8_t unicast_route_id_2;
    bool use_route_1;  // toggle to alternate between route_1 and route_2
};
