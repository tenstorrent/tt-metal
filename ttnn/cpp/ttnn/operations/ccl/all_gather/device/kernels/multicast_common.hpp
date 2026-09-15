// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

#ifdef FABRIC_2D
#include "tt_metal/fabric/hw/inc/mesh/api.h"
namespace fabric_api = tt::tt_fabric::mesh::experimental;
using FabricRange = tt::tt_fabric::mesh::experimental::MeshMcastRange;
inline FabricRange make_fabric_range(uint8_t e, uint8_t w, uint8_t n, uint8_t s) { return FabricRange{e, w, n, s}; }
#else
#include "tt_metal/fabric/hw/inc/linear/api.h"
namespace fabric_api = tt::tt_fabric::linear::experimental;
using FabricRange = uint8_t;  // under 1D each connection carries a single hop count
// 1D has a single active direction, so exactly one slot is nonzero
inline FabricRange make_fabric_range(uint8_t e, uint8_t w, uint8_t n, uint8_t s) { return e + w + n + s; }
#endif

#include "chunk_packets.hpp"

////////////////////////////////////////////////////////////////
// Multicast all_gather: the fabric copies each packet onward, so a device sends its own stripe once
// and never relays anybody else's.
//
// Glossary:
//   route    -- one multicast range: how many hops, in which direction.
//   alt route -- a second range to alternate with, so an even ring splits its load between the two
//                arcs. Forward alternates 4 hops then 3; backward 3 then 4.
////////////////////////////////////////////////////////////////

// Sends to every device along the routes it is given. Gives Packer its two writes, and owns the
// routes and headers behind them.
template <bool alternate_routes>
class MulticastSender {
public:
    MulticastSender(
        tt::tt_fabric::RoutingPlaneConnectionManager& manager,
        uint32_t num_connections,
        FabricRange* ranges,
        FabricRange* ranges_alt = nullptr) :
        fabric_connection{manager},
        // allocate_header_n (vs allocate_header) allows sending the same packet along multiple paths
        // in a single API invocation.
        scatter_route_1{PacketHeaderPool::allocate_header_n(num_connections)},
        scatter_route_2{alternate_routes ? PacketHeaderPool::allocate_header_n(num_connections) : scatter_route_1},
        unicast_route_1{PacketHeaderPool::allocate_header_n(num_connections)},
        unicast_route_2{alternate_routes ? PacketHeaderPool::allocate_header_n(num_connections) : unicast_route_1},
        use_route_1{true} {
        uint8_t starts[1] = {1};

        // Addresses and sizes both vary per packet, so set_state only fixes the route.
        fabric_api::fabric_multicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::None>(
            fabric_connection,
            scatter_route_1,
#ifndef FABRIC_2D
            starts,
#endif
            ranges);

        fabric_api::fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
            fabric_connection,
            unicast_route_1,
#ifndef FABRIC_2D
            starts,
#endif
            ranges);

        if constexpr (alternate_routes) {
            fabric_api::fabric_multicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::None>(
                fabric_connection,
                scatter_route_2,
#ifndef FABRIC_2D
                starts,
#endif
                ranges_alt);

            fabric_api::fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
                fabric_connection,
                unicast_route_2,
#ifndef FABRIC_2D
                starts,
#endif
                ranges_alt);
        }
    }

    FORCE_INLINE void write_one(uint32_t l1_addr, uint64_t dst, uint32_t bytes) {
        fabric_api::fabric_multicast_noc_unicast_write_with_state<
            UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
            fabric_connection,
            use_route_1 ? unicast_route_1 : unicast_route_2,
            l1_addr,
            tt::tt_fabric::NocUnicastCommandHeader{dst},
            bytes);
        next_route();
    }

    FORCE_INLINE void write_scatter(uint32_t l1_addr, NocUnicastScatterCommandHeader& header, uint32_t payload) {
        fabric_api::fabric_multicast_noc_scatter_write_with_state<
            UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
            UnicastScatterWriteUpdateMask::PayloadSize>(
            fabric_connection, use_route_1 ? scatter_route_1 : scatter_route_2, l1_addr, header, payload);
        next_route();
    }

private:
    FORCE_INLINE void next_route() {
        if constexpr (alternate_routes) {
            use_route_1 = !use_route_1;  // alternate between routes for load balancing
        }
    }

    tt::tt_fabric::RoutingPlaneConnectionManager& fabric_connection;
    uint8_t scatter_route_1;
    uint8_t scatter_route_2;
    uint8_t unicast_route_1;
    uint8_t unicast_route_2;
    bool use_route_1;  // toggle to alternate between route_1 and route_2
};
