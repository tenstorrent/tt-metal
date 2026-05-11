// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric::linear::experimental;

// Fabric API wrapper to scatter_write pages to remote device
template <uint32_t page_size, uint32_t pages_per_packet, bool load_balance_across_two_routes>
class FabricScatterWriter {
public:
    FabricScatterWriter(
        const experimental::Noc& noc,
        tt::tt_fabric::RoutingPlaneConnectionManager& manager,
        uint8_t range_hops_1,
        uint8_t range_hops_2,
        uint32_t num_connections) :
        noc{noc},
        fabric_connection{manager},
        route_id_1{PacketHeaderPool::allocate_header_n(num_connections)},
        route_id_2{load_balance_across_two_routes ? PacketHeaderPool::allocate_header_n(num_connections) : route_id_1},
        use_route_1{true},
        scatter_header({}, {}),
        chunk_count(0) {
        // PacketHeaderPool::allocate_header_n (vs allocate_header) allows sending the same packet along multiple
        // paths in a single API invocation
        static_assert(pages_per_packet <= 4, "pages per packet > 4 is unsupported");
        uint64_t dummy_addrs[4] = {0, 0, 0, 0};
        uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
        uint8_t starts[1] = {1};
        uint8_t ranges_1[1] = {range_hops_1};
        fabric_multicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            fabric_connection,
            route_id_1,
            starts,
            ranges_1,
            NocUnicastScatterCommandHeader(dummy_addrs, chunk_sizes, pages_per_packet),
            max_payload_size);

        // Ring topology: create a second route to alternate with for load balancing.
        // Example for 8 device ring:
        //    forward worker alternates between 4 hops and 3 hops (in that order).
        //    backward worker alternates between 3 hops and 4 hops (in that order).
        if constexpr (load_balance_across_two_routes) {
            uint8_t ranges_2[1] = {range_hops_2};
            fabric_multicast_noc_scatter_write_set_state<
                UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
                fabric_connection,
                route_id_2,
                starts,
                ranges_2,
                NocUnicastScatterCommandHeader(dummy_addrs, chunk_sizes, pages_per_packet),
                max_payload_size);
        }
    }

    void send(uint32_t l1_addr, uint64_t remote_noc_addr) {
        if (chunk_count == 0) {
            // Save the address of the first chunk in packet.
            // Fabric write function will start reading contiguous chunks from here.
            start_l1_addr = l1_addr;
        }
        scatter_header.noc_address[chunk_count++] = remote_noc_addr;

        if (chunk_count == pages_per_packet) {
            noc.async_writes_flushed();
            scatter_header.chunk_count = chunk_count;
            fabric_multicast_noc_scatter_write_with_state<
                UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::PayloadSize>(
                fabric_connection,
                use_route_1 ? route_id_1 : route_id_2,
                start_l1_addr,
                scatter_header,
                max_payload_size);

            chunk_count = 0;
            if constexpr (load_balance_across_two_routes) {
                use_route_1 = !use_route_1;  // alternate between routes for load balancing
            }
        }
    }

    // Call this before popping CB entry
    void flush() {
        // Send any outstanding pages. This can happen when total number of tensor pages doesn't evenly
        // divide by pages per packet.
        if (chunk_count > 0) {
            noc.async_writes_flushed();
            scatter_header.chunk_count = chunk_count;
            fabric_multicast_noc_scatter_write_with_state<
                UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::PayloadSize>(
                fabric_connection,
                use_route_1 ? route_id_1 : route_id_2,
                start_l1_addr,
                scatter_header,
                chunk_count * page_size);

            chunk_count = 0;
            if constexpr (load_balance_across_two_routes) {
                use_route_1 = !use_route_1;  // alternate between routes for load balancing
            }
        }
        noc.async_writes_flushed();
    }

private:
    static constexpr uint32_t max_payload_size = pages_per_packet * page_size;

    const experimental::Noc& noc;
    tt::tt_fabric::RoutingPlaneConnectionManager& fabric_connection;
    uint8_t route_id_1;
    uint8_t route_id_2;
    bool use_route_1;
    NocUnicastScatterCommandHeader scatter_header;
    uint8_t chunk_count;
    uint32_t start_l1_addr;
};
