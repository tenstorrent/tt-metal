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
        scatter_header({}, {}) {
        scatter_header.chunk_count = 0;
        const auto default_scatter_header = [] {
            if constexpr (pages_per_packet == 2) {
                return NocUnicastScatterCommandHeader({0, 0}, {static_cast<uint16_t>(page_size)});
            } else if constexpr (pages_per_packet == 3) {
                return NocUnicastScatterCommandHeader(
                    {0, 0, 0}, {static_cast<uint16_t>(page_size), static_cast<uint16_t>(page_size)});
            } else {
                return NocUnicastScatterCommandHeader(
                    {0, 0, 0, 0},
                    {static_cast<uint16_t>(page_size),
                     static_cast<uint16_t>(page_size),
                     static_cast<uint16_t>(page_size)});
            }
        }();

        // PacketHeaderPool::allocate_header_n (vs allocate_header) allows sending the same packet along multiple
        // paths in a single API invocation
        std::array starts = {static_cast<uint8_t>(1)};
        std::array ranges_1 = {range_hops_1};
        fabric_multicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            fabric_connection,
            route_id_1,
            starts.data(),
            ranges_1.data(),
            default_scatter_header,
            pages_per_packet * page_size);

        // Ring topology: create a second route to alternate with for load balancing.
        // Example for 8 device ring:
        //    route_1 = 4 devices forward and 3 devices backward
        //    route_2 = 3 devices forward and 4 devices backward
        if constexpr (load_balance_across_two_routes) {
            std::array ranges_2 = {range_hops_2};
            fabric_multicast_noc_scatter_write_set_state<
                UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
                fabric_connection,
                route_id_2,
                starts.data(),
                ranges_2.data(),
                default_scatter_header,
                pages_per_packet * page_size);
        }
    }

    void send(uint32_t cb_out_page_start, uint64_t tensor_page_addr) {
        if (scatter_header.chunk_count == 0) {
            // save the address of the first chunk in packet
            // write function will start reading here
            packet_data_read_ptr = cb_out_page_start;
        }
        scatter_header.noc_address[scatter_header.chunk_count++] = tensor_page_addr;

        if (scatter_header.chunk_count == pages_per_packet) {
            noc.async_writes_flushed();
            // TODO for variable pages_per_packet, add UnicastScatterWriteUpdateMask::PayloadSize and payload_size as
            // last arg
            fabric_multicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                fabric_connection, use_route_1 ? route_id_1 : route_id_2, packet_data_read_ptr, scatter_header);

            scatter_header.chunk_count = 0;
            if constexpr (load_balance_across_two_routes) {
                use_route_1 = !use_route_1;  // alternate between routes for load balancing
            }
        }
    }

private:
    const experimental::Noc& noc;
    tt::tt_fabric::RoutingPlaneConnectionManager& fabric_connection;
    uint8_t route_id_1;
    uint8_t route_id_2;
    bool use_route_1;
    uint32_t packet_data_read_ptr;
    NocUnicastScatterCommandHeader scatter_header;
};
