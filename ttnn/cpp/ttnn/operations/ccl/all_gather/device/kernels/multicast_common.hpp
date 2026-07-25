// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

#ifdef FABRIC_2D
#include "tt_metal/fabric/hw/inc/mesh/api.h"
namespace fabric_api = tt::tt_fabric::mesh::experimental;
using FabricRange = tt::tt_fabric::mesh::experimental::MeshMcastRange;
#else
#include "tt_metal/fabric/hw/inc/linear/api.h"
namespace fabric_api = tt::tt_fabric::linear::experimental;
using FabricRange = uint8_t;  // under 1D each connection carries a single hop count
#endif

// Helper class to send pages to remote device.
// Deals with how to packetize pages and interact with Fabric APIs.
template <uint32_t page_size, uint32_t packet_size, bool alternate_routes>
class FabricWriter {
public:
    FabricWriter(
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
        use_route_1{true},
        scatter_header({}, {}),
        chunk_count{0} {
        std::array<uint64_t, max_pages_per_packet> dummy_addrs{};  // init to 0s
        std::array<uint16_t, max_pages_per_packet - 1> chunk_sizes{};
        chunk_sizes.fill(page_size);
        uint8_t starts[1] = {1};

        fabric_api::fabric_multicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::ChunkSizes>(
            fabric_connection,
            scatter_route_id_1,
#ifndef FABRIC_2D
            starts,
#endif
            ranges,
            NocUnicastScatterCommandHeader(dummy_addrs.data(), chunk_sizes.data(), pages_per_packet));

        fabric_api::fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
            fabric_connection,
            unicast_route_id_1,
#ifndef FABRIC_2D
            starts,
#endif
            ranges);

        // Ring topology: create a second route to alternate with for load balancing.
        // Example for 8 device ring:
        //    forward worker alternates between 4 hops and 3 hops (in that order).
        //    backward worker alternates between 3 hops and 4 hops (in that order).
        if constexpr (alternate_routes) {
            fabric_api::fabric_multicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::ChunkSizes>(
                fabric_connection,
                scatter_route_id_2,
#ifndef FABRIC_2D
                starts,
#endif
                ranges_alt,
                NocUnicastScatterCommandHeader(dummy_addrs.data(), chunk_sizes.data(), pages_per_packet));

            fabric_api::fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
                fabric_connection,
                unicast_route_id_2,
#ifndef FABRIC_2D
                starts,
#endif
                ranges_alt);
        }
    }

    ~FabricWriter() {
        ASSERT(chunk_count == 0);  // outstanding chunks! flush() not called correctly
    }

    // Send a single page
    void async_write(uint32_t l1_addr, uint64_t remote_noc_addr) {
        if constexpr (use_scatter_write) {
            // Queue up multiple pages to send in a single packet.
            // Assumption: pages are contiguous in local memory (L1).
            // Note: currently, scatter_write necessitates chunk_count >= 2.
            if (chunk_count == 0) {
                start_l1_addr = l1_addr;
            }
            scatter_header.noc_address[chunk_count++] = remote_noc_addr;

            if (chunk_count == pages_per_packet) {
                noc.async_writes_flushed();
                scatter_header.chunk_count = chunk_count;
                fabric_api::fabric_multicast_noc_scatter_write_with_state<
                    UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::PayloadSize>(
                    fabric_connection,
                    use_route_1 ? scatter_route_id_1 : scatter_route_id_2,
                    start_l1_addr,
                    scatter_header,
                    payload_size);

                chunk_count = 0;
                if constexpr (alternate_routes) {
                    use_route_1 = !use_route_1;  // alternate between routes for load balancing
                }
            }
        } else {
            // Send a single page using multiple packets.
            for (uint32_t packet = 0; packet < packets_per_page; ++packet) {
                noc.async_writes_flushed();
                fabric_api::fabric_multicast_noc_unicast_write_with_state<
                    UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                    fabric_connection,
                    use_route_1 ? unicast_route_id_1 : unicast_route_id_2,
                    l1_addr,
                    tt::tt_fabric::NocUnicastCommandHeader{remote_noc_addr},
                    (packet < packets_per_page - 1) ? payload_size : last_payload_size);
                l1_addr += payload_size;
                remote_noc_addr += payload_size;
                if constexpr (alternate_routes) {
                    use_route_1 = !use_route_1;  // alternate between routes for load balancing
                }
            }
        }
    }

    // Call this before popping CB entry
    void async_writes_flushed() {
        if constexpr (use_scatter_write) {
            // Send any outstanding pages. This can happen when total number of tensor pages doesn't evenly
            // divide by pages per packet.
            static_assert(min_pages_per_packet == 2, "hardcoded to assume scatter_write min_pages_per_packet == 2");
            if (chunk_count > 0) {
                noc.async_writes_flushed();
                if (chunk_count == 1) {
                    // Note: currently, scatter_write necessitates chunk_count >= 2, so we use unicast_write
                    // for chunk_count == 1.
                    // Note: this is hardcoded assuming NOC_SCATTER_WRITE_MIN_CHUNKS == 2. Else need to put
                    // the below unicast_write in a loop.
                    fabric_api::fabric_multicast_noc_unicast_write_with_state<
                        UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                        fabric_connection,
                        use_route_1 ? unicast_route_id_1 : unicast_route_id_2,
                        start_l1_addr,
                        tt::tt_fabric::NocUnicastCommandHeader{scatter_header.noc_address[0]},
                        page_size);
                } else {
                    scatter_header.chunk_count = chunk_count;
                    fabric_api::fabric_multicast_noc_scatter_write_with_state<
                        UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::PayloadSize>(
                        fabric_connection,
                        use_route_1 ? scatter_route_id_1 : scatter_route_id_2,
                        start_l1_addr,
                        scatter_header,
                        chunk_count * page_size);
                }
                chunk_count = 0;
                if constexpr (alternate_routes) {
                    use_route_1 = !use_route_1;  // alternate between routes for load balancing
                }
            }
        } else {
            // no remaining data to send here
        }
        // Wait for Fabric writes to be sent out before popping CB entry
        noc.async_writes_flushed();
    }

private:
    // Fabric limits
    static constexpr uint32_t max_pages_per_packet = NOC_SCATTER_WRITE_MAX_CHUNKS;
    static constexpr uint32_t min_pages_per_packet = NOC_SCATTER_WRITE_MIN_CHUNKS;
    // When page_size < packet_size
    static constexpr uint32_t pages_per_packet = std::min(packet_size / page_size, max_pages_per_packet);  // div_down
    // When page_size > packet_size
    static constexpr uint32_t packets_per_page = (page_size + packet_size - 1) / packet_size;  // div_up
    // Use scatter_write or unicast_write (currently scatter_write imposes a min chunk_count)
    static constexpr bool use_scatter_write = pages_per_packet >= min_pages_per_packet;
    // Steady-state payload size. Note (pages_per_packet * page_size) may not equal packet_size.
    static constexpr uint32_t payload_size = use_scatter_write ? (pages_per_packet * page_size) : packet_size;
    // Last payload for the page_size >= packet_size case (a page sent as multiple packets).
    static constexpr uint32_t last_payload_size = page_size - ((packets_per_page - 1) * packet_size);

    const Noc& noc;
    tt::tt_fabric::RoutingPlaneConnectionManager& fabric_connection;
    uint8_t scatter_route_id_1;
    uint8_t scatter_route_id_2;
    uint8_t unicast_route_id_1;
    uint8_t unicast_route_id_2;
    bool use_route_1;  // toggle to alternate between route_1 and route_2
    NocUnicastScatterCommandHeader scatter_header;
    uint8_t chunk_count;     // accumulated chunks not yet sent in a packet
    uint32_t start_l1_addr;  // start address of the accumulated contiguous chunks
};
