// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

#include "gather_walk.hpp"

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

// Multicasts segments to remote devices along the routes it is given.
//
// A segment is a run of chunks contiguous at the destination. Packs segments into a packet until either
// the payload or the scatter-chunk count runs out, and sends the one-segment case as a unicast write --
// which costs each receiving ERISC a single NoC command instead of one per segment.
template <uint32_t chunk_size, uint32_t packet_size, bool alternate_routes>
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
        chunk_count{0},
        payload{0} {
        uint8_t starts[1] = {1};

        // Addresses and sizes both vary per packet, so set_state only fixes the route.
        fabric_api::fabric_multicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::None>(
            fabric_connection,
            scatter_route_id_1,
#ifndef FABRIC_2D
            starts,
#endif
            ranges);

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
            fabric_api::fabric_multicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::None>(
                fabric_connection,
                scatter_route_id_2,
#ifndef FABRIC_2D
                starts,
#endif
                ranges_alt);

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
        ASSERT(chunk_count == 0);  // outstanding segments! flush_packet_and_wait() not called correctly
    }

    // A segment that does not fit starts a new packet rather than spilling into this one: splitting it
    // would fill the tail but cost an extra scatter chunk, i.e. an extra NoC write at each receiver.
    FORCE_INLINE void queue_segment(uint32_t l1_addr, uint64_t remote_noc_addr, uint32_t bytes) {
        if constexpr (oversized) {
            ASSERT(bytes == chunk_size);  // a chunk this big always walks as a run of one
            for (uint32_t packet = 0; packet < packets_per_chunk; ++packet) {
                send_unicast(
                    l1_addr, remote_noc_addr, (packet < packets_per_chunk - 1) ? packet_size : last_payload_size);
                l1_addr += packet_size;
                remote_noc_addr += packet_size;
            }
        } else {
            if (chunk_count == max_chunks || payload + bytes > packet_size) {
                send();
            }
            push(l1_addr, remote_noc_addr, bytes);
        }
    }

    // Call this before popping a CB entry: a queued packet still points into it.
    void flush_packet_and_wait() {
        if constexpr (!oversized) {
            send();
        }
        noc.async_writes_flushed();
    }

private:
    static constexpr uint32_t max_chunks = NOC_SCATTER_WRITE_MAX_CHUNKS;
    // A chunk larger than a packet cannot be accumulated, so it is split across packets instead.
    static constexpr bool oversized = chunk_size > packet_size;
    static constexpr uint32_t packets_per_chunk = (chunk_size + packet_size - 1) / packet_size;  // div_up
    static constexpr uint32_t last_payload_size = chunk_size - ((packets_per_chunk - 1) * packet_size);
    static_assert(packet_size <= 0xFFFF, "NocUnicastScatterCommandHeader::chunk_size is uint16_t");
    static_assert(NOC_SCATTER_WRITE_MIN_CHUNKS == 2, "send() covers the too-few-chunks case with one unicast write");

    FORCE_INLINE void push(uint32_t l1_addr, uint64_t remote_noc_addr, uint32_t bytes) {
        if (chunk_count == 0) {
            start_l1_addr = l1_addr;
        }
        // Only the first max_chunks-1 sizes travel; the last one is implied by the payload size.
        if (chunk_count < max_chunks - 1) {
            scatter_header.chunk_size[chunk_count] = static_cast<uint16_t>(bytes);
        }
        scatter_header.noc_address[chunk_count++] = remote_noc_addr;
        payload += bytes;
    }

    void send() {
        if (chunk_count == 0) {
            return;
        }
        if (chunk_count == 1) {
            send_unicast(start_l1_addr, scatter_header.noc_address[0], payload);
        } else {
            noc.async_writes_flushed();
            scatter_header.chunk_count = chunk_count;
            fabric_api::fabric_multicast_noc_scatter_write_with_state<
                UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                UnicastScatterWriteUpdateMask::PayloadSize>(
                fabric_connection,
                use_route_1 ? scatter_route_id_1 : scatter_route_id_2,
                start_l1_addr,
                scatter_header,
                payload);
            next_route();
        }
        chunk_count = 0;
        payload = 0;
    }

    void send_unicast(uint32_t l1_addr, uint64_t remote_noc_addr, uint32_t bytes) {
        noc.async_writes_flushed();
        fabric_api::fabric_multicast_noc_unicast_write_with_state<
            UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
            fabric_connection,
            use_route_1 ? unicast_route_id_1 : unicast_route_id_2,
            l1_addr,
            tt::tt_fabric::NocUnicastCommandHeader{remote_noc_addr},
            bytes);
        next_route();
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
    NocUnicastScatterCommandHeader scatter_header;
    uint8_t chunk_count;     // segments queued for the current packet
    uint32_t payload;        // bytes queued for the current packet
    uint32_t start_l1_addr;  // start of the queued segments, contiguous in L1
};
