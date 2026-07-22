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

constexpr uint32_t fabric_explicit_path_word_count = 5;

struct FabricExplicitPath {
    uint8_t length;
    uint8_t escape_hop;
    uint32_t words[fabric_explicit_path_word_count];
};

template <bool enabled>
FORCE_INLINE void apply_explicit_fabric_path(uint8_t route_id, const FabricExplicitPath& path) {
#ifdef FABRIC_2D
    if constexpr (enabled) {
        ASSERT(path.length > 0 && path.length <= fabric_explicit_path_word_count * 8);
        ASSERT(PacketHeaderPool::get_num_headers(route_id) == 1);
        PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t) {
            packet_header->routing_fields.value = 0;
            // Zero means that the path remains on VC0. Otherwise this is the
            // one-based hop index at which the receiving router crosses over
            // to the escape VC for the remainder of the route.
            packet_header->mcast_params_64 = path.escape_hop;
            packet_header->is_mcast_active = tt::tt_fabric::RoutingFieldsConstants::Mesh::EXPLICIT_PATH_MCAST;
            for (uint32_t hop = 0; hop < path.length; ++hop) {
                packet_header->route_buffer[hop] = static_cast<uint8_t>((path.words[hop / 8] >> ((hop % 8) * 4)) & 0xF);
            }
        });
    }
#endif
}

// Helper class to send pages to remote device.
// Deals with how to packetize pages and interact with Fabric APIs.
template <uint32_t page_size, uint32_t packet_size, bool alternate_routes, bool explicit_path>
class FabricWriter {
public:
    FabricWriter(
        const Noc& noc,
        tt::tt_fabric::RoutingPlaneConnectionManager& manager,
        uint32_t num_connections,
        FabricRange* ranges,
        FabricRange* ranges_alt,
        const FabricExplicitPath& path,
        const FabricExplicitPath& path_alt) :
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
        chunks_are_contiguous{true} {
        std::array<uint64_t, max_pages_per_packet> dummy_addrs{};  // init to 0s
        std::array<uint16_t, max_pages_per_packet - 1> chunk_sizes{};
        chunk_sizes.fill(page_size);
        uint8_t starts[1] = {1};

        fabric_api::fabric_multicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            fabric_connection,
            scatter_route_id_1,
#ifndef FABRIC_2D
            starts,
#endif
            ranges,
            NocUnicastScatterCommandHeader(dummy_addrs.data(), chunk_sizes.data(), pages_per_packet),
            payload_size);

        fabric_api::fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
            fabric_connection,
            unicast_route_id_1,
#ifndef FABRIC_2D
            starts,
#endif
            ranges);
        apply_explicit_fabric_path<explicit_path>(scatter_route_id_1, path);
        apply_explicit_fabric_path<explicit_path>(unicast_route_id_1, path);

        // Ring topology: create a second route to alternate with for load balancing.
        // Example for 8 device ring:
        //    forward worker alternates between 4 hops and 3 hops (in that order).
        //    backward worker alternates between 3 hops and 4 hops (in that order).
        if constexpr (alternate_routes) {
            fabric_api::fabric_multicast_noc_scatter_write_set_state<
                UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
                fabric_connection,
                scatter_route_id_2,
#ifndef FABRIC_2D
                starts,
#endif
                ranges_alt,
                NocUnicastScatterCommandHeader(dummy_addrs.data(), chunk_sizes.data(), pages_per_packet),
                payload_size);

            fabric_api::fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
                fabric_connection,
                unicast_route_id_2,
#ifndef FABRIC_2D
                starts,
#endif
                ranges_alt);
            apply_explicit_fabric_path<explicit_path>(scatter_route_id_2, path_alt);
            apply_explicit_fabric_path<explicit_path>(unicast_route_id_2, path_alt);
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
                chunks_are_contiguous = true;
            } else {
                // A logical run may also be one physical NOC range (for example concat chunks within one
                // output page). In that case a single unicast write has less command/header work than a
                // scatter packet. Interleaved tensor pages normally resolve to different NOC coordinates and
                // fail this exact-address proof, retaining the generic scatter path.
                chunks_are_contiguous &= remote_noc_addr == scatter_header.noc_address[chunk_count - 1] + page_size;
            }
            scatter_header.noc_address[chunk_count++] = remote_noc_addr;

            if (chunk_count == pages_per_packet) {
                send_queued_chunks();
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

    // Common transport surface used by the compile-time selected sender loop.
    // This overload is unreachable for direct-scatter instantiations.
    void async_write(uint32_t, uint32_t, uint64_t, uint64_t) { ASSERT(false); }

    // Call this before popping CB entry
    void async_writes_flushed() {
        if constexpr (use_scatter_write) {
            // Send any outstanding pages. This can happen when total number of tensor pages doesn't evenly
            // divide by pages per packet.
            static_assert(min_pages_per_packet == 2, "hardcoded to assume scatter_write min_pages_per_packet == 2");
            if (chunk_count > 0) {
                send_queued_chunks();
            }
        } else {
            // no remaining data to send here
        }
        // Wait for Fabric writes to be sent out before popping CB entry
        noc.async_writes_flushed();
    }

private:
    void send_queued_chunks() {
        noc.async_writes_flushed();
        if (chunk_count == 1 || chunks_are_contiguous) {
            fabric_api::fabric_multicast_noc_unicast_write_with_state<
                UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                fabric_connection,
                use_route_1 ? unicast_route_id_1 : unicast_route_id_2,
                start_l1_addr,
                tt::tt_fabric::NocUnicastCommandHeader{scatter_header.noc_address[0]},
                chunk_count * page_size);
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
            use_route_1 = !use_route_1;
        }
    }

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
    bool chunks_are_contiguous;
};

// Send one contiguous transport batch to a mirrored Tensix L1 slot on every device
// covered by the route.  The fused atomic increment is ordered after the payload and
// tells the receiver that the slot is ready to drain.  Unlike FabricWriter, destination
// tensor pages are deliberately not encoded here: the receiver owns the final DRAM
// address generation and can fan the batch out across interleaved banks locally.
template <uint32_t packet_size, bool alternate_routes, bool fused_notify, bool explicit_path>
class FabricL1Writer {
public:
    FabricL1Writer(
        const Noc& noc,
        tt::tt_fabric::RoutingPlaneConnectionManager& manager,
        uint32_t num_connections,
        FabricRange* ranges,
        FabricRange* ranges_alt,
        const FabricExplicitPath& path,
        const FabricExplicitPath& path_alt) :
        noc{noc},
        fabric_connection{manager},
        payload_route_id_1{PacketHeaderPool::allocate_header_n(num_connections)},
        payload_route_id_2{
            alternate_routes ? PacketHeaderPool::allocate_header_n(num_connections) : payload_route_id_1},
        notify_route_id_1{fused_notify ? payload_route_id_1 : PacketHeaderPool::allocate_header_n(num_connections)},
        notify_route_id_2{
            fused_notify
                ? payload_route_id_2
                : (alternate_routes ? PacketHeaderPool::allocate_header_n(num_connections) : notify_route_id_1)},
        use_route_1{true} {
        uint8_t starts[1] = {1};
        if constexpr (fused_notify) {
            fabric_api::fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state<
                UnicastFusedAtomicIncUpdateMask::Val | UnicastFusedAtomicIncUpdateMask::Flush |
                UnicastFusedAtomicIncUpdateMask::DeferNotification>(
                fabric_connection,
                payload_route_id_1,
#ifndef FABRIC_2D
                starts,
#endif
                ranges,
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{0, 0, 1, true, true});
        } else {
            fabric_api::fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
                fabric_connection,
                payload_route_id_1,
#ifndef FABRIC_2D
                starts,
#endif
                ranges);
            fabric_api::fabric_multicast_noc_unicast_atomic_inc_set_state<
                UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                fabric_connection,
                notify_route_id_1,
#ifndef FABRIC_2D
                starts,
#endif
                ranges,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1});
        }
        apply_explicit_fabric_path<explicit_path>(payload_route_id_1, path);
        if constexpr (!fused_notify) {
            apply_explicit_fabric_path<explicit_path>(notify_route_id_1, path);
        }

        if constexpr (alternate_routes) {
            if constexpr (fused_notify) {
                fabric_api::fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state<
                    UnicastFusedAtomicIncUpdateMask::Val | UnicastFusedAtomicIncUpdateMask::Flush |
                    UnicastFusedAtomicIncUpdateMask::DeferNotification>(
                    fabric_connection,
                    payload_route_id_2,
#ifndef FABRIC_2D
                    starts,
#endif
                    ranges_alt,
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{0, 0, 1, true, true});
            } else {
                fabric_api::fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
                    fabric_connection,
                    payload_route_id_2,
#ifndef FABRIC_2D
                    starts,
#endif
                    ranges_alt);
                fabric_api::fabric_multicast_noc_unicast_atomic_inc_set_state<
                    UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                    fabric_connection,
                    notify_route_id_2,
#ifndef FABRIC_2D
                    starts,
#endif
                    ranges_alt,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1});
            }
            apply_explicit_fabric_path<explicit_path>(payload_route_id_2, path_alt);
            if constexpr (!fused_notify) {
                apply_explicit_fabric_path<explicit_path>(notify_route_id_2, path_alt);
            }
        }
    }

    void async_write(uint32_t l1_addr, uint32_t payload_size, uint64_t remote_l1_addr, uint64_t produced_sem_addr) {
        ASSERT(payload_size > 0 && payload_size <= packet_size);
        noc.async_writes_flushed();
        if constexpr (fused_notify) {
            fabric_api::fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<
                UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::SemaphoreAddr |
                UnicastFusedAtomicIncUpdateMask::PayloadSize>(
                fabric_connection,
                use_route_1 ? payload_route_id_1 : payload_route_id_2,
                l1_addr,
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{remote_l1_addr, produced_sem_addr, 0, true},
                payload_size);
        } else {
            fabric_api::fabric_multicast_noc_unicast_write_with_state<
                UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                fabric_connection,
                use_route_1 ? payload_route_id_1 : payload_route_id_2,
                l1_addr,
                tt::tt_fabric::NocUnicastCommandHeader{remote_l1_addr},
                payload_size);
            fabric_api::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric_connection,
                use_route_1 ? notify_route_id_1 : notify_route_id_2,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{produced_sem_addr, 0});
        }
        if constexpr (alternate_routes) {
            use_route_1 = !use_route_1;
        }
    }

    // Common transport surface used by the compile-time selected sender loop.
    // This overload is unreachable for receiver-L1 instantiations.
    void async_write(uint32_t, uint64_t) { ASSERT(false); }

    void async_writes_flushed() { noc.async_writes_flushed(); }

private:
    const Noc& noc;
    tt::tt_fabric::RoutingPlaneConnectionManager& fabric_connection;
    uint8_t payload_route_id_1;
    uint8_t payload_route_id_2;
    uint8_t notify_route_id_1;
    uint8_t notify_route_id_2;
    bool use_route_1;
};
