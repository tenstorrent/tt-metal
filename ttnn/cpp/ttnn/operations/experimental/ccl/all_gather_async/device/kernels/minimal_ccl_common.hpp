// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "fabric_host_interface.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <cstdint>
#include <utility>

FORCE_INLINE void write_and_advance_local_read_address_for_fabric_write(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
    const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
    const size_t payload_l1_address = l1_read_addr;

    pkt_hdr_forward->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
    pkt_hdr_backward->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

    noc_async_write(payload_l1_address, safe_get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr), payload_size_bytes);
    if (fabric_connection.has_forward_connection()) {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_forward, sizeof(PACKET_HEADER_TYPE));
    }

    if (fabric_connection.has_backward_connection()) {
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_backward, sizeof(PACKET_HEADER_TYPE));
    }

    noc_async_writes_flushed();

    l1_read_addr += payload_size_bytes;
}

#ifdef ARCH_WORMHOLE
FORCE_INLINE void scatter_write_for_fabric_write_forward(
    uint64_t first_noc0_dest_noc_addr,
    uint64_t second_noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t first_payload_size_bytes,
    uint32_t second_payload_size_bytes) {
    pkt_hdr_forward->to_noc_unicast_scatter_write(
        tt::tt_fabric::NocUnicastScatterCommandHeader{
            {first_noc0_dest_noc_addr, second_noc0_dest_noc_addr}, (uint16_t)first_payload_size_bytes},
        first_payload_size_bytes + second_payload_size_bytes);

    fabric_connection.get_forward_connection().wait_for_empty_write_slot();
    fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
        l1_read_addr, first_payload_size_bytes + second_payload_size_bytes);
    fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
        (uint32_t)pkt_hdr_forward, sizeof(PACKET_HEADER_TYPE));
    noc_async_writes_flushed();
}
#endif

FORCE_INLINE void write_for_fabric_write_forward(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
    pkt_hdr_forward->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

    fabric_connection.get_forward_connection().wait_for_empty_write_slot();
    fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
        l1_read_addr, payload_size_bytes);
    fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
        (uint32_t)pkt_hdr_forward, sizeof(PACKET_HEADER_TYPE));
    noc_async_writes_flushed();
}

#ifdef ARCH_WORMHOLE
FORCE_INLINE void scatter_write_for_fabric_write_backward(
    uint64_t first_noc0_dest_noc_addr,
    uint64_t second_noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t first_payload_size_bytes,
    uint32_t second_payload_size_bytes) {
    pkt_hdr_backward->to_noc_unicast_scatter_write(
        tt::tt_fabric::NocUnicastScatterCommandHeader{
            {first_noc0_dest_noc_addr, second_noc0_dest_noc_addr}, (uint16_t)first_payload_size_bytes},
        first_payload_size_bytes + second_payload_size_bytes);

    fabric_connection.get_backward_connection().wait_for_empty_write_slot();
    fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
        l1_read_addr, first_payload_size_bytes + second_payload_size_bytes);
    fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
        (uint32_t)pkt_hdr_backward, sizeof(PACKET_HEADER_TYPE));
    noc_async_writes_flushed();
}
#endif

FORCE_INLINE void write_for_fabric_write_backward(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
    pkt_hdr_backward->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

    fabric_connection.get_backward_connection().wait_for_empty_write_slot();
    fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
        l1_read_addr, payload_size_bytes);
    fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
        (uint32_t)pkt_hdr_backward, sizeof(PACKET_HEADER_TYPE));
    noc_async_writes_flushed();
}

// Function does not block or wait for writes to be sent out of L1. Caller must manage synchronization
FORCE_INLINE void fused_write_atomic_and_advance_local_read_address_for_fabric_write(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes,
    uint64_t semaphore_noc_addr,
    const uint16_t val,
    const uint16_t wrap,
    const bool flush) {
    const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
    const size_t payload_l1_address = l1_read_addr;

    noc_async_write(payload_l1_address, safe_get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr), payload_size_bytes);
    if (fabric_connection.has_forward_connection()) {
        pkt_hdr_forward->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                noc0_dest_noc_addr, semaphore_noc_addr, val, wrap, flush},
            payload_size_bytes);
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_forward, sizeof(PACKET_HEADER_TYPE));
    }

    if (fabric_connection.has_backward_connection()) {
        pkt_hdr_backward->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                noc0_dest_noc_addr, semaphore_noc_addr, val, wrap, flush},
            payload_size_bytes);
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_backward, sizeof(PACKET_HEADER_TYPE));
    }

    l1_read_addr += payload_size_bytes;
}

namespace ccl_routing_utils {

template <typename... T>
inline constexpr bool always_false_v = false;

struct line_unicast_route_info_t {
    uint32_t dst_mesh_id;
    union {
        uint32_t dst_chip_id;
        uint32_t distance_in_hops;
    };
};

struct line_multicast_route_info_t {
    tt::tt_fabric::eth_chan_directions routing_direction;
    uint32_t dst_mesh_id;
    union {
        uint32_t dst_chip_id;
        uint32_t start_distance_in_hops;
    };
    uint32_t range_hops;
};

inline constexpr uint32_t num_line_unicast_args = 2;
inline constexpr uint32_t num_line_multicast_args = 4;

template <uint32_t arg_idx>
constexpr line_unicast_route_info_t get_line_unicast_route_info_from_args() {
    return {.dst_mesh_id = get_compile_time_arg_val(arg_idx), .dst_chip_id = get_compile_time_arg_val(arg_idx + 1)};
}

template <uint32_t arg_idx>
constexpr line_multicast_route_info_t get_line_multicast_route_info_from_args() {
    return {
        .routing_direction = static_cast<tt::tt_fabric::eth_chan_directions>(get_compile_time_arg_val(arg_idx)),
        .dst_mesh_id = get_compile_time_arg_val(arg_idx + 1),
        .dst_chip_id = get_compile_time_arg_val(arg_idx + 2),
        .range_hops = get_compile_time_arg_val(arg_idx + 3)};
}

// dst_chip_id is the hop count for 1D routing, and the chip ID for 2D routing
// dst_mesh_id is the mesh ID of the destination chip (ignored for 1D routing)
template <typename packet_header_t>
FORCE_INLINE void fabric_set_line_unicast_route(
    volatile tt_l1_ptr packet_header_t* fabric_header_addr, const line_unicast_route_info_t& route_info) {
    if constexpr (std::is_same_v<packet_header_t, tt::tt_fabric::MeshPacketHeader>) {
        fabric_set_unicast_route(
            fabric_header_addr,
            tt::tt_fabric::eth_chan_directions::COUNT,  // Ignored
            0,                                          // Ignored
            route_info.dst_chip_id,
            route_info.dst_mesh_id,
            0  // Ignored
        );
    } else if constexpr (std::is_same_v<packet_header_t, tt::tt_fabric::LowLatencyPacketHeader>) {
        fabric_header_addr->to_chip_unicast(static_cast<uint8_t>(route_info.distance_in_hops));
    } else {
        static_assert(
            always_false_v<packet_header_t>, "Unsupported packet header type passed to fabric_set_unicast_route");
    }
}

// dst_chip_id is the start hop count for 1D routing, and the chip ID for 2D routing
// dst_mesh_id is the mesh ID of the destination chip (ignored for 1D routing)
// routing_direction is the direction of the multicast, and is ignored for 1D routing
// num_hops is the number of hops for the multicast in the specified direction
template <typename packet_header_t>
FORCE_INLINE void fabric_set_line_multicast_route(
    volatile tt_l1_ptr packet_header_t* fabric_header_addr, const line_multicast_route_info_t& route_info) {
    if constexpr (std::is_same_v<packet_header_t, tt::tt_fabric::MeshPacketHeader>) {
        // TODO: Originally tried to pass struct as template param to have constexpr ifs, but not supported until C++20
        // Should probably just pass in a fully populated struct now and remove the if
        if (route_info.routing_direction == tt::tt_fabric::eth_chan_directions::EAST) {
            fabric_set_mcast_route(
                fabric_header_addr, route_info.dst_chip_id, route_info.dst_mesh_id, route_info.range_hops, 0, 0, 0);
        } else if (route_info.routing_direction == tt::tt_fabric::eth_chan_directions::WEST) {
            fabric_set_mcast_route(
                fabric_header_addr, route_info.dst_chip_id, route_info.dst_mesh_id, 0, route_info.range_hops, 0, 0);
        } else if (route_info.routing_direction == tt::tt_fabric::eth_chan_directions::NORTH) {
            fabric_set_mcast_route(
                fabric_header_addr, route_info.dst_chip_id, route_info.dst_mesh_id, 0, 0, route_info.range_hops, 0);
        } else if (route_info.routing_direction == tt::tt_fabric::eth_chan_directions::SOUTH) {
            fabric_set_mcast_route(
                fabric_header_addr, route_info.dst_chip_id, route_info.dst_mesh_id, 0, 0, 0, route_info.range_hops);
        } else {
            ASSERT(0);
        }
    } else if constexpr (std::is_same_v<packet_header_t, tt::tt_fabric::LowLatencyPacketHeader>) {
        fabric_header_addr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
            static_cast<uint8_t>(route_info.start_distance_in_hops), static_cast<uint8_t>(route_info.range_hops)});
    } else {
        static_assert(
            always_false_v<packet_header_t>, "Unsupported packet header type passed to fabric_set_line_mcast_route");
    }
}

}  // namespace ccl_routing_utils
