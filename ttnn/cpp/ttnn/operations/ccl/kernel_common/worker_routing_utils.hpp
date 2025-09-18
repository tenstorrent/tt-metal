// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/fabric_common.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

namespace ccl_routing_utils {

template <typename... T>
inline constexpr bool always_false_v = false;

struct line_unicast_route_info_t {
    uint16_t dst_mesh_id;
    union {
        uint16_t dst_chip_id;
        uint16_t distance_in_hops;
    };
};

struct line_multicast_route_info_t {
    union {
        uint16_t dst_mesh_id;
        uint16_t start_distance_in_hops;
    };
    union {
        uint16_t dst_chip_id;
        uint16_t range_hops;
    };
    uint16_t e_num_hops;
    uint16_t w_num_hops;
    uint16_t n_num_hops;
    uint16_t s_num_hops;
};

inline constexpr uint32_t num_line_unicast_args = 2;
inline constexpr uint32_t num_line_multicast_args = 6;

template <uint32_t arg_idx>
constexpr line_unicast_route_info_t get_line_unicast_route_info_from_args() {
    return {.dst_mesh_id = get_compile_time_arg_val(arg_idx), .dst_chip_id = get_compile_time_arg_val(arg_idx + 1)};
}

template <uint32_t arg_idx>
constexpr line_multicast_route_info_t get_line_multicast_route_info_from_args() {
    return {
        .dst_mesh_id = get_compile_time_arg_val(arg_idx),
        .dst_chip_id = get_compile_time_arg_val(arg_idx + 1),
        .e_num_hops = get_compile_time_arg_val(arg_idx + 2),
        .w_num_hops = get_compile_time_arg_val(arg_idx + 3),
        .n_num_hops = get_compile_time_arg_val(arg_idx + 4),
        .s_num_hops = get_compile_time_arg_val(arg_idx + 5),
    };
}

// dst_chip_id is the hop count for 1D routing, and the chip ID for 2D routing
// dst_mesh_id is the mesh ID of the destination chip (ignored for 1D routing)
template <typename packet_header_t>
FORCE_INLINE void fabric_set_line_unicast_route(
    volatile tt_l1_ptr packet_header_t* fabric_header_addr, const line_unicast_route_info_t& route_info) {
    if constexpr (std::is_same_v<packet_header_t, tt::tt_fabric::MeshPacketHeader>) {
        fabric_set_unicast_route(
            fabric_header_addr,
            0,  // Ignored
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
        fabric_set_mcast_route(
            fabric_header_addr,
            route_info.dst_chip_id,
            route_info.dst_mesh_id,
            route_info.e_num_hops,
            route_info.w_num_hops,
            route_info.n_num_hops,
            route_info.s_num_hops
        );
    } else if constexpr (std::is_same_v<packet_header_t, tt::tt_fabric::LowLatencyPacketHeader>) {
        fabric_header_addr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
            static_cast<uint8_t>(route_info.start_distance_in_hops), static_cast<uint8_t>(route_info.range_hops)});
    } else {
        static_assert(
            always_false_v<packet_header_t>, "Unsupported packet header type passed to fabric_set_line_mcast_route");
    }
}

}  // namespace ccl_routing_utils
