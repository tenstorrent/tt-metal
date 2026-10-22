// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "fabric/fabric_edm_packet_header.hpp"

#include <cstdint>
#include <type_traits>

// One fabric route, from all_gather_program.cpp::append_route_args. Slot headers and semaphore incs are routed
// from the same block, so they cannot disagree. A range packs hops per physical direction (e | w << 8 | n << 16 |
// s << 24); the two multicast variants alternate per packet on an even ring.
struct RouteArgs {
    uint32_t dst_dev;
    uint32_t dst_mesh;
    uint32_t range_v0;
    uint32_t range_v1;

    static RouteArgs read(uint32_t& rt_idx) {
        RouteArgs r;
        r.dst_dev = get_arg_val<uint32_t>(rt_idx++);
        r.dst_mesh = get_arg_val<uint32_t>(rt_idx++);
        r.range_v0 = get_arg_val<uint32_t>(rt_idx++);
        r.range_v1 = get_arg_val<uint32_t>(rt_idx++);
        return r;
    }
};

// Keyed on the header type, not FABRIC_2D: that define only reaches kernels that open a fabric connection, and
// builders never do. The header type comes from ROUTING_MODE, which every kernel gets.
inline constexpr bool kMeshHeader = std::is_base_of_v<tt::tt_fabric::HybridMeshPacketHeader, PACKET_HEADER_TYPE>;

template <bool kMcast>
FORCE_INLINE void set_route(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, const RouteArgs& r, uint32_t variant) {
    const uint32_t range = variant == 0 ? r.range_v0 : r.range_v1;
    const uint8_t e = range & 0xFF;
    const uint8_t w = (range >> 8) & 0xFF;
    const uint8_t n = (range >> 16) & 0xFF;
    const uint8_t s = range >> 24;
    if constexpr (kMeshHeader) {
        auto* mesh_hdr = reinterpret_cast<volatile tt_l1_ptr tt::tt_fabric::HybridMeshPacketHeader*>(hdr);
        if constexpr (kMcast) {
            fabric_set_mcast_route(mesh_hdr, r.dst_dev, r.dst_mesh, e, w, n, s);
        } else {
            fabric_set_unicast_route(mesh_hdr, r.dst_dev, r.dst_mesh);
        }
    } else {
        if constexpr (kMcast) {
            // 1D: one connection runs along one direction, so exactly one of e/w/n/s is set.
            hdr->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(e + w + n + s)});
        } else {
            hdr->to_chip_unicast(1);
        }
    }
}

// Slot layout [pad | header 0 .. header C-1 | payload]: header c sits (C - c) headers before the payload, so the
// last header and the payload are one contiguous packet (one write for connection C - 1).
FORCE_INLINE constexpr uint32_t slot_header_offset(uint32_t c, uint32_t C, uint32_t hdr_bytes, uint32_t payload_off) {
    return payload_off - (C - c) * hdr_bytes;
}

// The slot CB, as named compile-time args from all_gather_program.cpp::create_kernels (every kernel gets them).
struct SlotCt {
    static constexpr uint32_t cb = get_named_compile_time_arg_val("cb");
    static constexpr uint32_t slots = get_named_compile_time_arg_val("slots");
    static constexpr uint32_t slot_bytes = get_named_compile_time_arg_val("slot_bytes");
    static constexpr uint32_t payload_off = get_named_compile_time_arg_val("payload_off");
    static constexpr uint32_t hdr_bytes = get_named_compile_time_arg_val("hdr_bytes");
    static constexpr uint32_t C = get_named_compile_time_arg_val("C");
    static constexpr uint32_t A = get_named_compile_time_arg_val("A");

    static_assert(sizeof(PACKET_HEADER_TYPE) == hdr_bytes, "host and kernel disagree on the header size");
};

// Routes header c of every slot once, from routes[c]; per packet the builder writes only the NOC command. With
// alternation, slot parity picks the variant: the slot count is even, so consecutive packets alternate.
template <bool kMcast, bool kAlternate>
FORCE_INLINE void route_slots(
    uint32_t base,
    uint32_t slots,
    uint32_t slot_bytes,
    uint32_t hdr_bytes,
    uint32_t payload_off,
    const RouteArgs* routes,
    uint32_t C) {
    for (uint32_t s = 0; s < slots; ++s) {
        for (uint32_t c = 0; c < C; ++c) {
            auto* hdr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
                base + s * slot_bytes + slot_header_offset(c, C, hdr_bytes, payload_off));
            set_route<kMcast>(hdr, routes[c], kAlternate ? s % 2 : 0);
        }
    }
}

// Waits for `count` on a local semaphore, then takes it back. Subtract, never clear: a peer one invocation
// ahead may already have posted its next increments.
FORCE_INLINE void sem_wait_consume(uint32_t sem, uint32_t count) {
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), count);
    noc_semaphore_inc(get_noc_addr(sem), uint32_t{0} - count);
    noc_async_atomic_barrier();
}
