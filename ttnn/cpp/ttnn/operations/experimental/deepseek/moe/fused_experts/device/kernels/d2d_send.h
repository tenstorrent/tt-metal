// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/mesh/api.h"

// Sends to the receivers of a D2D socket, which share one core and, allocated in lockstep, one FIFO and
// bytes_sent address: over routes the host derived (d2d_routes) -- one multicast, or a unicast per
// receiver -- all leaving through one fabric connection. A route's runtime args are its destination
// chip and mesh, then its hops E, W, N, S (all zero: a unicast), then the direction a unicast to a chip
// of the sender's column is steered along it (kD2DNoSteer: the routing table's way).

constexpr uint32_t kD2DRouteArgs = 7;
constexpr uint32_t kD2DNoSteer = 0xFF;

struct D2DRoute {
    uint8_t dst_chip;
    uint16_t dst_mesh;
    tt::tt_fabric::mesh::experimental::MeshMcastRange range;
    bool multicast;
    uint8_t steer;
};

// Rewrites a unicast header's row map so that every row but the destination's forwards it in the route's
// steering direction; at the destination's row the column map delivers it, as the table had it.
inline void steer_d2d_unicast(volatile tt_l1_ptr PACKET_HEADER_TYPE* header, const D2DRoute& route) {
    const auto* routing_table = reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const uint32_t rows = routing_table->mesh_y_size;
    const uint32_t dst_row = route.dst_chip / routing_table->mesh_x_size;
    for (uint32_t r = 0; r < rows; ++r) {
        header->route_buffer[r] = r == dst_row ? 0 : static_cast<uint8_t>(1u << route.steer);
    }
}

inline D2DRoute read_d2d_route(size_t& arg_idx) {
    D2DRoute route;
    route.dst_chip = get_arg_val<uint32_t>(arg_idx++);
    route.dst_mesh = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t e = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t w = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t n = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t s = get_arg_val<uint32_t>(arg_idx++);
    route.range = tt::tt_fabric::mesh::experimental::MeshMcastRange{e, w, n, s};
    route.multicast = (e | w | n | s) != 0;
    route.steer = get_arg_val<uint32_t>(arg_idx++);
    return route;
}

// `size` bytes from `src` to `dst` on core (x, y) of the route's chips, in packets of at most
// max_payload bytes.
template <uint32_t max_payload, typename Sender>
void d2d_send(
    Sender* sender,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header,
    const D2DRoute& route,
    uint32_t src,
    uint32_t size,
    uint32_t x,
    uint32_t y,
    uint32_t dst) {
    namespace fabric_api = tt::tt_fabric::mesh::experimental;
    for (uint32_t left = size; left > 0;) {
        const uint32_t chunk = left > max_payload ? max_payload : left;
        // The header is rewritten in place, so the previous packet's must have left.
        noc_async_writes_flushed();
        const tt::tt_fabric::NocUnicastCommandHeader cmd{safe_get_noc_addr(x, y, dst, 0)};
        if (route.multicast) {
            fabric_api::fabric_multicast_noc_unicast_write(
                sender, header, route.dst_chip, route.dst_mesh, route.range, src, chunk, cmd);
        } else if (route.steer != kD2DNoSteer) {
            using tt::tt_fabric::common::experimental::UnicastWriteUpdateMask;
            fabric_api::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::All>(
                header, route.dst_chip, route.dst_mesh, cmd, chunk);
            steer_d2d_unicast(header, route);
            fabric_api::fabric_unicast_noc_unicast_write_with_state(sender, header, src);
        } else {
            fabric_api::fabric_unicast_noc_unicast_write(
                sender, header, route.dst_chip, route.dst_mesh, src, chunk, cmd);
        }
        src += chunk;
        dst += chunk;
        left -= chunk;
    }
}

// The word `value` to `addr` on core (x, y) of the route's chips.
template <typename Sender>
void d2d_send_word(
    Sender* sender,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header,
    const D2DRoute& route,
    uint32_t x,
    uint32_t y,
    uint32_t addr,
    uint32_t value) {
    namespace fabric_api = tt::tt_fabric::mesh::experimental;
    noc_async_writes_flushed();
    const tt::tt_fabric::NocUnicastInlineWriteCommandHeader cmd{safe_get_noc_addr(x, y, addr, 0), value};
    if (route.multicast) {
        fabric_api::fabric_multicast_noc_unicast_inline_write(
            sender, header, route.dst_chip, route.dst_mesh, route.range, cmd);
    } else if (route.steer != kD2DNoSteer) {
        using tt::tt_fabric::common::experimental::UnicastInlineWriteUpdateMask;
        fabric_api::fabric_unicast_noc_unicast_inline_write_set_state<UnicastInlineWriteUpdateMask::All>(
            header, route.dst_chip, route.dst_mesh, cmd);
        steer_d2d_unicast(header, route);
        fabric_api::fabric_unicast_noc_unicast_inline_write_with_state(sender, header);
    } else {
        fabric_api::fabric_unicast_noc_unicast_inline_write(sender, header, route.dst_chip, route.dst_mesh, cmd);
    }
}

// A ring of packet headers, so a burst of packets does not wait for each one to leave L1 before building the
// next: a header is rewritten only after the ring wraps, behind one flush for the whole ring.
template <uint32_t N>
struct D2DHeaderRing {
    volatile tt_l1_ptr PACKET_HEADER_TYPE* headers[N];
    uint32_t next = 0;

    void init() {
        for (uint32_t i = 0; i < N; ++i) {
            headers[i] = PacketHeaderPool::allocate_header();
        }
    }
    // Before a burst: every header of the previous one has left.
    void reset() {
        noc_async_writes_flushed();
        next = 0;
    }
    volatile tt_l1_ptr PACKET_HEADER_TYPE* take() {
        if (next == N) {
            noc_async_writes_flushed();
            next = 0;
        }
        return headers[next++];
    }
};

// `size` bytes from `src` to `dst` on core (x, y) of the route's chips, then +1 on `flag` there: the last
// chunk carries the increment as one fused packet, issued after its write by the destination router, so the
// flag counts only once every byte has landed (the route's packets, and a packet's write and increment, arrive
// in order). Each packet takes its own
// header from `ring`, so the burst's packets are issued back to back.
template <uint32_t max_payload, uint32_t N, typename Sender>
void d2d_send_flagged(
    Sender* sender,
    D2DHeaderRing<N>& ring,
    const D2DRoute& route,
    uint32_t src,
    uint32_t size,
    uint32_t x,
    uint32_t y,
    uint32_t dst,
    uint32_t flag) {
    namespace fabric_api = tt::tt_fabric::mesh::experimental;
    using tt::tt_fabric::common::experimental::UnicastFusedAtomicIncUpdateMask;
    using tt::tt_fabric::common::experimental::UnicastWriteUpdateMask;
    for (uint32_t left = size; left > 0;) {
        const uint32_t chunk = left > max_payload ? max_payload : left;
        auto* header = ring.take();
        if (chunk == left) {
            // No flush at the destination: like the separate flag packet this replaces (and ar_sender's fused
            // packets), the increment follows the write from the same router to the same core in order; a flush
            // there would hold the router for each write's ack and serialize the burst.
            const tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader cmd{
                safe_get_noc_addr(x, y, dst, 0), safe_get_noc_addr(x, y, flag, 0), 1, false};
            if (route.multicast) {
                fabric_api::fabric_multicast_noc_fused_unicast_with_atomic_inc(
                    sender, header, route.dst_chip, route.dst_mesh, route.range, src, chunk, cmd);
            } else if (route.steer != kD2DNoSteer) {
                fabric_api::fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<
                    UnicastFusedAtomicIncUpdateMask::All>(header, route.dst_chip, route.dst_mesh, cmd, chunk);
                steer_d2d_unicast(header, route);
                fabric_api::fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(sender, header, src);
            } else {
                fabric_api::fabric_unicast_noc_fused_unicast_with_atomic_inc(
                    sender, header, route.dst_chip, route.dst_mesh, src, chunk, cmd);
            }
        } else {
            const tt::tt_fabric::NocUnicastCommandHeader cmd{safe_get_noc_addr(x, y, dst, 0)};
            if (route.multicast) {
                fabric_api::fabric_multicast_noc_unicast_write(
                    sender, header, route.dst_chip, route.dst_mesh, route.range, src, chunk, cmd);
            } else if (route.steer != kD2DNoSteer) {
                fabric_api::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::All>(
                    header, route.dst_chip, route.dst_mesh, cmd, chunk);
                steer_d2d_unicast(header, route);
                fabric_api::fabric_unicast_noc_unicast_write_with_state(sender, header, src);
            } else {
                fabric_api::fabric_unicast_noc_unicast_write(
                    sender, header, route.dst_chip, route.dst_mesh, src, chunk, cmd);
            }
        }
        src += chunk;
        dst += chunk;
        left -= chunk;
    }
}
