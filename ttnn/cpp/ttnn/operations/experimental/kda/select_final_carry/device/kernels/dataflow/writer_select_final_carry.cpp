// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/select_final_carry/device/kernels/dataflow/select_final_carry_common.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/persistent_counter.hpp"

using namespace tt::tt_fabric::linear::experimental;

constexpr uint32_t tiles_cb = get_compile_time_arg_val(0);
constexpr uint32_t mode_cb = get_compile_time_arg_val(1);
constexpr uint32_t packet_pages = get_compile_time_arg_val(2);
constexpr uint32_t page_size = get_compile_time_arg_val(3);
constexpr uint32_t line_targets = get_compile_time_arg_val(4);
constexpr uint32_t start_hops_forward = get_compile_time_arg_val(5);
constexpr uint32_t range_hops_forward = get_compile_time_arg_val(6);
constexpr uint32_t start_hops_backward = get_compile_time_arg_val(7);
constexpr uint32_t range_hops_backward = get_compile_time_arg_val(8);
// On a 2D fabric whose SP axis turns, a multicast cannot follow the axis: every send goes to each other rank as a
// 2D unicast instead (the per-peer connection, chip and mesh follow the connection arguments).
constexpr bool unicast_peers = get_compile_time_arg_val(9);
constexpr auto output_args = TensorAccessorArgs<10>();

// Write the final carry: the rank holding it writes it locally and multicasts it along the sequence-parallel line,
// and the other ranks wait for its arrival.
void kernel_main() {
    using namespace kda_select_final_carry;
    const uint32_t output_address = get_common_arg_val<uint32_t>(0);
    size_t arg = 0;
    const uint32_t send_begin = get_arg_val<uint32_t>(arg++);
    const uint32_t send_end = get_arg_val<uint32_t>(arg++);
    const auto output = TensorAccessor(output_args, output_address);

    cb_wait_front(mode_cb, 1);
    const Mode mode = static_cast<Mode>(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(mode_cb))[0]);
    cb_pop_front(mode_cb, 1);

    const size_t arrival_semaphore = get_arg_val<uint32_t>(arg++);
    const uint8_t drain_x = get_arg_val<uint32_t>(arg++);
    const uint8_t drain_y = get_arg_val<uint32_t>(arg++);
    const bool drain = get_arg_val<uint32_t>(arg++);
    const uint32_t arrivals = get_arg_val<uint32_t>(arg++);
    const size_t barrier_semaphore = get_arg_val<uint32_t>(arg++);
    const uint8_t barrier_x = get_arg_val<uint32_t>(arg++);
    const uint8_t barrier_y = get_arg_val<uint32_t>(arg++);
    const uint32_t num_connections = get_arg_val<uint32_t>(arg++);

    // The multicast routes, one header per connection each; unicast_peers sends with one header of its own instead.
    const uint8_t write_route = unicast_peers ? 0 : PacketHeaderPool::allocate_header_n(num_connections);
    const uint8_t scatter_route = unicast_peers ? 0 : PacketHeaderPool::allocate_header_n(num_connections);
    const uint8_t semaphore_route = unicast_peers ? 0 : PacketHeaderPool::allocate_header_n(num_connections);
    tt::tt_fabric::RoutingPlaneConnectionManager fabric;
    open_connections(fabric, num_connections, arg);
    // unicast_peers: up to max_peer_headers headers in turn, so the sends to several peers leave before one flush. A
    // header's route is rewritten only when it serves another peer, after the previous packet has left L1.
    const size_t peer_args = arg;
    constexpr size_t args_per_peer = 3;
    constexpr uint32_t max_peer_headers = 8;
    constexpr uint32_t peer_headers = !unicast_peers                    ? 0
                                      : line_targets < max_peer_headers ? line_targets
                                                                        : max_peer_headers;
    decltype(PacketHeaderPool::allocate_header()) peer_routes[peer_headers > 0 ? peer_headers : 1];
    for (uint32_t h = 0; h < peer_headers; ++h) {
        peer_routes[h] = PacketHeaderPool::allocate_header();
    }
    // With a header per peer, each is routed on its first send and keeps that route.
    bool routed = false;
    // The header that serves `peer`, routed to it (each header keeps its route while it serves one peer only).
    const auto peer_route = [&](uint32_t peer) {
        auto* header = peer_routes[peer % (peer_headers > 0 ? peer_headers : 1)];
#if defined(FABRIC_2D)
        if (peer_headers < line_targets || !routed) {
            const size_t args = peer_args + peer * args_per_peer;
            fabric_set_unicast_route(header, get_arg_val<uint32_t>(args + 1), get_arg_val<uint32_t>(args + 2));
        }
#endif
        return header;
    };
    const auto to_peer = [&](uint32_t peer) -> auto& {
        return fabric.get(get_arg_val<uint32_t>(peer_args + peer * args_per_peer)).sender;
    };
    // Flush after every peer_headers sends, before a header is reused, and once at the end.
    const auto after_send = [&](uint32_t peer) {
        if ((peer + 1) % (peer_headers > 0 ? peer_headers : 1) == 0 || peer + 1 == line_targets) {
            noc_async_writes_flushed();
        }
    };
    // Bump the semaphore at `address` on every other rank of the line.
    const auto increment_others = [&](uint64_t address) {
        if constexpr (unicast_peers) {
            for (uint32_t peer = 0; peer < line_targets; ++peer) {
                fabric_unicast_noc_unicast_atomic_inc(
                    &to_peer(peer),
                    peer_route(peer),
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{address, 1, true},
                    1);
                after_send(peer);
            }
            routed = true;
            return;
        }
        fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            fabric, semaphore_route, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{address, 0});
    };

    if constexpr (!unicast_peers) {
        uint8_t starts[] = {static_cast<uint8_t>(start_hops_forward), static_cast<uint8_t>(start_hops_backward)};
        uint8_t ranges[] = {static_cast<uint8_t>(range_hops_forward), static_cast<uint8_t>(range_hops_backward)};
        // Connections are packed in destination order, so a rank with only a backward neighbor sends on connection 0.
        if (ranges[0] == 0) {
            starts[0] = starts[1];
            ranges[0] = ranges[1];
        }
        fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            fabric, write_route, starts, ranges, nullptr, page_size);
        if constexpr (packet_pages > 1) {
            fabric_multicast_noc_scatter_write_set_state<
                UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
                fabric,
                scatter_route,
                starts,
                ranges,
                NocUnicastScatterCommandHeader({0, 0}, {static_cast<uint16_t>(page_size)}),
                page_size * 2);
        }
        fabric_multicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            fabric, semaphore_route, starts, ranges, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1u});
    }

    // Every rank on the line must have entered this program before the owner overwrites its output.
    increment_others(safe_get_noc_addr(barrier_x, barrier_y, barrier_semaphore, 0));
    kda_counter::wait_and_consume(barrier_semaphore, line_targets);

    if (mode == broadcast) {
        for (uint32_t tile = send_begin; tile < send_end;) {
            cb_wait_front(tiles_cb, packet_pages);
            const uint32_t l1_address = get_read_ptr(tiles_cb);
            const uint32_t packet_end = tile + packet_pages < send_end ? tile + packet_pages : send_end;
            // A packet holds one or two pages; scatter writes carry two destination pages in one fabric packet.
            if constexpr (unicast_peers) {
                for (uint32_t page = tile; page < packet_end; ++page) {
                    const uint32_t source = l1_address + (page - tile) * page_size;
                    noc_async_write_page(page, output, source);
                    for (uint32_t peer = 0; peer < line_targets; ++peer) {
                        fabric_unicast_noc_unicast_write(
                            &to_peer(peer),
                            peer_route(peer),
                            source,
                            page_size,
                            tt::tt_fabric::NocUnicastCommandHeader{
                                linear::addrgen_detail::get_noc_address(output, page, 0)},
                            1);
                        after_send(peer);
                    }
                }
            } else if (packet_end - tile == 1) {
                noc_async_write_page(tile, output, l1_address);
                fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                    fabric,
                    write_route,
                    l1_address,
                    tt::tt_fabric::NocUnicastCommandHeader{linear::addrgen_detail::get_noc_address(output, tile, 0)});
            } else if constexpr (packet_pages > 1) {
                noc_async_write_page(tile, output, l1_address);
                noc_async_write_page(tile + 1, output, l1_address + page_size);
                fabric_multicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                    fabric,
                    scatter_route,
                    l1_address,
                    tt::tt_fabric::NocUnicastScatterCommandHeader(
                        {linear::addrgen_detail::get_noc_address(output, tile, 0),
                         linear::addrgen_detail::get_noc_address(output, tile + 1, 0)}),
                    page_size * 2);
            }
            tile = packet_end;
            noc_async_writes_flushed();
            cb_pop_front(tiles_cb, packet_pages);
        }
        // Flushed after this link's data, the increment tells each receiver that its share has landed.
        increment_others(safe_get_noc_addr(drain_x, drain_y, arrival_semaphore, 0));
    } else if (drain) {
        kda_counter::wait_and_consume(arrival_semaphore, arrivals);
    }
    close_connections(fabric);
    noc_async_write_barrier();
}
