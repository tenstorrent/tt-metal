// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"  // tt_fabric::addrgen_detail::get_noc_address
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"

#include <cstdint>
#include <algorithm>
#include <type_traits>

#include "cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_common.hpp"

using address_t = uint32_t;

// Direct (one-shot) reduce-scatter writer: owns the fabric, sends this core's tile range of local input
// slice j to device j for every remote j, then writes the locally reduced output slice.
//
void kernel_main() {
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t tile_granularity = get_compile_time_arg_val(1);
    constexpr uint32_t chunks_per_slice = get_compile_time_arg_val(2);
    [[maybe_unused]] constexpr uint32_t pages_per_slice = get_compile_time_arg_val(3);
    constexpr uint32_t num_devices = get_compile_time_arg_val(4);
    constexpr uint32_t interm_tiles_per_packet = get_compile_time_arg_val(5);
    constexpr uint32_t cb_send_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(7);
    // Send straight into the destination's reduce CB (its staging is an L1 shard aliased into that CB) --
    // see the factory's CB comment. Off this path we write the interleaved staging tensor via its accessor.
    constexpr bool arrivals_in_cb = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t half_stride_tiles = get_compile_time_arg_val(9);  // tiles in one parity half
    constexpr bool needs_init_sync = get_compile_time_arg_val(10) != 0;
    constexpr auto staging_args = TensorAccessorArgs<11>();
    constexpr auto output_args = TensorAccessorArgs<staging_args.next_compile_time_args_offset()>();

    constexpr uint32_t num_dests = num_devices - 1;

    size_t ai = 0;
    const address_t staging_addr = get_arg_val<address_t>(ai++);
    const address_t output_addr = get_arg_val<address_t>(ai++);
    const uint32_t device_idx = get_arg_val<uint32_t>(ai++);
    const uint32_t chunk_start = get_arg_val<uint32_t>(ai++);
    const uint32_t chunk_count = get_arg_val<uint32_t>(ai++);
    const uint32_t tile_start = get_arg_val<uint32_t>(ai++);
    const uint32_t tile_count = get_arg_val<uint32_t>(ai++);
    const address_t gen_addr = get_arg_val<uint32_t>(ai++);     // this core's private invocation counter
    const address_t arrival_sem = get_arg_val<uint32_t>(ai++);  // our source slot's counter on every peer
    const uint8_t peer_core_x = get_arg_val<uint32_t>(ai++);    // mirror core (deterministic placement)
    const uint8_t peer_core_y = get_arg_val<uint32_t>(ai++);
    const uint32_t num_connections = get_arg_val<uint32_t>(ai++);  // 1 (fwd only) or 2 (fwd + bwd)
    const address_t init_sem = get_arg_val<uint32_t>(ai++);
    // Per-destination route, in send order (shared order with the reader's cb_send production).
    const size_t dest_conn = ai;
    ai += num_dests;
    const size_t dest_hops = ai;
    ai += num_dests;
    const size_t dest_block = ai;  // our block index inside that destination's reduce group (aliased path)
    ai += num_dests;
    // Destination chip/mesh per destination -- only read on a 2D fabric, which routes by destination
    // node rather than by hop count. See set_route_2d below.
    const size_t dest_chip = ai;
    ai += num_dests;
    const size_t dest_mesh = ai;
    ai += num_dests;
    size_t arg_for_fab = ai;  // fabric connection args are appended last

    auto output_acc = TensorAccessor(output_args, output_addr);  // tiled (page = tile_bytes)

    Noc noc;
    CircularBuffer cb_send(cb_send_id);
    CircularBuffer cb_out(cb_out_id);

    auto* gen_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gen_addr);
    const uint32_t invocation = *gen_ptr;
    // Same parity the peers' readers (and reducers) will use for this invocation -- see the reader's
    // parity note. Every peer runs the identical chunk partition, so our local chunk index k is also the
    // destination's, and the only per-destination term is which block of its group we own.
    const uint32_t staging_half = (invocation & 1u) * (num_devices * chunks_per_slice);
    const uint32_t dst_page_base = staging_half + device_idx * chunks_per_slice + chunk_start;
    const uint32_t half_off_bytes = (invocation & 1u) * half_stride_tiles * tile_bytes;
    constexpr uint32_t chunk_bytes = tile_granularity * tile_bytes;  // one block of one chunk

    // Connection 0 = forward neighbour, 1 = backward (host order); 1D routing takes the distance from the
    // packet header's hop count, so a header can be reused across both directions. Only START the open
    // here: open_finish carries a barrier, so it is deferred until just before the first send, leaving the
    // handshake to overlap the header setup below (and the reader's first input read).
    auto fabric_connection = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
        arg_for_fab, num_connections);

    auto data_pkt = PacketHeaderPool::allocate_header(1);
    auto fused_pkt = PacketHeaderPool::allocate_header(1);

    const uint64_t arrival_noc_addr = safe_get_noc_addr(peer_core_x, peer_core_y, arrival_sem, 0);

    // 1D routing takes the distance straight from the packet header's hop count, so a header is ready to
    // send once set_state has stamped num_hops on it. A 2D fabric instead routes by DESTINATION NODE:
    // set_state leaves the route empty (it only fills it in the RoutingPlaneConnectionManager overloads),
    // so the route has to be programmed onto each header for each destination.
    auto set_route_2d = [&](volatile PACKET_HEADER_TYPE* hdr, uint32_t dst) {
        if constexpr (std::is_base_of_v<tt::tt_fabric::HybridMeshPacketHeader, PACKET_HEADER_TYPE>) {
            fabric_set_unicast_route(
                reinterpret_cast<volatile tt::tt_fabric::HybridMeshPacketHeader*>(hdr),
                static_cast<uint16_t>(get_arg_val<uint32_t>(dest_chip + dst)),
                static_cast<uint16_t>(get_arg_val<uint32_t>(dest_mesh + dst)));
        }
    };

    fabric_connection.open_finish();

    if constexpr (needs_init_sync) {
        // One atomic inc per peer, once per invocation, sent on exactly the per-destination routes the
        // payload loop below uses: same connection, same hop count, same 2D route. So the barrier can
        // never be the one thing in this kernel that a reachable topology misroutes.
        //
        // The 1D path used to do this as two chip multicasts (one per direction, ranges from the host).
        // A multicast's hop fields are consumed by each router in turn, so it needs every hop of the
        // range to be a straight continuation of the direction it was injected in; a unicast only needs
        // the fabric to have a route. Those come apart on a mesh with more than one device on BOTH
        // axes, where the rings along a cluster_axis close over the mesh's wraparound links: the
        // multicast then stalls part of the ring in the wait below while the rest walks on (#54864).
        // The ring reduce_scatter's writer dropped multicast from its own barrier for the same reason
        // ("use neighbor unicast instead of multicast to support reshaped 'logical linear' mesh
        // devices"). Cost is num_dests header-only packets instead of two, once per invocation.
        auto init_pkt = PacketHeaderPool::allocate_header(1);
        const uint64_t init_noc_addr = safe_get_noc_addr(peer_core_x, peer_core_y, init_sem, 0);
        for (uint32_t dst = 0; dst < num_dests; ++dst) {
            auto* sender = &fabric_connection.get(get_arg_val<uint32_t>(dest_conn + dst)).sender;
            fabric_api::fabric_unicast_noc_unicast_atomic_inc_set_state<
                UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val |
                UnicastAtomicIncUpdateMask::Flush>(
                init_pkt,
                static_cast<uint8_t>(get_arg_val<uint32_t>(dest_hops + dst)),
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_noc_addr, 1u});
            set_route_2d(init_pkt, dst);
            fabric_api::fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::None>(
                sender, init_pkt);
            noc.async_writes_flushed();  // on the wire before the header is re-patched for the next dst
        }
        // Free function: init_sem arrives as a GlobalSemaphore address. See the reader's note.
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_sem), (invocation + 1) * num_dests);
    }

    for (uint32_t dst = 0; dst < num_dests; ++dst) {
        const uint8_t hops = static_cast<uint8_t>(get_arg_val<uint32_t>(dest_hops + dst));
        auto* sender = &fabric_connection.get(get_arg_val<uint32_t>(dest_conn + dst)).sender;
        // Aliased path: our slot in the destination's reduce CB. The shard is height-sharded, so it sits
        // at the same L1 address on the mirror core of every device -- our own staging address serves.
        const uint32_t block_base =
            arrivals_in_cb ? staging_addr + half_off_bytes + get_arg_val<uint32_t>(dest_block + dst) * chunk_bytes : 0u;

        fabric_api::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(data_pkt, hops);
        fabric_api::fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<
            UnicastFusedAtomicIncUpdateMask::Val | UnicastFusedAtomicIncUpdateMask::Flush>(
            fused_pkt,
            hops,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                0u,   // write dst (patched per packet)
                0u,   // semaphore dst (patched per packet)
                1u},  // increment 1 (flush defaults true: payload lands before the inc is applied)
            /*packet_size_bytes=*/0);
        set_route_2d(data_pkt, dst);
        set_route_2d(fused_pkt, dst);

        uint32_t tiles_done = 0;
        for (uint32_t k = 0; k < chunk_count; ++k) {
            const uint32_t tiles_in_chunk = std::min(tile_granularity, tile_count - tiles_done);
            cb_send.wait_front(tile_granularity);
            const uint32_t rd = cb_send.get_read_ptr();

            for (uint32_t t = 0; t < tiles_in_chunk; t += interm_tiles_per_packet) {
                const uint32_t tiles_in_pkt = std::min(tiles_in_chunk - t, interm_tiles_per_packet);
                const uint16_t payload = tiles_in_pkt * tile_bytes;
                uint64_t dst_noc;
                if constexpr (arrivals_in_cb) {
                    // Chunk-major within the half: the destination's CB read pointer advances one
                    // num_devices-block group per chunk, so consecutive chunks are a group apart.
                    dst_noc = safe_get_noc_addr(
                        peer_core_x, peer_core_y, block_base + k * num_devices * chunk_bytes + t * tile_bytes, 0);
                } else {
                    auto staging_acc = TensorAccessor(staging_args, staging_addr);  // chunk-paged
                    dst_noc =
                        tt::tt_fabric::addrgen_detail::get_noc_address(staging_acc, dst_page_base + k, t * tile_bytes);
                }
                const bool last_packet = (k == chunk_count - 1) && (t + interm_tiles_per_packet >= tiles_in_chunk);
                if (last_packet) {
                    fabric_api::fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<
                        UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::SemaphoreAddr |
                        UnicastFusedAtomicIncUpdateMask::PayloadSize>(
                        sender,
                        fused_pkt,
                        rd + t * tile_bytes,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc, arrival_noc_addr, 1u},
                        payload);
                } else {
                    fabric_api::fabric_unicast_noc_unicast_write_with_state<
                        UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                        sender,
                        data_pkt,
                        rd + t * tile_bytes,
                        tt::tt_fabric::NocUnicastCommandHeader{dst_noc},
                        payload);
                }
                noc.async_writes_flushed();
            }

            cb_send.pop_front(tile_granularity);
            tiles_done += tiles_in_chunk;
        }
    }

    // Reduced output slice -> tiled local output, per tile (contiguous page order).
    {
        uint32_t out_tile = tile_start;
        uint32_t tiles_done = 0;
        for (uint32_t k = 0; k < chunk_count; ++k) {
            const uint32_t tiles_in_chunk = std::min(tile_granularity, tile_count - tiles_done);
            cb_out.wait_front(tile_granularity);
            const uint32_t rd = cb_out.get_read_ptr();
            for (uint32_t t = 0; t < tiles_in_chunk; ++t) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(rd + t * tile_bytes), output_acc, tile_bytes, {}, {.page_id = out_tile}, {});
                ++out_tile;
            }
            noc.async_writes_flushed();
            cb_out.pop_front(tile_granularity);
            tiles_done += tiles_in_chunk;
        }
    }
    noc_semaphore_set(gen_ptr, invocation + 1);

    fabric_connection.close_start();

    // Via the Noc object so an explicitly-constructed `noc` cannot leave these on the default NoC.
    noc.async_write_barrier();
    noc.async_atomic_barrier();

    fabric_connection.close_finish();
}
