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

// Brings the fabric linear-api headers + the `fabric_api` alias + PacketHeaderPool.
#include "cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_common.hpp"

using address_t = uint32_t;

// Direct (one-shot) reduce-scatter writer: owns the fabric, sends this core's tile range of local input
// slice j to device j for every remote j, then writes the locally reduced output slice.
//
// Every send is a MULTI-HOP unicast (num_hops = ring distance, direction = which of the two connections),
// so a contribution reaches its destination without any intermediate device staging/reducing it -- one
// fabric traversal instead of the ring's N/2 store-and-forward steps. That is the whole point of this op.
// Destinations are visited farthest-first (host-ordered), since the destination cannot start reducing
// until its last contribution lands.
//
// Every destination writes into the SAME place: staging slot `device_idx` (indexed by SOURCE, so no two
// senders collide), chunk sub-range [chunk_start, chunk_start + chunk_count), in the invocation-parity
// half. So only the route changes per destination -- the dst addresses are computed once per chunk.
// The arrival increment for a destination rides that destination's LAST payload packet (fused
// write+atomic-inc), targeting the mirror core's per-source arrival counter.
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
    // Start barrier: hold every send until all peers have entered this invocation. Needed only when the
    // op allocates its own buffers -- see the block that runs it, below.
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
    // Start-barrier counter (same address on every peer) and the two multicast hop ranges that between
    // them cover the ring exactly once. Always passed so the arg layout does not depend on the CT flag.
    const address_t init_sem = get_arg_val<uint32_t>(ai++);
    const uint8_t fwd_mcast_range = static_cast<uint8_t>(get_arg_val<uint32_t>(ai++));
    const uint8_t bwd_mcast_range = static_cast<uint8_t>(get_arg_val<uint32_t>(ai++));
    // Per-destination route, in send order (shared order with the reader's cb_send production).
    const size_t dest_conn = ai;
    ai += num_dests;
    const size_t dest_hops = ai;
    ai += num_dests;
    const size_t dest_block = ai;  // our block index inside that destination's reduce group (aliased path)
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

    // One header for plain payload packets, one for the fused write+atomic-inc that closes a destination.
    // Both are re-set_state'd per destination (that is where num_hops lives); safe because every packet is
    // flushed before the next one re-patches a header.
    auto data_pkt = PacketHeaderPool::allocate_header(1);
    auto fused_pkt = PacketHeaderPool::allocate_header(1);

    const uint64_t arrival_noc_addr = safe_get_noc_addr(peer_core_x, peer_core_y, arrival_sem, 0);

    // MEASURED: deferring open_finish past the header setup is worth nothing on its own (the handshake
    // already overlaps the reader's first input read, which runs on another RISC); the win on this axis came
    // from the SPLIT CLOSE at the end. Kept deferred anyway since it cannot hurt.
    fabric_connection.open_finish();

    // ---- Optional start barrier ----
    //
    // Our sends target the peer's staging buffer at OUR OWN staging address (the mesh allocator is
    // lockstep, so the buffer sits at the same address on every device). That identity holds only if the
    // peer is on the same invocation as us -- and the parity double-buffer deliberately tolerates one
    // invocation of skew. With persistent buffers that is harmless: the address is pinned for the cached
    // program's lifetime, so a one-invocation-ahead sender writes the other parity half of the SAME
    // buffer, which is exactly what the parity scheme is for. When the op allocates its own buffers the
    // address is only stable as long as the allocator happens to repeat itself; a sender that has moved
    // on to invocation i+1 would write at i+1's address into a peer still on i, where that address may
    // belong to some other live buffer entirely (or not be allocated yet on the very first invocation).
    //
    // So: when either buffer is op-allocated, hold every send until all N-1 peers have entered this
    // invocation. Two multicasts (one per direction, hop ranges chosen host-side to cover the ring
    // exactly once) instead of N-1 unicasts. The counter is never reset -- like the arrival counters, we
    // wait on an absolute position, so a peer that raced ahead cannot satisfy an earlier wait.
    if constexpr (needs_init_sync) {
        auto init_pkt = PacketHeaderPool::allocate_header(1);
        const uint64_t init_noc_addr = safe_get_noc_addr(peer_core_x, peer_core_y, init_sem, 0);
        // Connection 0 = forward, 1 = backward; a range is 0 only when that connection is not open.
        if (fwd_mcast_range > 0) {
            fabric_api::fabric_multicast_noc_unicast_atomic_inc(
                &fabric_connection.get(0).sender,
                init_pkt,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_noc_addr, 1u},
                /*start_distance=*/1,
                fwd_mcast_range);
            noc.async_writes_flushed();  // on the wire before the header is re-patched below
        }
        if (bwd_mcast_range > 0) {
            fabric_api::fabric_multicast_noc_unicast_atomic_inc(
                &fabric_connection.get(1).sender,
                init_pkt,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{init_noc_addr, 1u},
                /*start_distance=*/1,
                bwd_mcast_range);
            noc.async_writes_flushed();
        }
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

        uint32_t tiles_done = 0;
        for (uint32_t k = 0; k < chunk_count; ++k) {
            const uint32_t tiles_in_chunk = std::min(tile_granularity, tile_count - tiles_done);
            cb_send.wait_front(tile_granularity);
            const uint32_t rd = cb_send.get_read_ptr();

            // Contiguous chunk -> peer staging page (dst_page_base + k), split into full packets. Flush
            // after EACH packet: the header is re-patched per write, so a packet must be on the wire
            // before the next one reuses it.
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
                    // Closes this destination: in-order delivery on the connection means every earlier
                    // payload of ours has landed before the inc is applied.
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
            // Only needs the payload to have LEFT L1 before the slot is handed back to compute; the final
            // write barrier below still guarantees the writes actually completed before the op ends.
            noc.async_writes_flushed();
            cb_out.pop_front(tile_granularity);
            tiles_done += tiles_in_chunk;
        }
    }

    noc_semaphore_set(gen_ptr, invocation + 1);

    // Split close: start the teardown handshake, then drain our own barriers while it runs. ONE write
    // barrier is enough -- it is what actually completes the output writes (the loop above only flushed
    // them) and any fabric payload writes; nothing is issued after it, and close_finish covers the
    // teardown's own traffic. The extra trailing barrier this kernel used to carry (inherited from the
    // unicast writer) was a redundant round-trip wait on the critical path.
    fabric_connection.close_start();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    fabric_connection.close_finish();
}
