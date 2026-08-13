// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Producer kernel (writer RISC, NOC_1). Owns the ONE fabric sender connection its eth channel allows (the
// L1 connection table is indexed by eth channel and the EDM stores a single worker_xy per channel, so a
// second core on the same channel would just hang) and drains the L1 ring the reader on this same core
// fills, one fabric packet per token. Every send is a single hop to the chip across this cable; tokens
// bound further are written into the next chip's forwarding buffer and re-sent from there.
//
// Slots are claimed and released in batches of `batch`, amortising the two counter bumps and the source
// flush. The flush matters because the ring is reused: a payload send reads L1 asynchronously, so a slot
// cannot go back to the reader until that read has drained. noc_async_writes_flushed() is exactly that
// guarantee and is cheaper than a barrier.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "combine_fabric2d_slot_tail.hpp"

// Forwarded tokens between semaphore bumps to the downstream reader. A bump always follows a sentinel
// regardless, so this only sets how finely that reader can pipeline within a chunk.
constexpr uint32_t FWD_BUMP_EVERY = 32;

void kernel_main() {
    constexpr uint32_t num_l1_slots = get_compile_time_arg_val(0);
    constexpr uint32_t token_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t slot_tail_bytes = get_compile_time_arg_val(2);
    // The immediate ring neighbour across this producer's own cable.
    constexpr uint32_t peer_chip_id = get_compile_time_arg_val(3);
    constexpr uint32_t peer_mesh_id = get_compile_time_arg_val(4);
    constexpr uint32_t peer_noc_x = get_compile_time_arg_val(5);
    constexpr uint32_t peer_noc_y = get_compile_time_arg_val(6);
    constexpr uint32_t ring_addr = get_compile_time_arg_val(7);
    constexpr uint32_t pkt_hdr_ring_addr = get_compile_time_arg_val(8);
    constexpr uint32_t pkt_hdr_drain_addr = get_compile_time_arg_val(9);
    // Uniform L1 address of the peer worker's 4-byte drain sink. Nothing reads it, which is the point.
    constexpr uint32_t drain_sink_addr = get_compile_time_arg_val(10);
    // Uniform across the mesh, so an accessor built from this chip's base produces addresses valid on any
    // destination chip.
    constexpr uint32_t dram_out_base_addr = get_compile_time_arg_val(11);
    // Ring handshake: the reader bumps `filled`, we bump `freed`. Both monotonic, single-writer.
    constexpr uint32_t batch = get_compile_time_arg_val(12);
    constexpr uint32_t filled_addr = get_compile_time_arg_val(13);
    constexpr uint32_t freed_addr = get_compile_time_arg_val(14);
    constexpr uint32_t my_noc_x = get_compile_time_arg_val(15);
    constexpr uint32_t my_noc_y = get_compile_time_arg_val(16);
    // The DOWNSTREAM chip's worker continuing in OUR direction on OUR plane — the reader that drains the
    // forwarding-buffer quarter we write. NOT the worker across our cable: that one's cable points back.
    constexpr uint32_t fwd_sem_noc_x = get_compile_time_arg_val(17);
    constexpr uint32_t fwd_sem_noc_y = get_compile_time_arg_val(18);
    constexpr uint32_t fwd_sem_addr = get_compile_time_arg_val(19);
    constexpr auto dram_out_args = TensorAccessorArgs<20>();
    constexpr uint32_t slot_stride = token_size_bytes + slot_tail_bytes;
    // cmd 2 also sends the tail's final address + dst chip, which sit immediately after the token, so the
    // two together are one contiguous run.
    constexpr uint32_t FWD_EXTRA_BYTES = 16;

    std::size_t rt_args_idx = 0;
    uint32_t num_connections = get_arg_val<uint32_t>(rt_args_idx++);
    auto fabric_connections = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION>(
        rt_args_idx, num_connections);
    auto& sender = fabric_connections.get(0).sender;

    // One prebuilt header per ring slot. Every send is a single hop, so the route is constant for the whole
    // run; only the write address varies per token, and a slot's header is untouched until the ring wraps.
    auto slot_hdr = [](uint32_t slot) -> volatile PACKET_HEADER_TYPE* {
        return reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_ring_addr + slot * sizeof(PACKET_HEADER_TYPE));
    };
    for (uint32_t slot = 0; slot < num_l1_slots; slot++) {
        fabric_set_unicast_route(
            (volatile tt::tt_fabric::HybridMeshPacketHeader*)slot_hdr(slot), peer_chip_id, peer_mesh_id);
    }
    // Shares the drain's scratch header: the drain only runs once the send loop is done with it.
    volatile PACKET_HEADER_TYPE* hdr_bump = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_drain_addr);
    fabric_set_unicast_route((volatile tt::tt_fabric::HybridMeshPacketHeader*)hdr_bump, peer_chip_id, peer_mesh_id);
    const uint64_t fwd_sem_noc = get_noc_addr(fwd_sem_noc_x, fwd_sem_noc_y, fwd_sem_addr);

    volatile tt_l1_ptr uint32_t* filled = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(filled_addr);
    const uint64_t my_freed_noc = get_noc_addr(my_noc_x, my_noc_y, freed_addr);
    const auto dram_out = TensorAccessor(dram_out_args, dram_out_base_addr);

    uint32_t sent = 0;
    // The stream's length is not known here: it is this producer's own tokens plus everything the reader
    // re-forwards for other chips, which depends on chunk sizes decided upstream. The reader terminates the
    // stream with a CMD_END slot instead, and we batch over whatever it has already published.
    bool end_of_stream = false;
    uint32_t fwd_since_bump = 0;
    while (!end_of_stream) {
        uint32_t avail = 0;
        while (true) {
            invalidate_l1_cache();
            avail = *filled - sent;
            if (avail > 0) {
                break;
            }
        }
        const uint32_t n = avail < batch ? avail : batch;

        uint32_t processed = 0;
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t slot = (sent + i) % num_l1_slots;
            const uint32_t slot_addr = ring_addr + slot * slot_stride;
            volatile tt_l1_ptr uint64_t* tail =
                reinterpret_cast<volatile tt_l1_ptr uint64_t*>(slot_addr + token_size_bytes);
            const uint64_t cmd = tail[TAIL_CMD];
            processed++;
            if (cmd == CMD_END) {
                end_of_stream = true;
                break;
            }
            const bool forwarding = (cmd == CMD_FORWARD);
            const uint32_t payload_bytes = forwarding ? (token_size_bytes + FWD_EXTRA_BYTES) : token_size_bytes;

            volatile PACKET_HEADER_TYPE* hdr = slot_hdr(slot);
            // Header first, THEN wait for the slot: building it while the EDM may still be busy is free
            // overlap, and reversing the two costs ~8% of the bandwidth.
            hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{tail[TAIL_THIS_ADDR]}, payload_bytes);
            sender.wait_for_empty_write_slot();
            sender.send_payload_without_header_non_blocking_from_address(slot_addr, payload_bytes);
            // No flush per token: a slot's header is untouched until the ring wraps and the payload is
            // flushed once per batch below, which is what lets token N+1 issue while N is still draining.
            // This leans on the NoC keeping the payload write ordered ahead of the EDM slot-credit write
            // that post_send_payload_increment_pointers issues on the sync cmd buf; the production idiom
            // flush-blocks here instead. A torn packet would surface as a wrong output token.
            sender.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));

            // Tell the downstream reader how far its quarter is filled. A sentinel always forces a bump: it
            // is the chunk boundary that reader switches on, so leaving it uncounted would strand it.
            if (forwarding) {
                fwd_since_bump++;
                if (tail[TAIL_DST_CHIP] == SENTINEL_DST_CHIP || fwd_since_bump >= FWD_BUMP_EVERY) {
                    // Header-only atomic inc, NOT the fused write+inc: that is documented to hang Blackhole
                    // when the payload destination is DRAM, and the forwarding buffer is DRAM.
                    hdr_bump->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                        fwd_sem_noc, /*val=*/fwd_since_bump, /*flush=*/true});
                    sender.wait_for_empty_write_slot();
                    sender.send_payload_flush_blocking_from_address((uint32_t)hdr_bump, sizeof(PACKET_HEADER_TYPE));
                    fwd_since_bump = 0;
                }
            }
        }
        // The batch's payload reads have drained out of L1, so these slots are safe to refill.
        noc_async_writes_flushed();
        sent += processed;
        noc_semaphore_inc(my_freed_noc, processed);
    }
    sent--;  // the CMD_END slot carried no payload

    // Delivery barrier. Program completion says nothing about whether our packets reached the DESTINATION
    // chip, so without this the host could read an output whose last tokens are still in flight.
    //
    // The worker's free-slot count is D = num_buffers_per_channel deep and satisfies
    // free = D - (packets_written - credits_returned), and a credit is only produced by the far end (the
    // router forwards what the remote receiver channel acked). So writing D-1 more packets and then
    // obtaining one further free slot forces credits_returned >= sent: every payload packet has reached the
    // destination chip. It does NOT prove the destination DRAM write retired — the far eRISC may ack on
    // write issue.
    //
    // The fillers are header-only atomic incs of value ZERO aimed at the peer worker's drain sink: real
    // fabric packets (there is no NOP send type) that change nothing. Their own completion is never
    // awaited. Reaching a free slot cannot deadlock, since the reverse direction is a different eth channel.
    if (sent > 0) {
        volatile PACKET_HEADER_TYPE* hdr_drain = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_drain_addr);
        const uint64_t peer_drain_sink_noc = get_noc_addr(peer_noc_x, peer_noc_y, drain_sink_addr);
        hdr_drain->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{peer_drain_sink_noc, /*val=*/0, /*flush=*/false});
        fabric_set_unicast_route(
            (volatile tt::tt_fabric::HybridMeshPacketHeader*)hdr_drain, peer_chip_id, peer_mesh_id);
        for (uint32_t d = 0; d + 1 < sender.num_buffers_per_channel; d++) {
            sender.wait_for_empty_write_slot();
            sender.send_payload_flush_blocking_from_address((uint32_t)hdr_drain, sizeof(PACKET_HEADER_TYPE));
        }
        sender.wait_for_empty_write_slot();
    }

    noc_async_writes_flushed();
    fabric_connections.close();
}
