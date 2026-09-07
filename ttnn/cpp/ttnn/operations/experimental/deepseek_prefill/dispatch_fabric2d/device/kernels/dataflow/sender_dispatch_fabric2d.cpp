// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sender kernel (writer RISC, NOC_1). Owns the ONE fabric sender connection its eth channel allows (the
// L1 connection table is indexed by eth channel and the EDM stores a single worker_xy per channel, so a
// second core on the same channel would just hang) and drains the L1 ring the reader on this same core
// fills. Every send is a single hop to the chip across this cable; tokens bound further are written into
// the next chip's forwarding buffer and re-sent from there.
//
// Where this differs from the combine sender: dispatch lands a token in TWO tensors at one page index, so
// the last hop is two writes out of one slot -- the payload to the token page and the metadata bytes from
// the slot's tail to the metadata page. A relay hop stays one write.
//
// Slots are claimed and released in batches, amortising the two counter bumps and the source flush. The
// flush matters because the ring is reused: a payload send reads L1 asynchronously, so a slot cannot go
// back to the reader until that read has drained.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "dispatch_fabric2d_sender_ct_args.hpp"

// Forwarded pages between semaphore bumps to the downstream reader. A bump always follows a chunk's last
// page regardless, so this only sets how finely that reader can pipeline within a chunk.
constexpr uint32_t FWD_BUMP_EVERY = 32;
constexpr dspf2d::SenderCtArgs ct{};

// TWO prebuilt headers per ring slot. Every send is a single hop, so the route is constant for the whole
// run and only the write address varies per token. Two rather than one because the last hop issues two
// writes out of a single slot: with one header, setting the second write's address would mutate the header
// the first write's non-blocking send may still be reading, and the payload would land wherever the torn
// header pointed.
volatile PACKET_HEADER_TYPE* slot_hdr(uint32_t slot, uint32_t which) {
    return reinterpret_cast<volatile PACKET_HEADER_TYPE*>(
        ct.pkt_hdr_ring_addr + (slot * 2 + which) * sizeof(PACKET_HEADER_TYPE));
}

volatile tt_l1_ptr dspf2d::FwdMetadata* slot_metadata(uint32_t slot) {
    return reinterpret_cast<volatile tt_l1_ptr dspf2d::FwdMetadata*>(
        ct.ring_addr + slot * ct.slot_stride() + ct.token_size_bytes);
}

void prebuild_routes() {
    for (uint32_t slot = 0; slot < ct.num_l1_slots; slot++) {
        for (uint32_t which = 0; which < 2; which++) {
            fabric_set_unicast_route(
                (volatile tt::tt_fabric::HybridMeshPacketHeader*)slot_hdr(slot, which),
                ct.peer_chip_id,
                ct.peer_mesh_id);
        }
    }
    // Shares the drain's scratch header: the drain only runs once the send loop is done with it.
    fabric_set_unicast_route(
        reinterpret_cast<volatile tt::tt_fabric::HybridMeshPacketHeader*>(ct.pkt_hdr_drain_addr),
        ct.peer_chip_id,
        ct.peer_mesh_id);
}

// Blocks until the reader has announced at least one slot beyond `sent`, then reports how many.
uint32_t wait_for_filled(uint32_t sent) {
    volatile tt_l1_ptr uint32_t* filled = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.filled_addr);
    while (true) {
        invalidate_l1_cache();
        const uint32_t avail = *filled - sent;
        if (avail > 0) {
            return avail;
        }
    }
}

// Tell the downstream reader how far its region is filled. A chunk's last page always forces a bump: it
// is the boundary that reader switches on, so leaving it uncounted strands the whole axis.
template <typename FabricSender>
void bump_downstream(FabricSender& fabric, uint32_t count) {
    volatile PACKET_HEADER_TYPE* hdr_bump = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(ct.pkt_hdr_drain_addr);
    // Header-only atomic inc, NOT the fused write+inc: that is documented to hang Blackhole when the
    // payload destination is DRAM, and the forwarding buffer is DRAM.
    hdr_bump->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        get_noc_addr(ct.fwd_sem_noc_x, ct.fwd_sem_noc_y, ct.fwd_sem_addr), /*val=*/count, /*flush=*/true});
    fabric.wait_for_empty_write_slot();
    fabric.send_payload_flush_blocking_from_address((uint32_t)hdr_bump, sizeof(PACKET_HEADER_TYPE));
}

// Put one slot's token on the cable. Returns its command word so the caller can spot the end of stream.
template <typename FabricSender>
uint64_t send_slot(FabricSender& fabric, uint32_t slot, uint32_t& fwd_since_bump) {
    volatile tt_l1_ptr dspf2d::FwdMetadata* metadata = slot_metadata(slot);
    const uint64_t cmd = metadata->cmd;
    if (cmd == dspf2d::CMD_END) {
        return cmd;
    }
    const bool forwarding = (cmd == dspf2d::CMD_FORWARD) || (cmd == dspf2d::CMD_FORWARD_END);
    const uint32_t slot_base = ct.ring_addr + slot * ct.slot_stride();

    volatile PACKET_HEADER_TYPE* hdr = slot_hdr(slot, 0);
    if (forwarding) {
        // One write: token plus the prefix of the tail the next hop needs, landing in its forwarding page.
        const uint32_t payload_bytes = ct.token_size_bytes + dspf2d::FWD_EXTRA_BYTES;
        // Header first, THEN wait for the slot: building it while the EDM may still be busy is free
        // overlap, and reversing the two costs ~8% of the bandwidth.
        hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{metadata->this_addr}, payload_bytes);
        fabric.wait_for_empty_write_slot();
        fabric.send_payload_without_header_non_blocking_from_address(slot_base, payload_bytes);
        fabric.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));

        fwd_since_bump++;
        if (cmd == dspf2d::CMD_FORWARD_END || fwd_since_bump >= FWD_BUMP_EVERY) {
            bump_downstream(fabric, fwd_since_bump);
            fwd_since_bump = 0;
        }
    } else {
        // Last hop: the token to its page, then the three metadata words to the same page index of the
        // metadata tensor. Both addresses were computed on the chip the token started from and travelled
        // with it, so this hop needs no address generator of its own.
        hdr->to_noc_unicast_write(
            tt::tt_fabric::NocUnicastCommandHeader{metadata->final_payload_addr}, ct.token_size_bytes);
        fabric.wait_for_empty_write_slot();
        fabric.send_payload_without_header_non_blocking_from_address(slot_base, ct.token_size_bytes);
        fabric.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));

        // The metadata words live in the slot's tail. Sending them from there keeps the reader's staging
        // to one place. Its own header, so setting this address cannot disturb the payload send above.
        volatile PACKET_HEADER_TYPE* hdr_meta = slot_hdr(slot, 1);
        const uint32_t meta_src = slot_base + ct.token_size_bytes + offsetof(dspf2d::FwdMetadata, meta);
        hdr_meta->to_noc_unicast_write(
            tt::tt_fabric::NocUnicastCommandHeader{metadata->final_meta_addr}, ct.metadata_page_bytes);
        fabric.wait_for_empty_write_slot();
        fabric.send_payload_without_header_non_blocking_from_address(meta_src, ct.metadata_page_bytes);
        fabric.send_payload_flush_non_blocking_from_address((uint32_t)hdr_meta, sizeof(PACKET_HEADER_TYPE));
    }
    return cmd;
}

// Drain the ring until the reader ends the stream. Returns the number of tokens actually sent.
template <typename FabricSender>
uint32_t pump_stream(FabricSender& fabric) {
    const uint64_t my_freed_noc = get_noc_addr(ct.freed_addr);
    uint32_t sent = 0;
    // The stream's length is not known here: it is this chip's own tokens plus everything the reader
    // re-forwards for other chips, which depends on chunk sizes decided upstream. The reader terminates
    // the stream with a CMD_END slot instead, and we batch over whatever it has already published.
    bool end_of_stream = false;
    uint32_t fwd_since_bump = 0;
    while (!end_of_stream) {
        const uint32_t avail = wait_for_filled(sent);
        const uint32_t n = avail < ct.batch ? avail : ct.batch;

        uint32_t processed = 0;
        for (uint32_t i = 0; i < n; i++) {
            processed++;
            if (send_slot(fabric, (sent + i) % ct.num_l1_slots, fwd_since_bump) == dspf2d::CMD_END) {
                end_of_stream = true;
                break;
            }
        }
        // The batch's payload reads have drained out of L1, so these slots are safe to refill.
        noc_async_writes_flushed();
        sent += processed;
        noc_semaphore_inc(my_freed_noc, processed);
    }
    return sent - 1;  // the CMD_END slot carried no payload
}

// Delivery barrier. Program completion says nothing about whether our packets reached the DESTINATION
// chip, so without this the host could read an output whose last tokens are still in flight.
//
// The worker's free-slot count is D = num_buffers_per_channel deep and satisfies
// free = D - (packets_written - credits_returned), and a credit is only produced by the far end. So
// writing D-1 more packets and then obtaining one further free slot forces credits_returned >= sent:
// every payload packet has reached the destination chip. It does NOT prove the destination DRAM write
// retired -- the far eRISC may ack on write issue.
//
// The fillers are header-only atomic incs of value ZERO aimed at a drain sink on the peer chip: real
// fabric packets (there is no NOP send type) that change nothing. Reaching a free slot cannot deadlock,
// since the reverse direction is a different eth channel.
template <typename FabricSender>
void drain_fabric(FabricSender& fabric) {
    volatile PACKET_HEADER_TYPE* hdr_drain = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(ct.pkt_hdr_drain_addr);
    hdr_drain->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        get_noc_addr(ct.fwd_sem_noc_x, ct.fwd_sem_noc_y, ct.drain_sink_addr), /*val=*/0, /*flush=*/false});
    fabric_set_unicast_route(
        (volatile tt::tt_fabric::HybridMeshPacketHeader*)hdr_drain, ct.peer_chip_id, ct.peer_mesh_id);
    for (uint32_t d = 0; d + 1 < fabric.num_buffers_per_channel; d++) {
        fabric.wait_for_empty_write_slot();
        fabric.send_payload_flush_blocking_from_address((uint32_t)hdr_drain, sizeof(PACKET_HEADER_TYPE));
    }
    fabric.wait_for_empty_write_slot();
}

void kernel_main() {
    std::size_t rt_args_idx = 0;
    uint32_t num_connections = get_arg_val<uint32_t>(rt_args_idx++);
    auto fabric_connections = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION>(
        rt_args_idx, num_connections);
    auto& fabric = fabric_connections.get(0).sender;

    prebuild_routes();
    const uint32_t sent = pump_stream(fabric);
    if (sent > 0) {
        drain_fabric(fabric);
    }

    noc_async_writes_flushed();
    fabric_connections.close();

    // Both ring counters back to zero for the next launch, which starts its own counts at zero. Safe here
    // and only here: the reader's last act was publishing the CMD_END slot this kernel has just drained,
    // so nothing is still reading or bumping either of them.
    //
    // `freed` is bumped by a NoC atomic, which completes on the atomic response and so is not covered by
    // noc_async_writes_flushed above. Without this the reset can be overtaken and the launch end with
    // freed == processed, leaving the next launch to evaluate claimed - freed as a negative wrap.
    noc_async_atomic_barrier();
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.filled_addr), 0);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.freed_addr), 0);
}
