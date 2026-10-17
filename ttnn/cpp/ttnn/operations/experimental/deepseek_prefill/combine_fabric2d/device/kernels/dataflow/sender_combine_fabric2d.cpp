// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sender kernel (writer RISC, NOC_1). Owns the ONE fabric sender connection its eth channel allows (the
// L1 connection table is indexed by eth channel and the EDM stores a single worker_xy per channel, so a
// second core on the same channel would just hang) and drains the L1 ring the reader on this same core
// fills, one fabric packet per token. Every send is a single hop to the chip across this cable; tokens
// bound further are written into the next chip's forwarding buffer and re-sent from there.
//
// Slots are claimed and released in batches, amortising the two counter bumps and the source
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
#include "hostdev/fabric_edm_packet_header.hpp"
#include "combine_fabric2d_sender_ct_args.hpp"

// Forwarded tokens between semaphore bumps to the downstream reader. A chunk's last page forces a bump
// regardless, so this only sets how finely that reader can pipeline within a chunk.
constexpr uint32_t FWD_BUMP_EVERY = 32;
// Final writes between bumps of the downstream reader's receive count. Nothing waits on that count until
// the end of the stream, so this only bounds how many header-only packets it costs; the tail is bumped when
// the stream ends.
constexpr uint32_t FINAL_BUMP_EVERY = 32;
constexpr cmbf2d::SenderCtArgs ct{};

// One prebuilt header per ring slot. Every send is a single hop, so the route is constant for the whole
// run; only the write address varies per token, and a slot's header is untouched until the ring wraps.
volatile PACKET_HEADER_TYPE* slot_hdr(uint32_t slot) {
    return reinterpret_cast<volatile PACKET_HEADER_TYPE*>(ct.pkt_hdr_ring_addr + slot * sizeof(PACKET_HEADER_TYPE));
}

volatile tt_l1_ptr cmbf2d::FwdMetadata* slot_metadata(uint32_t slot) {
    return reinterpret_cast<volatile tt_l1_ptr cmbf2d::FwdMetadata*>(
        ct.ring_addr + slot * ct.slot_stride() + ct.token_size_bytes);
}

void prebuild_routes() {
    for (uint32_t slot = 0; slot < ct.num_l1_slots; slot++) {
        fabric_set_unicast_route(
            (volatile tt::tt_fabric::HybridMeshPacketHeader*)slot_hdr(slot), ct.peer_chip_id, ct.peer_mesh_id);
    }
    // Shares the drain's scratch header: the drain only runs once the send loop is done with it.
    fabric_set_unicast_route(
        reinterpret_cast<volatile tt::tt_fabric::HybridMeshPacketHeader*>(ct.pkt_hdr_drain_addr),
        ct.peer_chip_id,
        ct.peer_mesh_id);
    // Credits go to a different CORE on the same chip this sender already talks to, so the route is the
    // same one and is built once here too.
    fabric_set_unicast_route(
        reinterpret_cast<volatile tt::tt_fabric::HybridMeshPacketHeader*>(ct.pkt_hdr_credit_addr),
        ct.peer_chip_id,
        ct.peer_mesh_id);
}

// Defined below, next to the other counter bumps; the wait needs it to announce what it has already sent
// before it goes idle.
template <typename FabricSender>
void bump_downstream(FabricSender& fabric, uint32_t sem_addr, uint32_t count);

// Courier for the OPPOSITE direction's reader on this chip, one queue per channel. That reader frees
// pages of regions the chip across our cable wrote, so the credit is owed to exactly the chip we already
// point at -- it just lands on the other direction's worker there rather than ours.
//
// Its own header and its own running totals, so it never disturbs this stream's tokens: a credit can be
// posted from wherever we happen to be, including from inside the wait below.
struct Courier {
    uint32_t posted[cmbf2d::MAX_VCS] = {};

    // Post whatever the opposite reader has queued for each channel since the last call. Monotonic
    // counters minus our own totals, so there is no read-modify-write racing that reader.
    template <typename FabricSender>
    void service(FabricSender& fabric) {
        for (uint32_t v = 0; v < ct.num_vcs; v++) {
            volatile tt_l1_ptr uint32_t* outbox =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.vc_outbox_addr(v));
            invalidate_l1_cache();
            const uint32_t n = *outbox - posted[v];
            if (n == 0) {
                continue;
            }
            volatile PACKET_HEADER_TYPE* hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(ct.pkt_hdr_credit_addr);
            hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                get_noc_addr(ct.credit_noc_x, ct.credit_noc_y, ct.vc_freed_addr(v)), /*val=*/n, /*flush=*/true});
            fabric.wait_for_empty_write_slot();
            fabric.send_payload_flush_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
            posted[v] += n;
        }
    }

    // Keep couriering until the opposite reader says it will queue nothing more. It raises `credit_done`
    // only after its last credit is in an outbox, so one more pass after seeing the flag drains them.
    //
    // This runs after our own stream has ended, which is the point of it: the chip we credit may still be
    // waiting on those pages, and a kernel that exited at its own CMD_END would strand them.
    template <typename FabricSender>
    void finish(FabricSender& fabric) {
        volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.credit_done_addr);
        while (true) {
            service(fabric);
            invalidate_l1_cache();
            if (*done != 0) {
                break;
            }
        }
        service(fabric);
    }
};

Courier courier;

// Blocks until the reader has announced at least one slot beyond `sent`, then reports how many.
//
// Credits are serviced inside the spin, not only between batches. That is what keeps the two directions
// independent: a clockwise sender idle on its own reader would otherwise stop carrying the anticlockwise
// reader's credits, that ring would stall for want of them, and the two would hold each other up.
template <typename FabricSender>
uint32_t wait_for_filled(FabricSender& fabric, uint32_t sent, uint32_t* fwd_since_bump) {
    volatile tt_l1_ptr uint32_t* filled = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.filled_addr);
    bool announced = false;
    while (true) {
        invalidate_l1_cache();
        const uint32_t avail = *filled - sent;
        if (avail > 0) {
            return avail;
        }
        // Nothing of ours to send, so announce what we already sent. A channel is only vc_pages deep while
        // the bump batch is FWD_BUMP_EVERY, so a producer that fills a channel without completing a batch
        // would never tell the consumer those pages are there -- and that consumer is the only source of
        // the credits the producer is now waiting on. Deadlock, and exactly the one CMD_FORWARD_END used
        // to rule out when a chunk's last page forced a bump.
        //
        // Once per idle spell rather than per spin, or announcing becomes the busy-wait.
        if (!announced) {
            for (uint32_t v = 0; v < ct.num_vcs; v++) {
                if (fwd_since_bump[v] > 0) {
                    bump_downstream(fabric, ct.vc_arrived_addr(v), fwd_since_bump[v]);
                    fwd_since_bump[v] = 0;
                }
            }
            announced = true;
        }
        courier.service(fabric);
    }
}

// Add `count` to a semaphore on the downstream reader's core: `fwd_arrived` for pages of its forwarding
// region, `final_arrived` for tokens written straight into its chip's output. A chunk's last page always
// forces a `fwd_arrived` bump: that reader consumes a whole chunk before moving on, so leaving the tail
// uncounted would strand it. The flush makes the far router drain the writes ahead of the bump first.
template <typename FabricSender>
void bump_downstream(FabricSender& fabric, uint32_t sem_addr, uint32_t count) {
    volatile PACKET_HEADER_TYPE* hdr_bump = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(ct.pkt_hdr_drain_addr);
    // Header-only atomic inc, NOT the fused write+inc: that is documented to hang Blackhole when the payload
    // destination is DRAM, and the forwarding buffer is DRAM.
    hdr_bump->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        get_noc_addr(ct.fwd_sem_noc_x, ct.fwd_sem_noc_y, sem_addr), /*val=*/count, /*flush=*/true});
    fabric.wait_for_empty_write_slot();
    fabric.send_payload_flush_blocking_from_address((uint32_t)hdr_bump, sizeof(PACKET_HEADER_TYPE));
}

// Put one slot's token on the cable. Returns its command word so the caller can spot the end of the stream.
template <typename FabricSender>
uint64_t send_slot(FabricSender& fabric, uint32_t slot, uint32_t* fwd_since_bump, uint32_t& final_since_bump) {
    volatile tt_l1_ptr cmbf2d::FwdMetadata* metadata = slot_metadata(slot);
    const uint64_t cmd = metadata->cmd;
    if (cmd == cmbf2d::CMD_END) {
        return cmd;
    }
    const bool forwarding = (cmd == cmbf2d::CMD_FORWARD || cmd == cmbf2d::CMD_FORWARD_END);
    const uint32_t payload_bytes = forwarding ? (ct.token_size_bytes + cmbf2d::FWD_EXTRA_BYTES) : ct.token_size_bytes;

    volatile PACKET_HEADER_TYPE* hdr = slot_hdr(slot);
    // Header first, THEN wait for the slot: building it while the EDM may still be busy is free overlap, and
    // reversing the two costs ~8% of the bandwidth.
    hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{metadata->this_addr}, payload_bytes);
    fabric.wait_for_empty_write_slot();
    fabric.send_payload_without_header_non_blocking_from_address(ct.ring_addr + slot * ct.slot_stride(), payload_bytes);
    // No flush per token: a slot's header is untouched until the ring wraps and the payload is flushed once
    // per batch below, which is what lets token N+1 issue while N is still draining. Payload and credit go
    // out on different cmd bufs but share a source NIU, destination node and VC, so the NoC keeps the credit
    // behind the payload. This is the same non-blocking send all_gather and reduce_scatter take.
    fabric.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));

    if (forwarding) {
        // Per channel, because the downstream reader waits on each channel separately -- that is what
        // lets it serve a channel it can drain while another is blocked.
        const uint32_t vc = (uint32_t)metadata->vc;
        fwd_since_bump[vc]++;
        if (fwd_since_bump[vc] >= FWD_BUMP_EVERY) {
            bump_downstream(fabric, ct.vc_arrived_addr(vc), fwd_since_bump[vc]);
            fwd_since_bump[vc] = 0;
        }
    } else {
        final_since_bump++;
        if (final_since_bump >= FINAL_BUMP_EVERY) {
            bump_downstream(fabric, ct.final_sem_addr, final_since_bump);
            final_since_bump = 0;
        }
    }
    return cmd;
}

// Drain the ring until the reader ends the stream. Returns the number of tokens actually sent.
template <typename FabricSender>
uint32_t pump_stream(FabricSender& fabric) {
    const uint64_t my_freed_noc = get_noc_addr(ct.freed_addr);
    uint32_t sent = 0;
    // The stream's length is not known here: it is this sender's own tokens plus everything the reader
    // re-forwards for other chips, which depends on chunk sizes decided upstream. The reader terminates the
    // stream with a CMD_END slot instead, and we batch over whatever it has already published.
    bool end_of_stream = false;
    uint32_t fwd_since_bump[cmbf2d::MAX_VCS] = {};
    uint32_t final_since_bump = 0;
    while (!end_of_stream) {
        const uint32_t avail = wait_for_filled(fabric, sent, fwd_since_bump);
        const uint32_t n = avail < ct.batch ? avail : ct.batch;

        uint32_t processed = 0;
        for (uint32_t i = 0; i < n; i++) {
            processed++;
            if (send_slot(fabric, (sent + i) % ct.num_l1_slots, fwd_since_bump, final_since_bump) == cmbf2d::CMD_END) {
                end_of_stream = true;
                break;
            }
        }
        // The batch's payload reads have drained out of L1, so these slots are safe to refill.
        noc_async_writes_flushed();
        sent += processed;
        noc_semaphore_inc(my_freed_noc, processed);
        courier.service(fabric);
    }
    // Residual arrival counts. A reader blocks until its channel's count passes what it has consumed, so a
    // tail left inside a partial bump batch would strand it with the stream already over.
    for (uint32_t v = 0; v < ct.num_vcs; v++) {
        if (fwd_since_bump[v] > 0) {
            bump_downstream(fabric, ct.vc_arrived_addr(v), fwd_since_bump[v]);
        }
    }
    // The receive count the downstream reader waits on before it exits; it needs every final write counted.
    if (final_since_bump > 0) {
        bump_downstream(fabric, ct.final_sem_addr, final_since_bump);
    }
    return sent - 1;  // the CMD_END slot carried no payload
}

// Delivery barrier. Program completion says nothing about whether our packets reached the DESTINATION chip,
// so without this the host could read an output whose last tokens are still in flight.
//
// The worker's free-slot count is D = num_buffers_per_channel deep and satisfies
// free = D - (packets_written - credits_returned), and a credit is only produced by the far end (the router
// forwards what the remote receiver channel acked). So writing D-1 more packets and then obtaining one
// further free slot forces credits_returned >= sent: every payload packet has reached the destination chip.
// It does NOT prove the destination DRAM write retired — the far eRISC may ack on write issue.
//
// The fillers are header-only atomic incs of value ZERO aimed at a drain sink on the peer chip: real fabric
// packets (there is no NOP send type) that change nothing. Their own completion is never awaited. Reaching a
// free slot cannot deadlock, since the reverse direction is a different eth channel.
template <typename FabricSender>
void drain_fabric(FabricSender& fabric) {
    volatile PACKET_HEADER_TYPE* hdr_drain = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(ct.pkt_hdr_drain_addr);
    // Any legal L1 address on the chip across our cable will do; the downstream worker we already address for
    // semaphore bumps sits on exactly that chip.
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
    // Before the drain, which fills every free write slot the channel has and would leave no room to post
    // a credit until it is done.
    courier.finish(fabric);
    if (sent > 0) {
        drain_fabric(fabric);
    }

    noc_async_writes_flushed();
    fabric_connections.close();

    // Both ring counters back to zero for the next launch, which starts its own counts at zero. Safe here
    // and only here: the reader's last act was publishing the CMD_END slot this kernel has just drained, so
    // nothing is still reading or bumping either of them.
    //
    // `freed` is bumped by a NoC atomic, which completes on the atomic response and so is not covered by
    // noc_async_writes_flushed above. Without this the reset can be overtaken and the launch end with
    // freed == processed, leaving the next launch to evaluate claimed - freed as a negative wrap.
    noc_async_atomic_barrier();
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.filled_addr), 0);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.freed_addr), 0);
    // The courier's counters, zeroed for the same reason and with the same safety: they never leave this
    // chip, and the opposite reader that writes them has already raised credit_done, which is the last
    // thing it does to any of them.
    for (uint32_t v = 0; v < ct.num_vcs; v++) {
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.vc_outbox_addr(v)), 0);
    }
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.credit_done_addr), 0);
}
