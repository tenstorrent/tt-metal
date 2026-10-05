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
#include "fabric/fabric_edm_packet_header.hpp"
#include "combine_fabric2d_sender_ct_args.hpp"

// Device-profiler build only: where the sender's time goes, summed over the run and reported once at the end,
// since a zone per token would overflow the profiler buffer. Compiles to nothing otherwise.
#if defined(PROFILE_KERNEL)
#include "tools/profiler/kernel_profiler.hpp"
namespace cmbf2d_prof {
uint64_t wait_filled = 0;  // the reader has not published a slot
uint64_t wait_edm = 0;     // no free fabric write slot: the router has not returned a credit
}  // namespace cmbf2d_prof
#define CMBF2D_PROF_NOW() get_timestamp()
#define CMBF2D_PROF_ADD(counter, t0) (cmbf2d_prof::counter += get_timestamp() - (t0))
#else
#define CMBF2D_PROF_NOW() 0ull
#define CMBF2D_PROF_ADD(counter, t0) ((void)(t0))
#endif

// Guard genuinely different BEHAVIOUR with CMBF2D_OVERLAPPED; anything that differs only in where a value
// lives belongs behind an accessor on the compile-time arguments instead.

// Forwarded tokens between semaphore bumps to the downstream reader. A chunk's last page forces a bump
// regardless, so this only sets how finely that reader can pipeline within a chunk.
constexpr uint32_t FWD_BUMP_EVERY = 32;
// Final writes between bumps of the downstream reader's receive count. Nothing waits on that count until
// the end of the stream, so this only bounds how many header-only packets it costs; the tail is bumped when
// the stream ends. The overlapped fork has no receive count for them yet, so only the standalone op keeps one.
constexpr uint32_t FINAL_BUMP_EVERY = 32;
constexpr ::cmbf2d::SenderCtArgs ct{};

// One prebuilt header per ring slot. Every send is a single hop, so the route is constant for the whole
// run; only the write address varies per token, and a slot's header is untouched until the ring wraps.
volatile PACKET_HEADER_TYPE* slot_hdr(uint32_t slot) {
    return reinterpret_cast<volatile PACKET_HEADER_TYPE*>(ct.pkt_hdr_ring_addr + slot * sizeof(PACKET_HEADER_TYPE));
}

volatile tt_l1_ptr ::cmbf2d::FwdMetadata* slot_metadata(uint32_t slot) {
    return reinterpret_cast<volatile tt_l1_ptr ::cmbf2d::FwdMetadata*>(
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
}

// Blocks until the reader has announced at least one slot beyond `sent`, then reports how many.
uint32_t wait_for_filled(uint32_t sent) {
    volatile tt_l1_ptr uint32_t* filled = ct.filled_ptr();
    while (true) {
        invalidate_l1_cache();
        const uint32_t avail = *filled - sent;
        if (avail > 0) {
            return avail;
        }
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
    [[maybe_unused]] const uint64_t prof_t0 = CMBF2D_PROF_NOW();
    fabric.wait_for_empty_write_slot();
    CMBF2D_PROF_ADD(wait_edm, prof_t0);
    fabric.send_payload_flush_blocking_from_address((uint32_t)hdr_bump, sizeof(PACKET_HEADER_TYPE));
}

// Put one slot's token on the cable. Returns its command word so the caller can spot the end of the stream.
template <typename FabricSender>
uint64_t send_slot(
    FabricSender& fabric, uint32_t slot, uint32_t& fwd_since_bump, [[maybe_unused]] uint32_t& final_since_bump) {
    volatile tt_l1_ptr ::cmbf2d::FwdMetadata* metadata = slot_metadata(slot);
    const uint64_t cmd = metadata->cmd;
    if (cmd == ::cmbf2d::CMD_END) {
        return cmd;
    }
    const bool forwarding = (cmd == ::cmbf2d::CMD_FORWARD || cmd == ::cmbf2d::CMD_FORWARD_END);
    const uint32_t payload_bytes = forwarding ? (ct.token_size_bytes + ::cmbf2d::FWD_EXTRA_BYTES) : ct.token_size_bytes;

    volatile PACKET_HEADER_TYPE* hdr = slot_hdr(slot);
    // Header first, THEN wait for the slot: building it while the EDM may still be busy is free overlap, and
    // reversing the two costs ~8% of the bandwidth.
    hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{metadata->this_addr}, payload_bytes);
    [[maybe_unused]] const uint64_t prof_t0 = CMBF2D_PROF_NOW();
    fabric.wait_for_empty_write_slot();
    CMBF2D_PROF_ADD(wait_edm, prof_t0);
    fabric.send_payload_without_header_non_blocking_from_address(ct.ring_addr + slot * ct.slot_stride(), payload_bytes);
    // No flush per token: a slot's header is untouched until the ring wraps and the payload is flushed once
    // per batch below, which is what lets token N+1 issue while N is still draining. Payload and credit go
    // out on different cmd bufs but share a source NIU, destination node and VC, so the NoC keeps the credit
    // behind the payload. This is the same non-blocking send all_gather and reduce_scatter take.
    fabric.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));

    if (forwarding) {
        fwd_since_bump++;
        if (cmd == ::cmbf2d::CMD_FORWARD_END || fwd_since_bump >= FWD_BUMP_EVERY) {
            bump_downstream(fabric, ct.fwd_sem_addr, fwd_since_bump);
            fwd_since_bump = 0;
        }
    }
#ifndef CMBF2D_OVERLAPPED
    else {
        final_since_bump++;
        if (final_since_bump >= FINAL_BUMP_EVERY) {
            bump_downstream(fabric, ct.final_sem_addr, final_since_bump);
            final_since_bump = 0;
        }
    }
#endif
    return cmd;
}

// Drain the ring until the reader ends the stream. Returns the number of tokens actually sent.
template <typename FabricSender>
uint32_t pump_stream(FabricSender& fabric) {
    const uint64_t my_freed_noc = get_noc_addr(reinterpret_cast<uint32_t>(ct.freed_ptr()));
    uint32_t sent = 0;
    // The stream's length is not known here: it is this sender's own tokens plus everything the reader
    // re-forwards for other chips, which depends on chunk sizes decided upstream. The reader terminates the
    // stream with a CMD_END slot instead, and we batch over whatever it has already published.
    bool end_of_stream = false;
    uint32_t fwd_since_bump = 0;
    uint32_t final_since_bump = 0;
    while (!end_of_stream) {
        [[maybe_unused]] const uint64_t prof_t0 = CMBF2D_PROF_NOW();
        const uint32_t avail = wait_for_filled(sent);
        CMBF2D_PROF_ADD(wait_filled, prof_t0);
        const uint32_t n = avail < ct.batch ? avail : ct.batch;

        uint32_t processed = 0;
        for (uint32_t i = 0; i < n; i++) {
            processed++;
            if (send_slot(fabric, (sent + i) % ct.num_l1_slots, fwd_since_bump, final_since_bump) ==
                ::cmbf2d::CMD_END) {
                end_of_stream = true;
                break;
            }
        }
        // The batch's payload reads have drained out of L1, so these slots are safe to refill.
        noc_async_writes_flushed();
        sent += processed;
        noc_semaphore_inc(my_freed_noc, processed);
    }
#ifndef CMBF2D_OVERLAPPED
    // The receive count the downstream reader waits on before it exits; it needs every final write counted.
    if (final_since_bump > 0) {
        bump_downstream(fabric, ct.final_sem_addr, final_since_bump);
    }
#endif
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
#ifdef CMBF2D_IDLE
    // Measurement mode: the routed expert runs as overlapped, with no combine traffic beside it.
    return;
#endif
    std::size_t rt_args_idx = 0;
    uint32_t num_connections = get_arg_val<uint32_t>(rt_args_idx++);
    auto fabric_connections = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION>(
        rt_args_idx, num_connections);
    auto& fabric = fabric_connections.get(0).sender;

    prebuild_routes();
    uint32_t sent = 0;
    {
        DeviceZoneScopedN("CMBF2D_S_PUMP");
        sent = pump_stream(fabric);
    }
    if (sent > 0) {
        DeviceZoneScopedN("CMBF2D_S_DRAIN");
        drain_fabric(fabric);
    }
#if defined(PROFILE_KERNEL)
    DeviceTimestampedData("CMBF2D_S_WAIT_FILLED", cmbf2d_prof::wait_filled);
    DeviceTimestampedData("CMBF2D_S_WAIT_EDM", cmbf2d_prof::wait_edm);
    DeviceTimestampedData("CMBF2D_S_TOKENS", sent);
#endif

    noc_async_writes_flushed();
    fabric_connections.close();

    // The last `freed` bump is a NoC atomic, which completes on the atomic response and so is NOT covered
    // by noc_async_writes_flushed above. It must land before the kernel exits.
    noc_async_atomic_barrier();
    // Whether the counters need zeroing depends on where they live, which is the owning op's choice.
    ct.reset_ring_counters();
}
