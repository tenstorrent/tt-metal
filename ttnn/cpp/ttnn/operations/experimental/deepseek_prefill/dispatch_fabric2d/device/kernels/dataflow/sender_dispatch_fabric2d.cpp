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
// the slot's tail to the metadata page. A relay hop stays one write. Under fan-out a slot is the last
// holder of several pages at once, so it issues that pair per destination and may still forward.
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

// Prebuilt headers per ring slot. Every send is a single hop, so the route is constant for the whole run
// and only the write address varies per token. One per packet in flight rather than one per slot because
// setting the next write's address would otherwise mutate the header a previous non-blocking send may
// still be reading, and that payload would land wherever the torn header pointed. Index 0 is the forward
// or the payload of a last hop, 1 the metadata beside it, and the pairs from 2 belong to the deliveries a
// fan-out slot makes into the next chip. The host reserves the pool from headers_per_slot as well.
volatile PACKET_HEADER_TYPE* slot_hdr(uint32_t slot, uint32_t which) {
    return reinterpret_cast<volatile PACKET_HEADER_TYPE*>(
        ct.pkt_hdr_ring_addr + (slot * dspf2d::headers_per_slot(ct.fanout) + which) * sizeof(PACKET_HEADER_TYPE));
}

volatile tt_l1_ptr dspf2d::FwdMetadata* slot_metadata(uint32_t slot) {
    return reinterpret_cast<volatile tt_l1_ptr dspf2d::FwdMetadata*>(
        ct.ring_addr + slot * ct.slot_stride() + ct.token_size_bytes);
}

// fanout: the same tail viewed as FanoutMetadata (the two layouts pin cmd/this_addr to the same
// offsets), and the deliveries the reader staged for this slot.
volatile tt_l1_ptr dspf2d::FanoutMetadata* slot_mc_metadata(uint32_t slot) {
    return reinterpret_cast<volatile tt_l1_ptr dspf2d::FanoutMetadata*>(
        ct.ring_addr + slot * ct.slot_stride() + ct.token_size_bytes);
}

volatile tt_l1_ptr dspf2d::FanoutDelivery* slot_delivery(uint32_t slot) {
    return reinterpret_cast<volatile tt_l1_ptr dspf2d::FanoutDelivery*>(
        ct.mc_delivery_addr + slot * dspf2d::FO_MAX_DESTS * sizeof(dspf2d::FanoutDelivery));
}

// A staged count as the sender is willing to act on it. The reader stages at most FO_MAX_DESTS records
// per slot and the header pool is sized for exactly that, so a larger value is a reader bug -- one that
// would run packet headers over the very delivery records these sends read their addresses from, and
// then send from whatever that wrote. ASSERT is compiled out on this hardware, so the bound is enforced
// here rather than checked.
uint32_t staged_count(uint32_t n, uint32_t budget) { return n < budget ? n : budget; }

// Write this slot's local deliveries out of it: the token to each destination page on THIS chip and
// its metadata words alongside. Plain NoC writes on this RISC's NoC, not fabric sends -- the point is
// to take them off the reader's port, which was the saturated one, onto this one, which was not.
// The batch flush that frees these slots is on this NoC too, so it already covers these.
void deliver_locally(uint32_t slot, uint32_t slot_base) {
    const uint32_t n = staged_count(slot_mc_metadata(slot)->local_count, dspf2d::FO_MAX_DESTS);
    volatile tt_l1_ptr dspf2d::FanoutDelivery* dl = slot_delivery(slot);
    for (uint32_t i = 0; i < n; i++) {
        noc_async_write(slot_base, dl[i].payload_addr, ct.token_size_bytes);
        noc_async_write(dl[i].meta_src, dl[i].meta_addr, dspf2d::METADATA_WIRE_BYTES);
    }
}

// Land one token and its metadata on the chip across this cable: two fabric packets out of one slot.
// The order inside is load-bearing -- the header is built BEFORE waiting for an EDM slot, because
// reversing the two costs about 8% of the bandwidth -- which is why the last hop of the unicast path
// and a fan-out delivery share it rather than each spelling it out.
template <typename FabricSender>
void send_payload_and_meta(
    FabricSender& fabric,
    uint64_t payload_addr,
    uint32_t payload_src,
    uint64_t meta_addr,
    uint32_t meta_src,
    volatile PACKET_HEADER_TYPE* hdr,
    volatile PACKET_HEADER_TYPE* hdr_meta) {
    hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{payload_addr}, ct.token_size_bytes);
    fabric.wait_for_empty_write_slot();
    fabric.send_payload_without_header_non_blocking_from_address(payload_src, ct.token_size_bytes);
    fabric.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));

    // Its own header, so setting this address cannot disturb the payload send above, and a source that
    // agrees with the destination modulo the transfer size: a NoC write from an unaligned source
    // arrives rotated by a word.
    hdr_meta->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{meta_addr}, dspf2d::METADATA_WIRE_BYTES);
    fabric.wait_for_empty_write_slot();
    fabric.send_payload_without_header_non_blocking_from_address(meta_src, dspf2d::METADATA_WIRE_BYTES);
    fabric.send_payload_flush_non_blocking_from_address((uint32_t)hdr_meta, sizeof(PACKET_HEADER_TYPE));
}

// The same records, for the destinations that live on the chip across this cable: the pages this slot
// is the last holder of. The addresses are usable here because every output buffer is interleaved DRAM
// with a mesh-uniform base, which is how the unicast tail's final addresses have always travelled.
//
// This is what one read feeding many sends buys: the page is never landed in the neighbour's region
// for it to read straight back out. It is not free on the cable -- a forward is ONE packet carrying
// the token and its tail together, while each delivery here is two -- so it trades one extra small
// packet per delivery against two DRAM transfers on the chip that would have landed the page.
template <typename FabricSender>
void deliver_remotely(FabricSender& fabric, uint32_t slot, uint32_t slot_base) {
    volatile tt_l1_ptr dspf2d::FanoutMetadata* metadata = slot_mc_metadata(slot);
    // The two lists share one FO_MAX_DESTS-record budget, so the remote share is what the local one
    // leaves: clamping each to FO_MAX_DESTS alone would still let `first + i` run off the end.
    const uint32_t first = staged_count(metadata->local_count, dspf2d::FO_MAX_DESTS);
    const uint32_t n = staged_count(metadata->remote_count, dspf2d::FO_MAX_DESTS - first);
    volatile tt_l1_ptr dspf2d::FanoutDelivery* dl = slot_delivery(slot);
    for (uint32_t i = 0; i < n; i++) {
        send_payload_and_meta(
            fabric,
            dl[first + i].payload_addr,
            slot_base,
            dl[first + i].meta_addr,
            dl[first + i].meta_src,
            slot_hdr(slot, dspf2d::FO_FIRST_DELIVERY_HDR + 2 * i),
            slot_hdr(slot, dspf2d::FO_FIRST_DELIVERY_HDR + 2 * i + 1));
    }
}

void prebuild_routes() {
    for (uint32_t slot = 0; slot < ct.num_l1_slots; slot++) {
        for (uint32_t which = 0; which < dspf2d::headers_per_slot(ct.fanout); which++) {
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
    // Listed positively rather than excluding the unicast commands: these three are what the fan-out
    // reader emits, and they are the ones whose tail is a FanoutMetadata with both counts written. A
    // FwdMetadata tail leaves those two words uninitialised (see the static_assert on their offset),
    // and they bound these loops, so anything else must not reach them. Either count may be zero --
    // what a slot delivers does not decide whether it also forwards.
    if (ct.fanout && (forwarding || cmd == dspf2d::CMD_NO_FORWARD)) {
        deliver_locally(slot, slot_base);
        deliver_remotely(fabric, slot, slot_base);
    }
    if (cmd == dspf2d::CMD_NO_FORWARD) {
        return cmd;  // its deliveries went out above; no PAGE follows them
    }

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
    } else if constexpr (!ct.fanout) {
        // Last hop: the token to its page, then the three metadata words to the same page index of the
        // metadata tensor. Both addresses were computed on the chip the token started from and travelled
        // with it, so this hop needs no address generator of its own. The metadata words sit at the very
        // start of the slot's tail, which is the one place in it a fabric write's source can be aligned.
        //
        // Compiled out under fan-out, which has no last hop of its own: there the same 64 bytes are a
        // FanoutMetadata, and `final_payload_addr` would be two packed destinations read as an address.
        send_payload_and_meta(
            fabric,
            metadata->final_payload_addr,
            slot_base,
            metadata->final_meta_addr,
            slot_base + ct.token_size_bytes,
            hdr,
            slot_hdr(slot, 1));
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
