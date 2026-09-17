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
// the last hop is a two-chunk scatter write out of one slot -- the token to its page and the metadata
// bytes behind it to the metadata page, one packet. A relay hop is a plain write. Under fan-out a slot
// is the last holder of several pages at once, so it issues one such packet per destination and may
// still forward.
//
// Slots are claimed and released in batches, amortising the two counter bumps and the source flush. The
// flush matters because the ring is reused: a payload send reads L1 asynchronously, so a slot cannot go
// back to the reader until that read has drained.

#include <cstdint>
#include <type_traits>
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
// still be reading, and that payload would land wherever the torn header pointed. The index map and the
// pool size are headers_per_slot in the kernel interface, which the host reserves the pool from as well.
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

// The bytes a terminal delivery carries: the token and, right behind it, the metadata words for the
// same page index.
constexpr uint32_t TOKEN_PLUS_META_BYTES = ct.token_size_bytes + dspf2d::METADATA_WIRE_BYTES;

// A token and its metadata land on the chip across this cable as ONE fabric packet: a scatter write
// whose first chunk is the token and whose second is the metadata bytes behind it in the payload. The
// two chunks land at unrelated addresses -- the payload page and the metadata page -- but leave here
// as one payload: one EDM slot, one header and one credit round trip per delivery, the same as a
// forward. `fabric_unicast_noc_scatter_write` in linear/api.h is these same steps with the route set
// on every call; here the route is constant for the run and prebuilt, so the header write is
// address-only.
void build_token_meta_header(volatile PACKET_HEADER_TYPE* hdr, uint64_t payload_addr, uint64_t meta_addr) {
    // Chunk and payload sizes are 16-bit header fields; the total is the larger and overflows first.
    static_assert(TOKEN_PLUS_META_BYTES <= 0xFFFFu, "a scatter payload size is a 16-bit field");
    // Only the first chunk's size is spelled; the second is the remainder of the payload, and the
    // router reads it from the payload base plus the first size. A NoC write whose source disagrees
    // with its destination modulo 16 arrives rotated by a word: (src, token, slot) reads back as
    // (junk, src, token).
    static_assert(ct.token_size_bytes % 16u == 0u, "the second scatter chunk is sourced at payload base + token size");
    hdr->to_noc_unicast_scatter_write(
        tt::tt_fabric::NocUnicastScatterCommandHeader{
            {payload_addr, meta_addr}, {static_cast<uint16_t>(ct.token_size_bytes)}},
        TOKEN_PLUS_META_BYTES);
}

// The unicast last hop. `src` is a ring slot, whose tail begins with the metadata words (FwdMetadata
// pins them at offset 0), so the whole payload is already laid out in L1 and goes out as it is. The
// order is load-bearing -- the header is built BEFORE waiting for an EDM slot, because reversing the
// two costs about 8% of the bandwidth.
template <typename FabricSender>
void send_token_with_inline_meta(
    FabricSender& fabric, uint64_t payload_addr, uint64_t meta_addr, uint32_t src, volatile PACKET_HEADER_TYPE* hdr) {
    build_token_meta_header(hdr, payload_addr, meta_addr);
    fabric.wait_for_empty_write_slot();
    fabric.send_payload_without_header_non_blocking_from_address(src, TOKEN_PLUS_META_BYTES);
    fabric.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
}

// Where the payload of the channel slot the adapter is filling begins, as a NoC address on the eth
// core. The adapter has no send at an offset, so a payload assembled from two sources has to address
// that slot directly: `edm_buffer_addr` is its base and the payload region begins one packet header
// in. Valid only between the adapter's payload send and its header send -- the header send is what
// advances the slot cursor -- so callers read it exactly there.
template <typename FabricSender>
uint64_t edm_current_payload_addr(const FabricSender& fabric) {
    // `edm_buffer_addr` tracks the live slot only for the adapter whose slot count is a runtime value;
    // the compile-time-slotted variant keeps its cursor elsewhere and leaves this field at the base.
    static_assert(
        !std::remove_reference_t<FabricSender>::USER_DEFINED_NUM_BUFFER_SLOTS,
        "edm_buffer_addr is not the live slot when the channel has user-defined slots");
    // Anything written behind the token has to keep the 16-byte alignment its source has
    // (MC_META_SLOT_BYTES), so the payload base plus the token must be a multiple of 16.
    static_assert(
        (sizeof(PACKET_HEADER_TYPE) + ct.token_size_bytes) % 16u == 0u,
        "bytes staged behind the token would land misaligned in the channel buffer");
    return get_noc_addr(
               fabric.edm_noc_x, fabric.edm_noc_y, fabric.edm_buffer_addr, tt::tt_fabric::get_fabric_worker_noc()) +
           sizeof(PACKET_HEADER_TYPE);
}

// A fan-out delivery. Its metadata is NOT behind the token in the slot -- a slot delivers to several
// destinations at once and each needs its own top-k word, so the reader staged them elsewhere -- and
// the payload is assembled in the fabric channel buffer instead: the adapter writes the token, this
// kernel writes the staged 16 bytes behind it, then the header goes out. Room for those 16 bytes is
// the program factory's business: it refuses a token the fabric payload cap does not admit with its
// tail, which is wider than the metadata.
//
// The write goes on the NoC the adapter sends on, which is what keeps it in issue order between the
// token and the header at the eth core; on any other NoC the header could arrive first, and the router
// would forward a slot still holding the previous delivery's metadata.
template <typename FabricSender>
void send_token_with_staged_meta(
    FabricSender& fabric,
    uint64_t payload_addr,
    uint64_t meta_addr,
    uint32_t token_src,
    uint32_t meta_src,
    volatile PACKET_HEADER_TYPE* hdr) {
    build_token_meta_header(hdr, payload_addr, meta_addr);
    fabric.wait_for_empty_write_slot();
    fabric.send_payload_without_header_non_blocking_from_address(token_src, ct.token_size_bytes);
    noc_async_write(
        meta_src,
        edm_current_payload_addr(fabric) + ct.token_size_bytes,
        dspf2d::METADATA_WIRE_BYTES,
        tt::tt_fabric::get_fabric_worker_noc());
    fabric.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
}

// The same records, for the destinations that live on the chip across this cable: the pages this slot
// is the last holder of. The addresses are usable here because every output buffer is interleaved DRAM
// with a mesh-uniform base, which is how the unicast tail's final addresses have always travelled.
//
// This is what one read feeding many sends buys: the page is never landed in the neighbour's region
// for it to read straight back out, and each delivery is one packet on the cable, like a forward.
template <typename FabricSender>
void deliver_remotely(FabricSender& fabric, uint32_t slot, uint32_t slot_base) {
    volatile tt_l1_ptr dspf2d::FanoutMetadata* metadata = slot_mc_metadata(slot);
    // The two lists share one FO_MAX_DESTS-record budget, so the remote share is what the local one
    // leaves: clamping each to FO_MAX_DESTS alone would still let `first + i` run off the end.
    const uint32_t first = staged_count(metadata->local_count, dspf2d::FO_MAX_DESTS);
    const uint32_t n = staged_count(metadata->remote_count, dspf2d::FO_MAX_DESTS - first);
    volatile tt_l1_ptr dspf2d::FanoutDelivery* dl = slot_delivery(slot);
    for (uint32_t i = 0; i < n; i++) {
        send_token_with_staged_meta(
            fabric,
            dl[first + i].payload_addr,
            dl[first + i].meta_addr,
            slot_base,
            dl[first + i].meta_src,
            slot_hdr(slot, dspf2d::FO_FIRST_DELIVERY_HDR + i));
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
        // Last hop: the token to its page and the three metadata words to the same page index of the
        // metadata tensor, in one packet. Both addresses were computed on the chip the token started
        // from and travelled with it, so this hop needs no address generator of its own. The metadata
        // words sit at the very start of the slot's tail, right behind the token, which is what lets the
        // packet be sent straight out of the slot.
        //
        // Compiled out under fan-out, which has no last hop of its own: there the same 64 bytes are a
        // FanoutMetadata, and `final_payload_addr` would be two packed destinations read as an address.
        send_token_with_inline_meta(fabric, metadata->final_payload_addr, metadata->final_meta_addr, slot_base, hdr);
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
