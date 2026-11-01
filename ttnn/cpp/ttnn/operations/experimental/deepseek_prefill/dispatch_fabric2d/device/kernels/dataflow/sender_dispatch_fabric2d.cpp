// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sender kernel (writer RISC, NOC_1). Drains the TokenQueue that the reader on this core fills, through
// the fabric sender connection of this core's eth channel; a channel allows one sender core. Each send is
// one hop to the chip across this cable. Tokens bound further go into that chip's fwd_section and are
// sent on from there.
//
// The last hop is a two-part scatter write out of one entry: the token to its page and the metadata words
// behind it to the metadata page. A forward hop is a plain write.
//
// Entries are claimed and released in batches to amortise the counter signals and the source flush. The
// flush is needed because a payload send reads L1 asynchronously, so an entry cannot go back to the
// reader until that read has finished.
//
// Nothing this sender writes may reach the next chip in a launch that chip has not started. Until then its
// reader may still be reading the previous launch's pages of the forwarding region, which is reused at the same
// offsets every launch, and the previous launch's output may still be in use at an address this launch's output
// can reuse. That chip grants a credit when it starts, and this sender takes it before its first send.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "dispatch_fabric2d_sender_ct_args.hpp"

// Forwarded pages between arrival bumps to the downstream reader, per channel. With the section split by
// movement class a channel has no chunk boundaries to close, so this is the only thing that decides how
// finely that reader can pipeline within a channel -- and the ceiling on how much of a channel the producer
// can fill before the consumer is allowed to know.
constexpr uint32_t FWD_SIGNAL_CAP = 32;
constexpr dspf2d::SenderCtArgs ct{};

// A batch comparable to the channel's depth is the trap this op's own forwarding region used to avoid by
// bumping on a chunk's last page: fill 32 pages of a 34-page channel before announcing any and the consumer
// sits idle while the producer stalls for credits it is the only source of. Capacity is a free parameter
// now, so every constant measured in pages has to scale with it -- a quarter of the channel, exactly as the
// reader's credit return does.
constexpr uint32_t fwd_signal_at() {
    const uint32_t quarter = ct.vc_pages / 4;
    return quarter < 1 ? 1u : (quarter < FWD_SIGNAL_CAP ? quarter : FWD_SIGNAL_CAP);
}
static_assert(ct.batch <= ct.queue_depth / 2, "the sender must be able to drain one batch while the next is read");

// One prebuilt header per queue entry. The route is constant, so only the write address changes per
// token. A single shared header could be rewritten while a previous non-blocking send still reads it.
volatile PACKET_HEADER_TYPE* entry_hdr(uint32_t entry) {
    return reinterpret_cast<volatile PACKET_HEADER_TYPE*>(ct.pkt_hdr_queue_addr + entry * sizeof(PACKET_HEADER_TYPE));
}

volatile tt_l1_ptr dspf2d::FwdMetadata* entry_meta(uint32_t entry) {
    return reinterpret_cast<volatile tt_l1_ptr dspf2d::FwdMetadata*>(
        ct.queue_addr + entry * ct.entry_stride() + ct.token_size_bytes);
}

// The bytes a last-hop delivery carries: the token and the metadata words behind it.
constexpr uint32_t TOKEN_PLUS_META_BYTES = ct.token_size_bytes + dspf2d::METADATA_WIRE_BYTES;

// A token and its metadata leave as one fabric packet: a scatter write whose first scatter part is the token
// and whose second scatter part is the metadata words behind it. The parts land on different pages but use
// one EDM slot, one header and one credit, like a forward. The route is prebuilt, so only the addresses are written.
void build_token_meta_header(volatile PACKET_HEADER_TYPE* hdr, uint64_t payload_addr, uint64_t meta_addr) {
    // Scatter part and payload sizes are 16-bit header fields; the total is the larger.
    static_assert(TOKEN_PLUS_META_BYTES <= 0xFFFFu, "a scatter payload size is a 16-bit field");
    // Only the first scatter part's size is given; the router takes the rest of the payload as the second
    // scatter part, sourced at payload base + token size. A NoC write whose source and destination differ
    // modulo 16 arrives shifted by a word.
    static_assert(ct.token_size_bytes % 16u == 0u, "the second scatter part is sourced at payload base + token size");
    hdr->to_noc_unicast_scatter_write(
        tt::tt_fabric::NocUnicastScatterCommandHeader{
            {payload_addr, meta_addr}, {static_cast<uint16_t>(ct.token_size_bytes)}},
        TOKEN_PLUS_META_BYTES);
}

// The last hop. `src` is a queue entry whose fwd_meta starts with the metadata words, so the payload is
// already laid out in L1. The header is built before waiting for a free EDM slot so that building it
// overlaps the wait.
template <typename FabricSender>
void send_token_with_inline_meta(
    FabricSender& fabric, uint64_t payload_addr, uint64_t meta_addr, uint32_t src, volatile PACKET_HEADER_TYPE* hdr) {
    build_token_meta_header(hdr, payload_addr, meta_addr);
    fabric.wait_for_empty_write_slot();
    fabric.send_payload_without_header_non_blocking_from_address(src, TOKEN_PLUS_META_BYTES);
    fabric.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
}

void prebuild_routes() {
    for (uint32_t entry = 0; entry < ct.queue_depth; entry++) {
        fabric_set_unicast_route(
            (volatile tt::tt_fabric::HybridMeshPacketHeader*)entry_hdr(entry),
            ct.downstream_chip_id,
            ct.downstream_mesh_id);
    }
    // pkt_hdr_signal is used by grant_launch_credit before the send loop, by bump_downstream during it and
    // by drain_fabric after it. Setting the command fields later leaves this route in place.
    fabric_set_unicast_route(
        reinterpret_cast<volatile tt::tt_fabric::HybridMeshPacketHeader*>(ct.pkt_hdr_signal_addr),
        ct.downstream_chip_id,
        ct.downstream_mesh_id);
    // Credits go to a different CORE on the chip this sender already talks to, so the route is the same one
    // and is built once here too.
    fabric_set_unicast_route(
        reinterpret_cast<volatile tt::tt_fabric::HybridMeshPacketHeader*>(ct.pkt_hdr_credit_addr),
        ct.downstream_chip_id,
        ct.downstream_mesh_id);
}

// Defined below, next to the other counter bumps; the wait needs it to announce what it has already sent
// before it goes idle.
template <typename FabricSender>
void bump_downstream(FabricSender& fabric, uint32_t sem_addr, uint32_t count);

// Courier for the OPPOSITE direction's reader on this chip, one queue per channel. That reader frees pages
// of the section the chip across our cable wrote, so the credit is owed to exactly the chip we already
// point at -- it just lands on the other direction's worker there rather than ours.
//
// Its own header and its own running totals, so it never disturbs this stream's tokens: a credit can be
// posted from wherever we happen to be, including from inside the wait below.
struct Courier {
    uint32_t posted[dspf2d::MAX_VCS] = {};

    // Post whatever the opposite reader has queued for each channel since the last call. Monotonic counters
    // minus our own totals, so there is no read-modify-write racing that reader.
    template <typename FabricSender>
    void service(FabricSender& fabric) {
        for (uint32_t v = 0; v < ct.num_vcs; v++) {
            volatile tt_l1_ptr uint32_t* outbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.vc_outbox_addr(v));
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

// Tell the chip across the cable that this chip has started the launch. Its sender on the stream back toward
// us waits for this before it writes to us. Being in this launch means every kernel of this op's previous
// launch on this chip, our readers included, has finished: a launch's go signal waits for every core of its
// sub-device to finish the one before.
//
// Sent before this kernel waits on the reader or on any other chip, so the credit never depends on this
// launch making progress.
template <typename FabricSender>
void grant_launch_credit(FabricSender& fabric) {
    volatile PACKET_HEADER_TYPE* hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(ct.pkt_hdr_signal_addr);
    hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        get_noc_addr(ct.credit_noc_x, ct.credit_noc_y, ct.launch_credit_addr), /*val=*/1, /*flush=*/true});
    fabric.wait_for_empty_write_slot();
    fabric.send_payload_flush_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
}

// Wait until the chip across the cable has started this launch, then take its credit. Taken by subtracting
// one rather than by zeroing: that chip can start the next launch, and grant the next credit, while this
// one is still running.
void take_launch_credit() {
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.launch_credit_addr), 1);
    noc_semaphore_inc(get_noc_addr(ct.launch_credit_addr), 0u - 1u);
}

// Blocks until the reader has announced at least one entry beyond `sent`, then reports how many.
//
// Credits are serviced inside the spin, not only between batches. That is what keeps the two directions
// independent: a clockwise sender idle on its own reader would otherwise stop carrying the anticlockwise
// reader's credits, that ring would stall for want of them, and the two would hold each other up.
template <typename FabricSender>
uint32_t wait_for_filled(FabricSender& fabric, uint32_t sent, uint32_t* fwd_since_signal) {
    volatile tt_l1_ptr uint32_t* filled = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.filled_addr);
    bool announced = false;
    while (true) {
        invalidate_l1_cache();
        const uint32_t avail = *filled - sent;
        if (avail > 0) {
            return avail;
        }
        // Nothing of ours to send, so announce what we already sent. A channel is only vc_pages deep while
        // the bump batch is fwd_signal_at(), so a producer that fills a channel without completing a batch
        // would never tell the consumer those pages are there -- and that consumer is the only source of the
        // credits the producer is now waiting on. Deadlock, and exactly the one a chunk-ending forward used
        // to rule out by forcing a signal on a chunk's last page. Channels have no chunk ends, so this
        // announce is what rules it out instead.
        //
        // Once per idle spell rather than per spin, or announcing becomes the busy-wait.
        if (!announced) {
            for (uint32_t v = 0; v < ct.num_vcs; v++) {
                if (fwd_since_signal[v] > 0) {
                    bump_downstream(fabric, ct.vc_arrived_addr(v), fwd_since_signal[v]);
                    fwd_since_signal[v] = 0;
                }
            }
            announced = true;
        }
        courier.service(fabric);
    }
}

// Add `count` to a counter on the downstream reader's core: the arrival count of one of its channels, or
// `final_sem_addr` for tokens written straight into its chip's output. The flush makes the far router drain the
// writes ahead of the bump first.
template <typename FabricSender>
void bump_downstream(FabricSender& fabric, uint32_t sem_addr, uint32_t count) {
    volatile PACKET_HEADER_TYPE* hdr_signal = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(ct.pkt_hdr_signal_addr);
    // downstream_noc_x/y are the downstream stream core's coordinates. A header-only atomic inc. The fused
    // write + inc hangs Blackhole when the payload destination is DRAM, and the section and the output are in DRAM.
    hdr_signal->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        get_noc_addr(ct.downstream_noc_x, ct.downstream_noc_y, sem_addr), /*val=*/count, /*flush=*/true});
    fabric.wait_for_empty_write_slot();
    fabric.send_payload_flush_blocking_from_address((uint32_t)hdr_signal, sizeof(PACKET_HEADER_TYPE));
}

// Put one entry's token on the cable. Returns its command word so the caller can spot the end of stream.
template <typename FabricSender>
uint64_t send_entry(FabricSender& fabric, uint32_t entry, uint32_t* fwd_since_signal, uint32_t& finals_sent) {
    volatile tt_l1_ptr dspf2d::FwdMetadata* metadata = entry_meta(entry);
    const uint64_t cmd = metadata->cmd;
    if (cmd == dspf2d::CMD_END) {
        return cmd;
    }
    const uint32_t entry_base = ct.queue_addr + entry * ct.entry_stride();
    volatile PACKET_HEADER_TYPE* hdr = entry_hdr(entry);
    if (cmd == dspf2d::CMD_FORWARD) {
        // One write of the token and its whole fwd_meta into a forwarding page on the next chip.
        const uint32_t payload_bytes = ct.token_size_bytes + dspf2d::FWD_EXTRA_BYTES;
        // Header first, then wait for the EDM slot, as in send_token_with_inline_meta.
        hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{metadata->this_addr}, payload_bytes);
        fabric.wait_for_empty_write_slot();
        fabric.send_payload_without_header_non_blocking_from_address(entry_base, payload_bytes);
        fabric.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));

        // Per channel, because the downstream reader waits on each channel separately -- that is what lets
        // it serve a channel it can drain while another is blocked.
        const uint32_t vc = (uint32_t)metadata->vc;
        fwd_since_signal[vc]++;
        if (fwd_since_signal[vc] >= fwd_signal_at()) {
            bump_downstream(fabric, ct.vc_arrived_addr(vc), fwd_since_signal[vc]);
            fwd_since_signal[vc] = 0;
        }
    } else {
        // Last hop. Both addresses were computed on the token's source chip and came with it, so this hop
        // needs no address generator.
        send_token_with_inline_meta(fabric, metadata->final_payload_addr, metadata->final_meta_addr, entry_base, hdr);
        finals_sent++;
    }
    return cmd;
}

// Drain the queue until the reader ends the stream.
template <typename FabricSender>
void pump_stream(FabricSender& fabric) {
    const uint64_t my_freed_noc = get_noc_addr(ct.freed_addr);
    uint32_t sent = 0;
    // The stream length is not known here because it includes the tokens the reader forwards for other
    // chips. The reader ends the stream with a CMD_END entry.
    bool end_of_stream = false;
    uint32_t fwd_since_signal[dspf2d::MAX_VCS] = {};
    uint32_t finals_sent = 0;
    while (!end_of_stream) {
        const uint32_t avail = wait_for_filled(fabric, sent, fwd_since_signal);
        const uint32_t n = avail < ct.batch ? avail : ct.batch;

        uint32_t processed = 0;
        for (uint32_t i = 0; i < n; i++) {
            processed++;
            if (send_entry(fabric, (sent + i) % ct.queue_depth, fwd_since_signal, finals_sent) == dspf2d::CMD_END) {
                end_of_stream = true;
                break;
            }
        }
        // The batch's payload reads have drained out of L1, so these entries are safe to refill.
        noc_async_writes_flushed();
        sent += processed;
        noc_semaphore_inc(my_freed_noc, processed);
        courier.service(fabric);
    }
    // Residual arrival counts. A reader blocks until its channel's count passes what it has consumed, so a
    // tail left inside a partial bump batch would strand it with the stream already over.
    for (uint32_t v = 0; v < ct.num_vcs; v++) {
        if (fwd_since_signal[v] > 0) {
            bump_downstream(fabric, ct.vc_arrived_addr(v), fwd_since_signal[v]);
        }
    }
    // The downstream reader reads its receive count once, after its own stream has ended, so one signal for the
    // whole stream is enough; a signal per batch would only add flushes to the cable the tokens share.
    if (finals_sent > 0) {
        bump_downstream(fabric, ct.final_sem_addr, finals_sent);
    }
}

// Delivery barrier. Program completion does not mean our packets reached the destination chip, so
// without this the host could read an output while its last tokens are still in flight, and a launch credit
// could land after the program that granted it.
//
// With D = num_buffers_per_channel, the free EDM slot count is D - (packets_written - credits_returned),
// and only the far end returns credits. Writing D-1 more packets and then waiting for one more free EDM slot
// therefore means every payload packet has reached the destination chip. It does not prove the
// destination DRAM write has finished: the far eRISC may acknowledge when it issues the write.
//
// The filler packets are header-only atomic incs of 0 to a drain sink on the downstream chip, because the
// fabric has no no-op packet. Waiting for a free slot cannot deadlock: the reverse direction uses
// another eth channel.
template <typename FabricSender>
void drain_fabric(FabricSender& fabric) {
    volatile PACKET_HEADER_TYPE* hdr_drain = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(ct.pkt_hdr_signal_addr);
    hdr_drain->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        get_noc_addr(ct.downstream_noc_x, ct.downstream_noc_y, ct.drain_sink_addr), /*val=*/0, /*flush=*/false});
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
    grant_launch_credit(fabric);
    uint32_t no_bumps[dspf2d::MAX_VCS] = {};
    // Once per launch, so the credits stay in step with the launches: every stream has at least its CMD_END
    // entry. The reader publishes its first entry only after building its tables, so the credit has usually
    // arrived by the time this wait starts.
    wait_for_filled(fabric, 0, no_bumps);
    take_launch_credit();
    pump_stream(fabric);
    // Before the drain, which fills every free write slot the channel has and would leave no room to post a
    // credit until it is done.
    courier.finish(fabric);
    // Even on a stream that sent no token, the launch credit is on the cable.
    drain_fabric(fabric);

    noc_async_writes_flushed();
    fabric_connections.close();

    // Reset both queue counters for the next launch, which counts from zero. This is safe only here: the
    // reader neither publishes nor signals either counter after the CMD_END entry drained above, so nothing
    // still reads or signals them.
    //
    // `freed` is incremented by NoC atomics, which noc_async_writes_flushed does not wait for. Without the
    // barrier an increment can land after the reset, and the next launch computes the reader's
    // `claimed - *freed` as a negative wrap. The barrier also retires the launch credit taken above.
    noc_async_atomic_barrier();
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.filled_addr), 0);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.freed_addr), 0);
    // The courier's counters are this op's either way: they never leave this chip, and the opposite reader
    // that writes them has already raised credit_done, which is the last thing it does to any of them.
    for (uint32_t v = 0; v < ct.num_vcs; v++) {
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.vc_outbox_addr(v)), 0);
    }
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.credit_done_addr), 0);
}
