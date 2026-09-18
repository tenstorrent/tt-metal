// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel (reader RISC, NOC_0). Leads the build of this chip's routing index, then fills the L1
// ring the sender on this same core drains.
//
// The routing index is the piece with no counterpart in combine. Combine's input is already grouped by
// origin chip, so a chunk is four words out of a control table. Dispatch's input is token order and a
// token's destination is data-dependent (indices -> dispatch table), so the destination-grouped runs the
// protocol needs have to be manufactured here. The build itself lives in dispatch_fabric2d_prologue.hpp,
// shared with the compute kernel: this RISC reads the tables in and runs one of the four lanes.
//
// Replaying the production op's allocator EXACTLY is what makes the pages byte-identical to it, including
// the rule that a token past the buffer's capacity is dropped while its counter still advances. Every
// stream core replays the whole walk independently and identically, which is what lets the baton
// semaphore the production op passes between its workers disappear.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/debug/assert.h"
#include "../dispatch_fabric2d_prologue.hpp"

namespace {

namespace prologue = dspf2d::prologue;
using prologue::Control;
using prologue::ct;

// Every source chip's row of the offsets table, plus the two tensors that close the last row, plus the
// dispatch table. A few kB, read once and indexed from L1 thereafter.
void read_control_tables(const Control& c) {
    const auto offsets_acc = TensorAccessor(
        dspf2d::ReaderCtArgs::offsets_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kExpertOffsetsAddr));
    const auto table_acc = TensorAccessor(
        dspf2d::ReaderCtArgs::table_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kDispatchTableAddr));
    const auto counts_acc =
        TensorAccessor(dspf2d::ReaderCtArgs::counts_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kCountsAddr));
    const auto region_acc = TensorAccessor(
        dspf2d::ReaderCtArgs::region_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kRegionOffsetsAddr));

    const uint32_t row_bytes = ct.num_routed_experts * 4u;
    for (uint32_t r = 0; r < ct.extent; r++) {
        noc_async_read(offsets_acc.get_noc_addr(r), (uint32_t)(c.offsets + r * ct.num_routed_experts), row_bytes);
    }
    noc_async_read(counts_acc.get_noc_addr(0), (uint32_t)c.counts, row_bytes);
    noc_async_read(region_acc.get_noc_addr(0), (uint32_t)c.region, row_bytes);
    // The table carries a trailing sentinel column so a padded token's unguarded lookup maps to -1.
    noc_async_read(table_acc.get_noc_addr(0), (uint32_t)c.table, (ct.num_routed_experts + 1) * 4u);
    if constexpr (ct.has_padding_config) {
        const auto padding_acc = TensorAccessor(
            dspf2d::ReaderCtArgs::padding_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kPaddingConfigAddr));
        noc_async_read(padding_acc.get_noc_addr(0), (uint32_t)c.padding, dspf2d::PADDING_CONFIG_BYTES);
    }
    noc_async_read_barrier();
}

// One 64-byte-padded record per token. The pad is not a convenience: a DRAM read needs a 64-byte-aligned
// L1 destination on Blackhole, so reading topk uint16 per token into a packed array would put every token
// after the first at a wrong address and build the whole index out of garbage.
void read_indices(const Control& c) {
    const auto indices_acc =
        TensorAccessor(dspf2d::ReaderCtArgs::indices_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kIndicesAddr));
    const uint32_t record_bytes = ct.topk * 2u;
    for (uint32_t t = 0; t < ct.seq_len; t++) {
        noc_async_read(indices_acc.get_noc_addr(t), (uint32_t)c.indices + t * ct.indices_pad_stride, record_bytes);
    }
    noc_async_read_barrier();
}

// Tokens one origin chip owes one expert. Every chip on the axis computes this identically from the
// replicated table, which is what lets a chip size a run it neither wrote nor receives. Rows are
// absolute buffer positions, so the last origin closes against counts + region, not counts alone.
uint32_t run_len(const Control& c, uint32_t origin_row, uint32_t e) {
    const uint32_t at = c.offsets[origin_row * ct.num_routed_experts + e];
    const uint32_t routed = (origin_row + 1 < ct.extent) ? c.offsets[(origin_row + 1) * ct.num_routed_experts + e] - at
                                                         : c.counts[e] + c.region[e] - at;
    // The table counts every token routed to e, but the origin drops the ones past the expert's
    // capacity while still advancing the counter. Sizing a chunk by `routed` would make every reader
    // downstream wait for pages the origin never sent.
    return prologue::survivors_of(at, routed);
}

// expert -> bucket slot, and the chip -> experts inverse beside it.
//
// Every chip on the axis builds the same inverse because the dispatch table is replicated along it,
// which is what lets a relay expand a (origin, destination) descriptor into the same experts_per_chip
// chunks the writer expanded it into. Resolving a pick is then one indexed load rather than a table
// lookup followed by a linear search over the destination chip's experts.
void build_expert_slots(const Control& c) {
    for (uint32_t i = 0; i < ct.extent * ct.experts_per_chip; i++) {
        c.chip_experts[i] = 0;
    }
    for (uint32_t r = 0; r < ct.extent; r++) {
        c.row_fill[r] = 0;
    }
    // Inclusive of the dispatch table's trailing sentinel column, so that a padded token's unguarded
    // lookup resolves to "not in this group" here exactly as it did when the pass read the table
    // itself.
    //
    // The guard bounds the write rather than trusting the table, whose WIDTH is validated on the host
    // but whose VALUES are not: a column naming a row off the axis, or one row too many, would put
    // `slot` in the next row's buckets and at the last row past the block entirely. A relay would then
    // size a chunk from an inverse its neighbour does not share, and the axis would wait forever.
    // Refusing the entry makes a malformed table produce no pages instead of corrupting L1.
    for (uint32_t e = 0; e <= ct.num_routed_experts; e++) {
        const int32_t row = c.table[e];
        if (row < 0 || (uint32_t)row >= ct.extent || c.row_fill[(uint32_t)row] >= ct.experts_per_chip) {
            c.expert_slot[e] = dspf2d::ES_NOT_HERE;
            continue;
        }
        const uint32_t r = (uint32_t)row;
        const uint32_t j = c.row_fill[r];
        c.row_fill[r] = j + 1u;
        const uint32_t slot = r * ct.experts_per_chip + j;
        c.chip_experts[slot] = e;
        c.expert_slot[e] = slot;
    }
    for (uint32_t r = 0; r < ct.extent; r++) {
        // The whole protocol sizes a relayed chunk group as experts_per_chip terms, so a chip hosting a
        // different number would desynchronise the writer and the reader of a forwarding region.
        ASSERT(c.row_fill[r] == ct.experts_per_chip);
    }
}

// Where each bucket sits and how long it is, plus each bucket's first output page.
//
// The lengths come straight out of the offsets table rather than from a counting pass over the picks:
// this chip owes expert e exactly run_len tokens, because that table was derived from the same
// routing. Every chunk length in the protocol is already run_len, so taking the buckets from it makes
// the two agree by construction instead of by an ASSERT this hardware compiles out.
//
// bucket_start is an exclusive prefix sum with a closing total: bucket b runs from bucket_start[b] to
// bucket_start[b + 1], which is what bounds the fill and what gives the phases their run lengths.
void size_buckets(const Control& c) {
    const uint32_t n_slots = ct.extent * ct.experts_per_chip;
    // One entry per (token, pick) is what the block holds, and the total cannot exceed it. The lengths
    // come from host tensors that nothing ties to seq_len, so an offsets table inconsistent with the
    // input would otherwise run the fill -- and the tail zeroing behind it -- through every block after
    // this one and into the global semaphores. A consistent table never reaches the cap.
    const uint32_t max_entries = ct.seq_len * ct.topk;
    uint32_t at = 0;
    for (uint32_t b = 0; b < n_slots; b++) {
        const uint32_t e = c.chip_experts[b];
        c.first_page[b] = c.offsets[ct.my_row * ct.num_routed_experts + e];
        c.bucket_start[b] = at;
        const uint32_t n = run_len(c, ct.my_row, e);
        at = (at + n > max_entries) ? max_entries : at + n;
    }
    c.bucket_start[n_slots] = at;
}

// After the four lanes have filled their slices: does every bucket hold exactly the length the offsets
// table sized it at? The lanes' counts per bucket sum to the picks routed to it, and the survivors are
// the first of those, in token order, which is what the lanes wrote. A divergence means the replay and
// the production allocator disagree, which surfaces as wrong pages or, once chunk lengths are computed
// from these same numbers, as a deadlock.
//
// The tail of a bucket the replay left short is neutralised rather than merely asserted about. The
// phases take their run length from the bucket, not from the fill, so an entry the pass never wrote
// would be read out of a control region nothing zeroes -- and its `page` word becomes a fabric write
// to an arbitrary DRAM address on another chip. This costs one load per bucket when the tables
// agree, which they do, and turns that into a duplicate write of token 0 to page 0.
void merge_routing_index(const Control& c) {
    for (uint32_t b = 0; b < prologue::bucket_slots(); b++) {
        uint32_t routed = 0;
        for (uint32_t lane = 0; lane < prologue::LANES; lane++) {
            routed += prologue::lane_view(c, lane).cnt[b];
        }
        const uint32_t survivors = prologue::survivors_of(c.first_page[b], routed);
        const uint32_t len = c.bucket_start[b + 1u] - c.bucket_start[b];
        const uint32_t fill = c.bucket_start[b] + (survivors < len ? survivors : len);
        ASSERT(fill == c.bucket_start[b + 1u]);
        for (uint32_t at = fill; at < c.bucket_start[b + 1u]; at++) {
            volatile tt_l1_ptr uint32_t* ent = c.entries + at * dspf2d::entry_words();
            ent[0] = 0;
            ent[1] = 0;
            ent[2] = 0;
        }
    }
}

// Under a TILE input the tokens are not where the input tensor is: the untilizer pool is writing them
// into a staging buffer, which is what `in_acc` addresses. Nothing may be read out of it until every
// stripe has landed.
//
// Placed immediately before the first phase that reads a token rather than before the prologue,
// which is the whole reason the staging design pays: the pool's DRAM traffic and this RISC's scalar
// index build are different resources and overlap for free. If this zone is not ~0 the pool is the
// critical path and needs more cores, not the prologue.
void wait_for_untilize() {
    if constexpr (ct.untilize_stripes > 0) {
        DeviceZoneScopedN("dspf2d_wait_untilize");
        volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.untilize_sem_addr);
        while (true) {
            // Another core on this chip owns the counter, so a cached line would never show its
            // increments.
            invalidate_l1_cache();
            if (*landed >= ct.untilize_stripes) {
                return;
            }
        }
    }
}

using prologue::slice_begin;

// The reader -> sender ring. Two monotonic single-writer counters: this side owns `filled`, the sender
// owns `freed`, and each works on the difference, so neither needs a read-modify-write the other could
// race. `claimed` runs ahead of `published` so several token reads are in flight at once; BATCH <=
// NUM_L1_SLOTS/2 is what proves the reader cannot take every slot before announcing any.
struct Ring {
    uint32_t claimed = 0;
    // Slots whose routing tail is final. A slot is announced only once it reaches here, never merely
    // because it was claimed: a reader that claims several slots to keep DRAM reads in flight would
    // otherwise hand the sender a slot whose command word it has not written yet, every time a claim
    // blocks and flushes.
    uint32_t ready = 0;
    uint32_t published = 0;

    uint32_t claim_slot() {
        volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.freed_addr);
        while (true) {
            invalidate_l1_cache();
            if (claimed - *freed < ct.num_l1_slots) {
                return claimed++ % ct.num_l1_slots;
            }
            // Never block holding slots the sender has not been told about, or it waits on us while we
            // wait on it.
            flush_publish();
        }
    }

    // A slot's tail is written; it may now be announced.
    void mark_ready() { ready++; }

    // Give back the capacity of every claim that was never marked ready. The sender never saw them --
    // flush_publish only ever announces up to `ready` -- so this only stops them counting against
    // claimed - freed for the rest of the run.
    void release_unready() {
        ASSERT(claimed >= ready);
        claimed = ready;
    }

    void flush_publish() {
        if (published == ready) {
            return;
        }
        // The tokens have to be in L1 before the sender is told the slots are filled.
        noc_async_read_barrier();
        noc_semaphore_inc(get_noc_addr(ct.filled_addr), ready - published);
        published = ready;
    }
};

uint32_t chunk_len(const Control& c, uint32_t origin_row, uint32_t e, uint32_t idx, uint32_t count) {
    const uint32_t n = run_len(c, origin_row, e);
    return slice_begin(n, idx + 1, count) - slice_begin(n, idx, count);
}

// Where each chunk of a descriptor list starts, as a page offset into a stream's region. The region is
// dense and holds no addresses, so a chunk is found only by summing the lengths before it -- and both
// sides of a region run this over lists validate_chunk_agreement proved identical.
uint32_t chunk_starts(const Control& c, uint32_t block_base, volatile tt_l1_ptr uint32_t* start) {
    uint32_t at = 0;
    for (uint32_t d = 0; d < ct.num_relay; d++) {
        const uint32_t base = block_base + d * dspf2d::ASSIGNMENT_WORDS;
        const uint32_t origin = kernel_compile_time_args[base + 0];
        const uint32_t dst = kernel_compile_time_args[base + 1];
        const uint32_t idx = kernel_compile_time_args[base + 2];
        const uint32_t cnt = kernel_compile_time_args[base + 3];
        for (uint32_t j = 0; j < ct.experts_per_chip; j++) {
            const uint32_t e = c.chip_experts[dst * ct.experts_per_chip + j];
            start[d * ct.experts_per_chip + j] = at;
            at += chunk_len(c, origin, e, idx, cnt);
        }
    }
    // The host bounds the region without knowing any of these lengths, so this is where that bound is
    // actually tested against the data.
    ASSERT(at <= ct.fwd_pages_per_stream);
    return at;
}

uint32_t slot_addr(uint32_t slot) { return ct.ring_addr + slot * ct.slot_stride(); }

volatile tt_l1_ptr dspf2d::FwdMetadata* slot_tail(uint32_t slot) {
    return reinterpret_cast<volatile tt_l1_ptr dspf2d::FwdMetadata*>(slot_addr(slot) + ct.token_size_bytes);
}

// This chip's own tokens for its remote destinations.
template <typename InAcc, typename OutAcc, typename MetaAcc, typename FwdAcc>
void own_phase(
    const Control& c,
    Ring& ring,
    const InAcc& in_acc,
    const OutAcc& out_acc,
    const MetaAcc& meta_acc,
    const FwdAcc& fwd_acc,
    uint32_t my_region) {
    // Own assignments, furthest first. The nearest one is the chip across the cable: a single hop that
    // lands straight in its output. Everything further goes into that chip's forwarding region instead,
    // at the position the two chips agree this chunk occupies -- own assignment a is outgoing
    // descriptor a, because both lists are emitted furthest-first by the same generator.
    for (uint32_t a = 0; a < ct.num_own; a++) {
        const uint32_t base = ct.assignment_base + a * dspf2d::ASSIGNMENT_WORDS;
        const uint32_t dst_chip = kernel_compile_time_args[base + 0];
        const uint32_t dst_row = kernel_compile_time_args[base + 1];
        const uint32_t split_idx = kernel_compile_time_args[base + 2];
        const uint32_t split_count = kernel_compile_time_args[base + 3];
        const bool direct = (dst_chip == ct.nbr_chip_id);
        for (uint32_t j = 0; j < ct.experts_per_chip; j++) {
            const uint32_t b = dst_row * ct.experts_per_chip + j;
            const uint32_t bucket = c.bucket_start[b];
            const uint32_t n = c.bucket_start[b + 1u] - bucket;
            const uint32_t from = slice_begin(n, split_idx, split_count);
            const uint32_t to = slice_begin(n, split_idx + 1, split_count);
            const uint32_t out_base = direct ? 0 : c.out_start[a * ct.experts_per_chip + j];
            for (uint32_t i = from; i < to; i++) {
                const uint32_t at = (bucket + i) * dspf2d::entry_words();
                const uint32_t token = c.entries[at + 0];
                const uint32_t page = c.entries[at + 1];
                const uint32_t slot = ring.claim_slot();
                noc_async_read(in_acc.get_noc_addr(token), slot_addr(slot), ct.token_size_bytes);

                // Both destination addresses are computed HERE and travel with the token: every buffer
                // is interleaved DRAM whose base is uniform across the mesh, so a page index names the
                // same place on any chip and no later hop needs an address generator.
                volatile tt_l1_ptr dspf2d::FwdMetadata* tail = slot_tail(slot);
                tail->final_payload_addr = out_acc.get_noc_addr(page);
                tail->final_meta_addr = meta_acc.get_noc_addr(page);
                tail->dst_chip = dst_chip;
                tail->meta[0] = ct.linearized_coord;
                tail->meta[1] = token;
                tail->meta[2] = c.entries[at + 2];
                tail->pad = 0;
                if (direct) {
                    tail->cmd = dspf2d::CMD_FINAL_WRITE;
                    tail->this_addr = tail->final_payload_addr;
                } else {
                    // The last page of a chunk forces the downstream bump, which is the boundary that
                    // reader switches on: leave it uncounted and the whole axis waits.
                    tail->cmd = (i + 1 == to) ? dspf2d::CMD_FORWARD_END : dspf2d::CMD_FORWARD;
                    tail->this_addr = fwd_acc.get_noc_addr(my_region + out_base + (i - from));
                }
                ring.mark_ready();
            }
        }
    }
}

// Pages this stream relays: read out of its own forwarding region and pushed one hop further, or
// delivered if the chip across the cable is where they were going.
template <typename FwdAcc>
uint32_t relay_phase(const Control& c, Ring& ring, const FwdAcc& fwd_acc, uint32_t my_region, uint32_t nbr_row) {
    uint32_t consumed = 0;  // pages taken out of this stream's region, which is what end_stream gives back
    // Arrivals, in the order upstream wrote them. A page here is bound for the chip across the cable or
    // further; the first case is a final write, the second goes into that chip's region at the position
    // the outgoing list gives it.
    uint32_t out_d = ct.num_own - 1;  // own assignments occupy the first num_own - 1 outgoing descriptors
    for (uint32_t d = 0; d < ct.num_relay; d++) {
        const uint32_t base = ct.in_chunks_base + d * dspf2d::ASSIGNMENT_WORDS;
        const uint32_t origin = kernel_compile_time_args[base + 0];
        const uint32_t dst_row = kernel_compile_time_args[base + 1];
        const uint32_t idx = kernel_compile_time_args[base + 2];
        const uint32_t cnt = kernel_compile_time_args[base + 3];
        const bool continues = (dst_row != nbr_row);
        const uint32_t dst_chip = kernel_compile_time_args[ct.ring_chip_ids_base + dst_row];
        const uint32_t this_out_d = continues ? out_d++ : 0;
        for (uint32_t j = 0; j < ct.experts_per_chip; j++) {
            const uint32_t e = c.chip_experts[dst_row * ct.experts_per_chip + j];
            const uint32_t len = chunk_len(c, origin, e, idx, cnt);
            consumed += len;
            const uint32_t in_base = c.in_start[d * ct.experts_per_chip + j];
            const uint32_t out_base = continues ? c.out_start[this_out_d * ct.experts_per_chip + j] : 0;
            volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.fwd_sem_addr);
            // Pages are read a batch at a time. One read in flight per stream is a DRAM round trip
            // per page -- the op's actual limit, not its bandwidth -- and the ring already has the
            // depth to cover it. BATCH <= NUM_L1_SLOTS/2 is what keeps a batch of unannounced slots
            // from filling the ring against a sender that is still draining the previous one.
            for (uint32_t p = 0; p < len;) {
                const uint32_t n = (len - p < ct.batch) ? (len - p) : ct.batch;
                uint32_t slots[dspf2d::BATCH];
                for (uint32_t i = 0; i < n; i++) {
                    // Upstream fills the region strictly left to right, so its page count is the
                    // high-water offset and a page is ready once that count passes it.
                    // The invalidate has to precede the read: an upstream chip owns this counter, so a
                    // cached line would never show its increments.
                    while (true) {
                        invalidate_l1_cache();
                        if (*arrived > in_base + p + i) {
                            break;
                        }
                        ring.flush_publish();  // let our own sender work while we wait on upstream
                    }
                    slots[i] = ring.claim_slot();
                    noc_async_read(
                        fwd_acc.get_noc_addr(my_region + in_base + p + i),
                        slot_addr(slots[i]),
                        ct.token_size_bytes + dspf2d::FWD_EXTRA_BYTES);
                }
                noc_async_read_barrier();  // the routing tails decide the next hop, so they must be here
                // The reads landed behind the data cache, and these slots carried different pages a
                // few iterations ago: without this, a tail's addresses can still be that page's.
                invalidate_l1_cache();

                for (uint32_t i = 0; i < n; i++) {
                    volatile tt_l1_ptr dspf2d::FwdMetadata* tail = slot_tail(slots[i]);
                    // Whether this hop is the last is a property of the CHUNK, so it comes from the
                    // descriptor rather than from the arriving tail: every page of (origin, dst_row)
                    // shares one destination, and the descriptor lists are the pair
                    // validate_chunk_agreement proved the two chips agree on. Reading it back out of
                    // the tail would instead make control flow depend on a DRAM round-trip, and an
                    // unwritten tail reads as chip 0 -- indistinguishable from a genuine destination
                    // on the chip whose id is 0.
                    ASSERT(tail->dst_chip == (uint64_t)dst_chip);
                    if (!continues) {
                        tail->cmd = dspf2d::CMD_FINAL_WRITE;
                        tail->this_addr = tail->final_payload_addr;
                    } else {
                        tail->cmd = (p + i + 1 == len) ? dspf2d::CMD_FORWARD_END : dspf2d::CMD_FORWARD;
                        tail->this_addr = fwd_acc.get_noc_addr(my_region + out_base + p + i);
                    }
                    ring.mark_ready();
                }
                p += n;
            }
        }
    }
    return consumed;
}

// Tokens routed to an expert this chip hosts. They never touch the fabric, so each is a DRAM read and
// two DRAM writes out of scratch slots that are deliberately never published -- publishing them would
// put these tokens on the cable as well. Every stream runs this over its own fraction, so the split
// has to cover exactly once, the same rule the remote assignments follow.
//
// One read in flight is a DRAM round trip per token, which is the limit here, so the phase keeps
// ct.batch outstanding and waits once per batch on each side. Scratch is claimed as tokens need it,
// so a stream with few local tokens does not wait for slots it will not use. The ring argument is not
// the relays': scratch never becomes ready, so the flush_publish inside claim_slot can free nothing
// here, and progress rests on the sender draining what the earlier phases published -- which it does
// unconditionally -- with this phase holding at most ct.batch of the num_l1_slots meanwhile.
template <typename InAcc, typename OutAcc, typename MetaAcc>
void local_phase(const Control& c, Ring& ring, const InAcc& in_acc, const OutAcc& out_acc, const MetaAcc& meta_acc) {
    // Every earlier phase ended in flush_publish, so nothing unready is held: the scratch this phase
    // claims is exactly what release_unready gives back.
    ASSERT(ring.claimed == ring.ready);
    // held <= ct.batch: a slot is claimed only when every held one is pending, and a batch is written
    // out at ct.batch pending. Every batch restarts at scratch[0], reusing the same held slots, which
    // write_batch's departure wait makes safe.
    static_assert(ct.batch <= dspf2d::BATCH, "the scratch arrays are sized by BATCH");
    uint32_t scratch[dspf2d::BATCH];
    uint32_t pages[dspf2d::BATCH];
    uint32_t held = 0;     // scratch slots claimed, for the whole phase
    uint32_t pending = 0;  // tokens read into scratch[0..pending) and not yet written
    // Wait for the batch's reads, write each token and its metadata to their pages, wait for departure.
    const auto write_batch = [&]() {
        noc_async_read_barrier();
        for (uint32_t i = 0; i < pending; i++) {
            const uint32_t addr = slot_addr(scratch[i]);
            noc_async_write(addr, out_acc.get_noc_addr(pages[i]), ct.token_size_bytes);
            noc_async_write(addr + ct.token_size_bytes, meta_acc.get_noc_addr(pages[i]), dspf2d::METADATA_WIRE_BYTES);
        }
        // The slots ARE the buffers: a refill may start once the writes have read them out of L1,
        // which is departure, not completion -- the wait the sender uses before it frees a slot.
        noc_async_writes_flushed();
        pending = 0;
    };
    for (uint32_t j = 0; j < ct.experts_per_chip; j++) {
        const uint32_t b = ct.my_row * ct.experts_per_chip + j;
        const uint32_t bucket = c.bucket_start[b];
        const uint32_t n = c.bucket_start[b + 1u] - bucket;
        const uint32_t from = slice_begin(n, ct.stream, 2 * ct.num_links);
        const uint32_t to = slice_begin(n, ct.stream + 1, 2 * ct.num_links);
        for (uint32_t i = from; i < to; i++) {
            if (pending == held) {  // no held slot left for this token
                scratch[held] = ring.claim_slot();
                slot_tail(scratch[held])->pad = 0;  // goes to the metadata page; stays zero for the phase
                held++;
            }
            const uint32_t at = (bucket + i) * dspf2d::entry_words();
            const uint32_t token = c.entries[at + 0];
            const uint32_t slot = scratch[pending];
            noc_async_read(in_acc.get_noc_addr(token), slot_addr(slot), ct.token_size_bytes);
            volatile tt_l1_ptr dspf2d::FwdMetadata* tail = slot_tail(slot);
            tail->meta[0] = ct.linearized_coord;
            tail->meta[1] = token;
            tail->meta[2] = c.entries[at + 2];
            pages[pending] = c.entries[at + 1];
            if (++pending == ct.batch) {
                write_batch();
            }
        }
    }
    if (pending > 0) {
        write_batch();
    }
    // Departure was enough for the slots; the output pages need completion, and this RISC is the only
    // thing that waits for it -- the firmware barriers nothing at kernel end. One wait per phase.
    noc_async_write_barrier();
    ring.release_unready();
}

}  // namespace

void kernel_main() {
    const Control c = prologue::carve_control();
    ASSERT(c.end <= ct.filled_addr);  // the control region must not run into the semaphores

    {
        DeviceZoneScopedN("dspf2d_prologue");
        // The compute RISCs have been waiting for the tables since launch; once they are in, the four
        // lanes walk together and this RISC's own lane is one of them.
        prologue::reader_prologue(c, [&] {
            read_control_tables(c);
            read_indices(c);
            build_expert_slots(c);
            size_buckets(c);
        });
        {
            // How long this lane waits for the slowest of the other three: the split's imbalance.
            DeviceZoneScopedN("dspf2d_prologue_wait");
            prologue::wait_all_lanes(dspf2d::kLaneFilled);
        }
        {
            DeviceZoneScopedN("dspf2d_prologue_merge");
            merge_routing_index(c);
        }
    }

    const auto in_acc =
        TensorAccessor(dspf2d::ReaderCtArgs::in_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kInputAddr));
    const auto out_acc = TensorAccessor(
        dspf2d::ReaderCtArgs::out_payload_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kOutPayloadAddr));
    const auto meta_acc =
        TensorAccessor(dspf2d::ReaderCtArgs::out_meta_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kOutMetaAddr));

    const auto fwd_acc =
        TensorAccessor(dspf2d::ReaderCtArgs::fwd_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kFwdAddr));
    const uint32_t my_region = ct.stream * ct.fwd_pages_per_stream;

    // Which position on the axis the chip across this cable holds. The outgoing list is ordered by it,
    // and a page bound for it is delivered rather than forwarded.
    uint32_t nbr_row = 0;
    for (uint32_t r = 0; r < ct.extent; r++) {
        if (kernel_compile_time_args[ct.ring_chip_ids_base + r] == ct.nbr_chip_id) {
            nbr_row = r;
        }
    }

    Ring ring;
    {
        DeviceZoneScopedN("dspf2d_starts");
        chunk_starts(c, ct.in_chunks_base, c.in_start);
        chunk_starts(c, ct.out_chunks_base, c.out_start);
    }
    wait_for_untilize();  // the own phase is the first thing that reads a token
    {
        DeviceZoneScopedN("dspf2d_own");
        own_phase(c, ring, in_acc, out_acc, meta_acc, fwd_acc, my_region);
        ring.flush_publish();
    }
    uint32_t consumed = 0;  // pages this stream took out of its forwarding region
    {
        DeviceZoneScopedN("dspf2d_relay");
        consumed = relay_phase(c, ring, fwd_acc, my_region, nbr_row);
        ring.flush_publish();
    }
    // Last, because these pages never leave the chip: anything ahead of them in this order is a chip
    // downstream waiting.
    {
        DeviceZoneScopedN("dspf2d_local");
        local_phase(c, ring, in_acc, out_acc, meta_acc);
    }

    {
        DeviceZoneScopedN("dspf2d_end");
        const uint32_t end_slot = ring.claim_slot();
        slot_tail(end_slot)->cmd = dspf2d::CMD_END;
        ring.mark_ready();
        ring.flush_publish();

        noc_async_atomic_barrier();
        // Give back exactly what was taken, rather than zeroing. The upstream chip owns this counter's
        // increments and is under no obligation to have stopped: it may already be a launch ahead and
        // bumping for the next one. A zero throws those away and its relay then waits for pages that,
        // as far as the counter is concerned, never arrived -- which is a ring-wide hang rather than
        // wrong data. Subtracting leaves an early bump standing, and it is already the right base for
        // the next launch, whose positions start at zero again.
        //
        // The NoC has only an atomic add, so a subtract is the two's complement; `noc_semaphore.h`
        // does the same where it decrements. Both counts are bounded by the region, so the unsigned
        // wrap is exact.
        noc_semaphore_inc(get_noc_addr(ct.fwd_sem_addr), (uint32_t)(0u - consumed));
        noc_async_atomic_barrier();
        if constexpr (ct.untilize_stripes > 0) {
            // Safe here and only here: the wait above this launch's first token read proved all
            // untilize_stripes increments had arrived, so no writer is still bumping this counter.
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.untilize_sem_addr), 0);
        }
    }
}
