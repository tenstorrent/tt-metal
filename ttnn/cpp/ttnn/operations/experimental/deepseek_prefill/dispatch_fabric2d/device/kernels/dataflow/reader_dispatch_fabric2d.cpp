// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel of a stream core (reader RISC, NOC_0). Reads the tables and builds this chip's routing index
// together with the three TRISCs, then reads tokens into the TokenQueue that the sender on this core drains.
// Every stream core builds the whole index on its own, so stream cores never wait on each other for it.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/debug/assert.h"
#include "../dispatch_fabric2d_routing_index.hpp"

namespace {

namespace routing_index = dspf2d::routing_index;
using routing_index::ct;
using routing_index::Scratch;

// Every source chip's row of the offsets table, the two tensors that close the last row, and the dispatch
// table. A few kB, read into scratch once.
void read_control_tables(const Scratch& c) {
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
    noc_async_read(region_acc.get_noc_addr(0), (uint32_t)c.region_offsets, row_bytes);
    // The table carries a trailing sentinel column so a padded token's unguarded lookup maps to -1.
    noc_async_read(table_acc.get_noc_addr(0), (uint32_t)c.table, (ct.num_routed_experts + 1) * 4u);
    if constexpr (ct.has_padding_config) {
        const auto padding_acc = TensorAccessor(
            dspf2d::ReaderCtArgs::padding_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kPaddingConfigAddr));
        noc_async_read(padding_acc.get_noc_addr(0), (uint32_t)c.padding, dspf2d::PADDING_CONFIG_BYTES);
    }
    noc_async_read_barrier();
}

// One record per token, padded to indices_pad_stride because on Blackhole a DRAM read needs a 64-byte-aligned
// L1 destination.
void read_indices(const Scratch& c) {
    const auto indices_acc =
        TensorAccessor(dspf2d::ReaderCtArgs::indices_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kIndicesAddr));
    const uint32_t record_bytes = ct.topk * 2u;
    for (uint32_t t = 0; t < ct.seq_len; t++) {
        noc_async_read(indices_acc.get_noc_addr(t), (uint32_t)c.indices + t * ct.indices_pad_stride, record_bytes);
    }
    noc_async_read_barrier();
}

// run_len: tokens origin_pos sends to expert e after the capacity drop. Every chip on the axis computes the
// same value from the replicated table, so a chip can size a chunk it neither writes nor receives. Offsets are
// absolute buffer positions, so the last origin position closes against counts + region_offsets.
uint32_t run_len(const Scratch& c, uint32_t origin_pos, uint32_t e) {
    const uint32_t at = c.offsets[origin_pos * ct.num_routed_experts + e];
    const uint32_t routed = (origin_pos + 1 < ct.extent) ? c.offsets[(origin_pos + 1) * ct.num_routed_experts + e] - at
                                                         : c.counts[e] + c.region_offsets[e] - at;
    // The origin drops tokens past the expert's capacity and does not send them.
    return routing_index::kept_count(at, routed);
}

// expert -> bucket, and the inverse chip -> experts. Every chip on the axis builds the same inverse from the
// replicated dispatch table, so a forwarding chip expands an (origin, destination) descriptor into the same
// experts_per_chip chunks the writing chip did.
void build_expert_buckets(const Scratch& c) {
    for (uint32_t i = 0; i < ct.extent * ct.experts_per_chip; i++) {
        c.chip_experts[i] = 0;
    }
    for (uint32_t pos = 0; pos < ct.extent; pos++) {
        c.pos_fill[pos] = 0;
    }
    // Includes the table's sentinel column, so a padded token's lookup resolves to BUCKET_NOT_HERE.
    //
    // The host checks the table's width but not its values. The guard keeps a bad position from writing into
    // the next position's buckets or past the block. A skipped entry leaves a chip_experts bucket at expert 0; that
    // bucket is sized run_len(my_pos, 0), fills nothing, and each of its pages becomes a duplicate write of
    // token 0 to page 0 (see merge_routing_index), unless the double-counted run pushes the total past
    // max_records, in which case the clamp in size_buckets applies (see there).
    for (uint32_t e = 0; e <= ct.num_routed_experts; e++) {
        const int32_t table_pos = c.table[e];
        if (table_pos < 0 || (uint32_t)table_pos >= ct.extent ||
            c.pos_fill[(uint32_t)table_pos] >= ct.experts_per_chip) {
            c.expert_bucket[e] = dspf2d::BUCKET_NOT_HERE;
            continue;
        }
        const uint32_t pos = (uint32_t)table_pos;
        const uint32_t j = c.pos_fill[pos];
        c.pos_fill[pos] = j + 1u;
        const uint32_t bucket = pos * ct.experts_per_chip + j;
        c.chip_experts[bucket] = e;
        c.expert_bucket[e] = bucket;
    }
    for (uint32_t pos = 0; pos < ct.extent; pos++) {
        // Chunk groups are sized as experts_per_chip chunks, so every chip must host exactly that many.
        ASSERT(c.pos_fill[pos] == ct.experts_per_chip);
    }
}

// Bucket b holds records [bucket_start[b], bucket_start[b + 1]) and starts at output page first_page[b].
// Its length is run_len from the offsets table, the same number every chunk length uses, so bucket sizes and
// chunk sizes always agree.
void size_buckets(const Scratch& c) {
    const uint32_t n_buckets = ct.extent * ct.experts_per_chip;
    // The block holds one record per (token, topk index). The clamp keeps an offsets table inconsistent with
    // the input from overrunning L1; a consistent table never reaches it. If it does fire, this chip sends
    // fewer pages than the downstream chips' chunk_len expects, and they hang.
    const uint32_t max_records = ct.seq_len * ct.topk;
    uint32_t at = 0;
    for (uint32_t b = 0; b < n_buckets; b++) {
        const uint32_t e = c.chip_experts[b];
        c.first_page[b] = c.offsets[ct.my_pos * ct.num_routed_experts + e];
        c.bucket_start[b] = at;
        const uint32_t n = run_len(c, ct.my_pos, e);
        at = (at + n > max_records) ? max_records : at + n;
    }
    c.bucket_start[n_buckets] = at;
}

// Checks that every bucket was filled to the length size_buckets gave it. The kept picks of a bucket are the
// first of the picks routed to it, in token order, which is what the RISCs wrote.
//
// A bucket can fall short only if the offsets table disagrees with the indices. Its length still comes from
// run_len, so send and receive counts match and nothing hangs, but its unwritten records would hold stale
// scratch whose `page` word becomes a write to an arbitrary address. They are zeroed, so each becomes a
// duplicate write of token 0 to page 0: wrong pages, no corruption elsewhere.
void merge_routing_index(const Scratch& c) {
    for (uint32_t b = 0; b < routing_index::num_buckets(); b++) {
        uint32_t routed = 0;
        for (uint32_t risc = 0; risc < routing_index::RISCS; risc++) {
            routed += routing_index::risc_view(c, risc).cnt[b];
        }
        const uint32_t kept = routing_index::kept_count(c.first_page[b], routed);
        const uint32_t len = c.bucket_start[b + 1u] - c.bucket_start[b];
        const uint32_t fill = c.bucket_start[b] + (kept < len ? kept : len);
        ASSERT(fill == c.bucket_start[b + 1u]);
        for (uint32_t at = fill; at < c.bucket_start[b + 1u]; at++) {
            volatile tt_l1_ptr uint32_t* rec = c.records + at * dspf2d::record_words();
            rec[0] = 0;
            rec[1] = 0;
            rec[2] = 0;
        }
    }
}

// With a TILE input, `in_acc` addresses a staging buffer that the untilize cores write, so no token may be
// read until every tile row has landed. The wait sits just before the first token read so that untilize
// runs in parallel with the routing index build. If this zone is not near zero, untilize is the critical
// path and needs more untilizer cores (UNTILIZERS_PER_LINK).
void wait_for_untilize() {
    if constexpr (ct.untilize_tile_rows > 0) {
        DeviceZoneScopedN("dspf2d_wait_untilize");
        volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.untilize_sem_addr);
        while (true) {
            // Another core increments this counter, so drop any cached copy before each read.
            invalidate_l1_cache();
            if (*landed >= ct.untilize_tile_rows) {
                return;
            }
        }
    }
}

using routing_index::slice_begin;

// The reader -> sender queue of queue_depth entries. Two monotonic counters with one writer each: this side
// owns `filled`, the sender owns `freed`, and each uses the difference, so neither needs an atomic
// read-modify-write. `claimed` runs ahead of `published` so several token reads are in flight at once.
struct TokenQueue {
    uint32_t claimed = 0;
    // Entries whose fwd_meta is written. Only these are published: when a blocked claim flushes, other
    // claimed entries may not have their command word yet.
    uint32_t ready = 0;
    uint32_t published = 0;

    uint32_t claim_entry() {
        volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.freed_addr);
        while (true) {
            invalidate_l1_cache();
            if (claimed - *freed < ct.queue_depth) {
                return claimed++ % ct.queue_depth;
            }
            // Publish before blocking, or the sender and this reader wait on each other.
            flush_publish();
        }
    }

    // The entry's fwd_meta is written; it may now be published.
    void mark_ready() { ready++; }

    // Returns the capacity of claims never marked ready. The sender never saw them.
    void release_unready() {
        ASSERT(claimed >= ready);
        claimed = ready;
    }

    void flush_publish() {
        if (published == ready) {
            return;
        }
        // The tokens have to be in L1 before the sender is told the entries are filled.
        noc_async_read_barrier();
        noc_semaphore_inc(get_noc_addr(ct.filled_addr), ready - published);
        published = ready;
    }
};

// chunk: one stream's slice of the run for one (origin, expert); a descriptor expands into experts_per_chip
// chunks, one per expert on the destination.
uint32_t chunk_len(const Scratch& c, uint32_t origin_pos, uint32_t e, uint32_t idx, uint32_t count) {
    const uint32_t n = run_len(c, origin_pos, e);
    return slice_begin(n, idx + 1, count) - slice_begin(n, idx, count);
}

// Page of one CHANNEL of this stream's section. Each channel is its own ring of ct.vc_pages pages, written
// by one producer and read by one consumer, so a running page count is still all the two chips agree on --
// there are simply num_vcs of them now instead of one flat section. Wrapping costs nothing: both sides
// address the region a page at a time, so nothing ever straddles the wrap.
uint32_t fwd_page(uint32_t vc, uint32_t page_in_vc) {
    return (ct.stream * ct.num_vcs + vc) * ct.vc_pages + page_in_vc % ct.vc_pages;
}

// Pages a reader sits on before posting them back upstream. Capped at a quarter of the channel, because a
// threshold comparable to the channel's depth starves the producer: hold 15 credits back on a 17-page
// channel and the chip upstream sees two pages free and stops. With capacity a free parameter, every
// constant measured in pages has to scale with it.
uint32_t credit_flush_at() {
    const uint32_t quarter = ct.vc_pages / 4;
    const uint32_t cap = dspf2d::CREDIT_RETURN_BATCH;
    return quarter < 1 ? 1u : (quarter < cap ? quarter : cap);
}

// Pages that will arrive in channel d: descriptor d's chunks, summed over the experts the destination
// hosts. A channel IS a descriptor's movement class, so the two are the same index -- which is why nothing
// here has to map between them.
uint32_t channel_expected(const Scratch& c, uint32_t d) {
    const uint32_t base = ct.in_descriptors_base + d * dspf2d::CHUNK_DESCRIPTOR_WORDS;
    const uint32_t origin = kernel_compile_time_args[base + 0];
    const uint32_t dst = kernel_compile_time_args[base + 1];
    const uint32_t idx = kernel_compile_time_args[base + 2];
    const uint32_t cnt = kernel_compile_time_args[base + 3];
    uint32_t total = 0;
    for (uint32_t j = 0; j < ct.experts_per_chip; j++) {
        const uint32_t e = c.chip_experts[dst * ct.experts_per_chip + j];
        total += chunk_len(c, origin, e, idx, cnt);
    }
    return total;
}

uint32_t entry_addr(uint32_t entry) { return ct.queue_addr + entry * ct.entry_stride(); }

volatile tt_l1_ptr dspf2d::FwdMetadata* entry_meta(uint32_t entry) {
    return reinterpret_cast<volatile tt_l1_ptr dspf2d::FwdMetadata*>(entry_addr(entry) + ct.token_size_bytes);
}

// Per-channel accounting, and the scheduler that moves pages between channels.
//
// Three running counts per channel, each with a single writer. `consumed` is what this reader has lifted
// out of its own channel; `out_page` is what it has placed in the DOWNSTREAM chip's channel, which is what
// that chip's credits are read against; `credits_pending` is what it owes the chip upstream and has not
// posted yet.
template <typename FwdAcc>
struct Channels {
    const Scratch& c;
    TokenQueue& queue;
    const FwdAcc& fwd_acc;

    uint32_t expected[dspf2d::MAX_VCS] = {};
    uint32_t consumed[dspf2d::MAX_VCS] = {};
    uint32_t out_page[dspf2d::MAX_VCS] = {};
    uint32_t credits_pending[dspf2d::MAX_VCS] = {};

    void plan_arrivals() {
        for (uint32_t v = 0; v < ct.num_vcs; v++) {
            expected[v] = channel_expected(c, v);
        }
    }

    // Free space in the downstream chip's channel `vc`, in pages.
    uint32_t downstream_credit(uint32_t vc) {
        volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.vc_freed_addr(vc));
        invalidate_l1_cache();
        return ct.vc_pages - (out_page[vc] - *freed);
    }

    // How much of `want` may be staged into channel `vc` right now, at least one. While that channel is full
    // we do OTHER work rather than spin -- which is the whole point of splitting the section up: some channel
    // always has a page it can move, because the ones that deliver to the next chip's output need no space
    // anywhere and so can always drain.
    uint32_t reserve_downstream(uint32_t vc, uint32_t want) {
        uint32_t credit = downstream_credit(vc);
        while (credit == 0) {
            queue.flush_publish();  // never make the sender wait on entries we are holding
            if (!serve_one_channel()) {
                flush_credits();  // nothing of ours to move; make sure what we owe is posted
            }
            credit = downstream_credit(vc);
        }
        return credit < want ? credit : want;
    }

    // Hand `n` pages of our channel `vc` back to the chip that wrote them. It cannot be told directly -- a
    // stream's fabric connection only points downstream -- so the credit goes to the opposite direction's
    // worker on this chip, whose sender is aimed at exactly that chip and carries it across.
    void credit_upstream(uint32_t vc, uint32_t n) {
        credits_pending[vc] += n;
        if (credits_pending[vc] >= credit_flush_at()) {
            flush_credits();
        }
    }

    void flush_credits() {
        for (uint32_t v = 0; v < ct.num_vcs; v++) {
            if (credits_pending[v] > 0) {
                noc_semaphore_inc(
                    get_noc_addr(ct.credit_outbox_noc_x, ct.credit_outbox_noc_y, ct.vc_outbox_addr(v)),
                    credits_pending[v]);
                credits_pending[v] = 0;
            }
        }
    }

    // Lift `k` pages out of channel `v` and push them one hop: delivered to the next chip's output when this
    // was their last hop, otherwise staged into that chip's channel `next`.
    void take_from_channel(uint32_t v, uint32_t next, uint32_t k) {
        uint32_t claimed[dspf2d::BATCH];
        for (uint32_t j = 0; j < k; j++) {
            claimed[j] = queue.claim_entry();
            noc_async_read(
                fwd_acc.get_noc_addr(fwd_page(v, consumed[v] + j)),
                entry_addr(claimed[j]),
                ct.token_size_bytes + dspf2d::FWD_EXTRA_BYTES);
        }
        noc_async_read_barrier();  // fwd_meta decides the next hop, so it must have landed
        // The reads bypass the data cache and these entries held other pages before, so a cached fwd_meta
        // could be stale.
        invalidate_l1_cache();
        for (uint32_t j = 0; j < k; j++) {
            volatile tt_l1_ptr dspf2d::FwdMetadata* fwd_meta = entry_meta(claimed[j]);
            if (next == dspf2d::VC_NONE) {
                fwd_meta->cmd = dspf2d::CMD_FINAL_WRITE;
                fwd_meta->this_addr = fwd_meta->final_payload_addr;
            } else {
                fwd_meta->cmd = dspf2d::CMD_FORWARD;
                fwd_meta->vc = next;
                fwd_meta->this_addr = fwd_acc.get_noc_addr(fwd_page(next, out_page[next]++));
            }
            queue.mark_ready();
        }
        consumed[v] += k;
        // The barrier above is what makes these pages free -- their tokens are in L1, so the chip that wrote
        // them may write them again.
        credit_upstream(v, k);
    }

    // Move up to one batch out of the highest-priority channel that can move anything, and say whether
    // anything moved. A channel can move when pages have arrived in it and either it delivers to the next
    // chip's output -- needing no space at all -- or the channel it feeds has room.
    bool serve_one_channel() {
        for (uint32_t v = 0; v < ct.num_vcs; v++) {
            const uint32_t remaining = expected[v] - consumed[v];
            if (remaining == 0) {
                continue;
            }
            volatile tt_l1_ptr uint32_t* arrived =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.vc_arrived_addr(v));
            invalidate_l1_cache();
            const uint32_t avail = *arrived - consumed[v];
            if (avail == 0) {
                continue;
            }
            uint32_t k = avail < remaining ? avail : remaining;
            if (k > ct.batch) {
                k = ct.batch;
            }
            const uint32_t next = ct.vc_next(v);
            if (next != dspf2d::VC_NONE) {
                const uint32_t credit = downstream_credit(next);
                if (credit == 0) {
                    continue;  // blocked; some other channel will not be
                }
                if (k > credit) {
                    k = credit;
                }
            }
            take_from_channel(v, next, k);
            return true;
        }
        return false;
    }

    bool all_drained() const {
        for (uint32_t v = 0; v < ct.num_vcs; v++) {
            if (consumed[v] != expected[v]) {
                return false;
            }
        }
        return true;
    }

    // Keep serving until every channel has passed on everything that will arrive in it.
    void drain() {
        while (!all_drained()) {
            if (!serve_one_channel()) {
                queue.flush_publish();
                flush_credits();
            }
        }
        queue.flush_publish();
        flush_credits();
    }
};

// This chip's own tokens for its remote destinations.
template <typename InAcc, typename OutAcc, typename MetaAcc, typename FwdAcc>
void own_phase(
    const Scratch& c,
    TokenQueue& queue,
    Channels<FwdAcc>& channels,
    const InAcc& in_acc,
    const OutAcc& out_acc,
    const MetaAcc& meta_acc,
    const FwdAcc& fwd_acc) {
    // Own assignments, furthest first. The nearest is the downstream chip: one hop, straight into its
    // output, so it needs no channel. Anything further is staged into the downstream chip's channel for
    // that distance, which the host named in own_vc(a).
    for (uint32_t a = 0; a < ct.num_own; a++) {
        const uint32_t base = ct.assignment_base + a * dspf2d::ASSIGNMENT_WORDS;
        const uint32_t dst_chip = kernel_compile_time_args[base + 0];
        const uint32_t dst_pos = kernel_compile_time_args[base + 1];
        const uint32_t split_idx = kernel_compile_time_args[base + 2];
        const uint32_t split_count = kernel_compile_time_args[base + 3];
        const bool direct = (dst_chip == ct.downstream_chip_id);
        const uint32_t own_vc = ct.own_vc(a);
        ASSERT(direct == (own_vc == dspf2d::VC_NONE));
        for (uint32_t j = 0; j < ct.experts_per_chip; j++) {
            const uint32_t b = dst_pos * ct.experts_per_chip + j;
            const uint32_t first_record = c.bucket_start[b];
            const uint32_t n = c.bucket_start[b + 1u] - first_record;
            const uint32_t from = slice_begin(n, split_idx, split_count);
            const uint32_t to = slice_begin(n, split_idx + 1, split_count);
            uint32_t i = from;
            while (i < to) {
                // A staged run needs room in the downstream channel before any of it is aimed. Reserving a
                // span at a time keeps the credit read off the per-token path; while the channel is full
                // this serves our own channels instead of spinning, which is what keeps credits flowing
                // back to the chip upstream.
                uint32_t span = to - i;
                if (!direct) {
                    span = channels.reserve_downstream(own_vc, span);
                }
                if (span > ct.batch) {
                    span = ct.batch;
                }
                for (uint32_t k = 0; k < span; k++, i++) {
                    const uint32_t at = (first_record + i) * dspf2d::record_words();
                    const uint32_t token = c.records[at + 0];
                    const uint32_t page = c.records[at + 1];
                    const uint32_t entry = queue.claim_entry();
                    noc_async_read(in_acc.get_noc_addr(token), entry_addr(entry), ct.token_size_bytes);

                    // Both final addresses are computed here and travel with the token. Every buffer is
                    // interleaved DRAM with the same base on every chip, so a page index names the same
                    // place on any chip.
                    volatile tt_l1_ptr dspf2d::FwdMetadata* fwd_meta = entry_meta(entry);
                    fwd_meta->final_payload_addr = out_acc.get_noc_addr(page);
                    fwd_meta->final_meta_addr = meta_acc.get_noc_addr(page);
                    fwd_meta->dst_chip = dst_chip;
                    fwd_meta->meta[0] = ct.linearized_coord;
                    fwd_meta->meta[1] = token;
                    fwd_meta->meta[2] = c.records[at + 2];
                    fwd_meta->pad = 0;
                    if (direct) {
                        fwd_meta->cmd = dspf2d::CMD_FINAL_WRITE;
                        fwd_meta->this_addr = fwd_meta->final_payload_addr;
                    } else {
                        // No chunk-end term: with the section split by movement class a channel has no
                        // chunk boundaries to close, so the only reason to announce early is the sender's
                        // batch count.
                        fwd_meta->cmd = dspf2d::CMD_FORWARD;
                        fwd_meta->vc = own_vc;
                        fwd_meta->this_addr = fwd_acc.get_noc_addr(fwd_page(own_vc, channels.out_page[own_vc]++));
                    }
                    queue.mark_ready();
                }
                queue.flush_publish();
                // Serve a channel between spans, not only when a span blocks for credit. The chip upstream
                // cannot write to us again until we credit the pages it already sent, and those credits are
                // ours alone to produce -- so a reader that spends a whole own phase without looking at its
                // channels leaves every upstream chip stalled on it, which lands on the slowest chip rather
                // than on this one. Cheap when there is nothing to move: one counter read per channel.
                channels.serve_one_channel();
            }
        }
    }
}

// Tokens routed to an expert this chip hosts. They do not use the fabric: each is one DRAM read and two DRAM
// writes through queue entries used as scratch, which are never published so the sender does not send them.
// Each stream takes its own slice of every bucket; the slices cover each token exactly once.
//
// ct.batch reads are kept in flight. Entries are claimed only as tokens need them. They never become
// ready, so claim_entry's flush frees nothing here; progress relies on the sender draining what earlier
// phases published, while this phase holds at most ct.batch entries.
template <typename InAcc, typename OutAcc, typename MetaAcc>
void local_phase(
    const Scratch& c, TokenQueue& queue, const InAcc& in_acc, const OutAcc& out_acc, const MetaAcc& meta_acc) {
    // Every earlier phase ended with flush_publish, so release_unready returns exactly this phase's entries.
    ASSERT(queue.claimed == queue.ready);
    // held <= ct.batch: a new entry is claimed only when every entry already held has a pending token, and a
    // batch is written out at ct.batch pending. Each batch reuses held_entries from index 0, safe after
    // write_batch's flush.
    static_assert(ct.batch <= dspf2d::BATCH, "these arrays are sized by BATCH");
    uint32_t held_entries[dspf2d::BATCH];
    uint32_t pages[dspf2d::BATCH];
    uint32_t held = 0;     // entries this phase has claimed and keeps until it ends
    uint32_t pending = 0;  // tokens read into held_entries[0..pending) and not yet written
    // Wait for the batch's reads, write each token and its metadata to their pages, wait for departure.
    const auto write_batch = [&]() {
        noc_async_read_barrier();
        for (uint32_t i = 0; i < pending; i++) {
            const uint32_t addr = entry_addr(held_entries[i]);
            noc_async_write(addr, out_acc.get_noc_addr(pages[i]), ct.token_size_bytes);
            noc_async_write(addr + ct.token_size_bytes, meta_acc.get_noc_addr(pages[i]), dspf2d::METADATA_WIRE_BYTES);
        }
        // The entries are the write sources, so they may be refilled once the writes have left L1.
        noc_async_writes_flushed();
        pending = 0;
    };
    for (uint32_t j = 0; j < ct.experts_per_chip; j++) {
        const uint32_t b = ct.my_pos * ct.experts_per_chip + j;
        const uint32_t first_record = c.bucket_start[b];
        const uint32_t n = c.bucket_start[b + 1u] - first_record;
        const uint32_t from = slice_begin(n, ct.stream, 2 * ct.num_links);
        const uint32_t to = slice_begin(n, ct.stream + 1, 2 * ct.num_links);
        for (uint32_t i = from; i < to; i++) {
            if (pending == held) {  // every held entry has a pending token, so claim one more
                held_entries[held] = queue.claim_entry();
                entry_meta(held_entries[held])->pad = 0;  // goes to the metadata page; stays zero for the phase
                held++;
            }
            const uint32_t at = (first_record + i) * dspf2d::record_words();
            const uint32_t token = c.records[at + 0];
            const uint32_t entry = held_entries[pending];
            noc_async_read(in_acc.get_noc_addr(token), entry_addr(entry), ct.token_size_bytes);
            volatile tt_l1_ptr dspf2d::FwdMetadata* fwd_meta = entry_meta(entry);
            fwd_meta->meta[0] = ct.linearized_coord;
            fwd_meta->meta[1] = token;
            fwd_meta->meta[2] = c.records[at + 2];
            pages[pending] = c.records[at + 1];
            if (++pending == ct.batch) {
                write_batch();
            }
        }
    }
    if (pending > 0) {
        write_batch();
    }
    // The output pages need completion, and the firmware does not wait for writes at kernel end.
    noc_async_write_barrier();
    queue.release_unready();
}

// Tokens the upstream chip on this stream writes straight into this chip's output over the whole launch. The
// host lists those chunks from the assignments and descriptors the upstream chip's program is built from
// (incoming_final_descriptors), and their lengths come from the same replicated tables the upstream chip sizes its
// sends with, so this equals what the upstream sender counts. Short of it, the wait below never ends.
uint32_t expected_final_writes(const Scratch& c) {
    uint32_t total = 0;
    for (uint32_t d = 0; d < ct.num_final; d++) {
        const uint32_t base = ct.final_descriptors_base + d * dspf2d::CHUNK_DESCRIPTOR_WORDS;
        const uint32_t origin = kernel_compile_time_args[base + 0];
        const uint32_t dst = kernel_compile_time_args[base + 1];
        const uint32_t idx = kernel_compile_time_args[base + 2];
        const uint32_t cnt = kernel_compile_time_args[base + 3];
        for (uint32_t j = 0; j < ct.experts_per_chip; j++) {
            total += chunk_len(c, origin, c.chip_experts[dst * ct.experts_per_chip + j], idx, cnt);
        }
    }
    return total;
}

// Block until the upstream sender has counted every token it wrote straight into this chip's output. The
// sender's drain only holds back the upstream chip's own program, so without this the next op on this chip
// could read an output whose last tokens are still on the way. The signal is a flushing atomic, so the far
// router has finished the writes ahead of it before they are counted.
//
// `>=` and a subtraction rather than `==` and a reset: like the channel counters, this stays right without
// relying on the launch credit to keep the next launch's signals out.
void wait_for_final_writes(const Scratch& c) {
    const uint32_t expected = expected_final_writes(c);
    volatile tt_l1_ptr uint32_t* final_arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.final_sem_addr);
    while (true) {
        // The upstream chip increments this counter, so drop any cached copy before each read.
        invalidate_l1_cache();
        if (*final_arrived >= expected) {
            break;
        }
    }
    noc_semaphore_inc(get_noc_addr(ct.final_sem_addr), 0u - expected);
}

}  // namespace

void kernel_main() {
    const Scratch c = routing_index::layout_scratch();
    ASSERT(c.end <= ct.filled_addr);  // the scratch must not run into the semaphores

    {
        DeviceZoneScopedN("dspf2d_index");
        // The TRISCs wait for the tables; once they are in, all four RISCs build the index together.
        routing_index::run_on_reader(c, [&] {
            read_control_tables(c);
            read_indices(c);
            build_expert_buckets(c);
            size_buckets(c);
        });
        {
            // How long this RISC waits for the slowest of the other three: the imbalance of the split.
            DeviceZoneScopedN("dspf2d_index_wait");
            routing_index::wait_all_riscs(dspf2d::kRiscFilled);
        }
        {
            DeviceZoneScopedN("dspf2d_index_merge");
            merge_routing_index(c);
        }
    }

    const auto in_acc =
        TensorAccessor(dspf2d::ReaderCtArgs::in_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kInputAddr));
    const auto out_acc = TensorAccessor(
        dspf2d::ReaderCtArgs::out_payload_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kOutPayloadAddr));
    const auto meta_acc =
        TensorAccessor(dspf2d::ReaderCtArgs::out_meta_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kOutMetaAddr));

    // Not const: Channels is instantiated on this type and own_phase deduces the same one from a
    // `const FwdAcc&` parameter, which would deduce the unqualified type and not match a const decltype.
    auto fwd_acc = TensorAccessor(dspf2d::ReaderCtArgs::fwd_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kFwdAddr));

    TokenQueue queue;
    Channels<decltype(fwd_acc)> channels{c, queue, fwd_acc};
    {
        DeviceZoneScopedN("dspf2d_starts");
        channels.plan_arrivals();
    }
    wait_for_untilize();  // the own phase is the first thing that reads a token
    {
        // Own tokens first: the chips downstream are waiting on them, and staging one blocks only when the
        // channel it goes to is full, at which point this serves our own channels instead.
        DeviceZoneScopedN("dspf2d_own");
        own_phase(c, queue, channels, in_acc, out_acc, meta_acc, fwd_acc);
        queue.flush_publish();
    }
    {
        // Everything still owed to us, in channel priority order.
        DeviceZoneScopedN("dspf2d_forward");
        channels.drain();
    }
    // Local tokens go last because they never leave the chip, while earlier pages have chips downstream
    // waiting.
    {
        DeviceZoneScopedN("dspf2d_local");
        local_phase(c, queue, in_acc, out_acc, meta_acc);
    }

    {
        DeviceZoneScopedN("dspf2d_end");
        const uint32_t end_entry = queue.claim_entry();
        entry_meta(end_entry)->cmd = dspf2d::CMD_END;
        queue.mark_ready();
        queue.flush_publish();

        // Everything we owed upstream is posted before the courier is told to stop, or those pages are
        // stranded with the chip that wrote them still waiting on them.
        channels.flush_credits();
        noc_async_atomic_barrier();
        noc_semaphore_inc(get_noc_addr(ct.credit_outbox_noc_x, ct.credit_outbox_noc_y, ct.credit_done_addr), 1);

        // After CMD_END and credit_done, so neither our own sender nor the courier waits on this. The upstream
        // sender we wait on needs only its own reader and the credits already posted, never this reader's end,
        // so this cannot close a cycle around the ring.
        {
            DeviceZoneScopedN("dspf2d_wait_final");
            wait_for_final_writes(c);
        }

        // Leave the cross-chip counters as the next launch expects them. Both are bumped from another chip,
        // so they are drained by subtraction rather than zeroed: the NoC only has an atomic add, so this
        // adds the two's complement, and both counts are bounded by the channel, so the wrap is exact.
        //
        // The credits are WAITED for first. The launch credit keeps the upstream sender's NEXT launch out
        // until this chip has started it, but it says nothing about credits still in flight for pages this
        // launch placed downstream: one landing after the subtraction would sit on the next launch's
        // counter and let that launch run a whole channel ahead of a reader that has not got there, which
        // is a silent overwrite of live tokens rather than a hang. It costs nothing on the common path --
        // by here the downstream chip has long since consumed what we sent it.
        for (uint32_t v = 0; v < ct.num_vcs; v++) {
            volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.vc_freed_addr(v));
            invalidate_l1_cache();
            while (*freed < channels.out_page[v]) {
                invalidate_l1_cache();
            }
            noc_semaphore_inc(get_noc_addr(ct.vc_arrived_addr(v)), (uint32_t)(0u - channels.consumed[v]));
            noc_semaphore_inc(get_noc_addr(ct.vc_freed_addr(v)), (uint32_t)(0u - channels.out_page[v]));
        }
        noc_async_atomic_barrier();
        if constexpr (ct.untilize_tile_rows > 0) {
            // Safe only here: the wait before the first token read saw all untilize_tile_rows increments,
            // so no writer still signals this counter.
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.untilize_sem_addr), 0);
        }
    }
}
