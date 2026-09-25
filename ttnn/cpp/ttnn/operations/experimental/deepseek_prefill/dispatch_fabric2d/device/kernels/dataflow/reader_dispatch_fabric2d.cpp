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

// Page offset of each chunk of a descriptor list within a stream's fwd_section. The section holds only pages,
// so a chunk starts at the sum of the lengths before it. Both chips of a section run this over the same list;
// the host function validate_descriptor_agreement guarantees that.
uint32_t chunk_starts(const Scratch& c, uint32_t block_base, volatile tt_l1_ptr uint32_t* start) {
    uint32_t at = 0;
    for (uint32_t d = 0; d < ct.num_forward; d++) {
        const uint32_t base = block_base + d * dspf2d::CHUNK_DESCRIPTOR_WORDS;
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
    // The host sizes the section without knowing these lengths; this checks the data fits.
    ASSERT(at <= ct.fwd_pages_per_stream);
    return at;
}

uint32_t entry_addr(uint32_t entry) { return ct.queue_addr + entry * ct.entry_stride(); }

volatile tt_l1_ptr dspf2d::FwdMetadata* entry_meta(uint32_t entry) {
    return reinterpret_cast<volatile tt_l1_ptr dspf2d::FwdMetadata*>(entry_addr(entry) + ct.token_size_bytes);
}

// This chip's own tokens for its remote destinations.
template <typename InAcc, typename OutAcc, typename MetaAcc, typename FwdAcc>
void own_phase(
    const Scratch& c,
    TokenQueue& queue,
    const InAcc& in_acc,
    const OutAcc& out_acc,
    const MetaAcc& meta_acc,
    const FwdAcc& fwd_acc,
    uint32_t my_fwd_section) {
    // Own assignments, furthest first. The nearest is the downstream chip: one hop, straight into its output.
    // Anything further goes into the downstream chip's fwd_section. Own assignment a is outgoing descriptor a,
    // because dispatch_fabric2d_assignments.cpp emits both lists furthest-first. The last own assignment is
    // the downstream chip, written directly with no outgoing descriptor, so forward_phase's descriptors start
    // at num_own - 1.
    for (uint32_t a = 0; a < ct.num_own; a++) {
        const uint32_t base = ct.assignment_base + a * dspf2d::ASSIGNMENT_WORDS;
        const uint32_t dst_chip = kernel_compile_time_args[base + 0];
        const uint32_t dst_pos = kernel_compile_time_args[base + 1];
        const uint32_t split_idx = kernel_compile_time_args[base + 2];
        const uint32_t split_count = kernel_compile_time_args[base + 3];
        const bool direct = (dst_chip == ct.downstream_chip_id);
        for (uint32_t j = 0; j < ct.experts_per_chip; j++) {
            const uint32_t b = dst_pos * ct.experts_per_chip + j;
            const uint32_t first_record = c.bucket_start[b];
            const uint32_t n = c.bucket_start[b + 1u] - first_record;
            const uint32_t from = slice_begin(n, split_idx, split_count);
            const uint32_t to = slice_begin(n, split_idx + 1, split_count);
            const uint32_t out_base = direct ? 0 : c.out_start[a * ct.experts_per_chip + j];
            for (uint32_t i = from; i < to; i++) {
                const uint32_t at = (first_record + i) * dspf2d::record_words();
                const uint32_t token = c.records[at + 0];
                const uint32_t page = c.records[at + 1];
                const uint32_t entry = queue.claim_entry();
                noc_async_read(in_acc.get_noc_addr(token), entry_addr(entry), ct.token_size_bytes);

                // Both final addresses are computed here and travel with the token. Every buffer is
                // interleaved DRAM with the same base on every chip, so a page index names the same place
                // on any chip.
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
                    // A chunk's last page forces the downstream signal; without it the downstream reader
                    // hangs waiting for the chunk to finish.
                    fwd_meta->cmd = (i + 1 == to) ? dspf2d::CMD_FORWARD_END : dspf2d::CMD_FORWARD;
                    fwd_meta->this_addr = fwd_acc.get_noc_addr(my_fwd_section + out_base + (i - from));
                }
                queue.mark_ready();
            }
        }
    }
}

// Pages this stream forwards: read from its fwd_section and sent one hop further, or delivered if the
// downstream chip is their destination.
template <typename FwdAcc>
uint32_t forward_phase(
    const Scratch& c, TokenQueue& queue, const FwdAcc& fwd_acc, uint32_t my_fwd_section, uint32_t downstream_pos) {
    uint32_t consumed = 0;  // pages taken out of this stream's section, which is what end_stream gives back
    // Arriving descriptors, in the order upstream wrote them.
    uint32_t out_d = ct.num_own - 1;  // own assignments occupy the first num_own - 1 outgoing descriptors
    for (uint32_t d = 0; d < ct.num_forward; d++) {
        const uint32_t base = ct.in_descriptors_base + d * dspf2d::CHUNK_DESCRIPTOR_WORDS;
        const uint32_t origin = kernel_compile_time_args[base + 0];
        const uint32_t dst_pos = kernel_compile_time_args[base + 1];
        const uint32_t idx = kernel_compile_time_args[base + 2];
        const uint32_t cnt = kernel_compile_time_args[base + 3];
        const bool continues = (dst_pos != downstream_pos);
        const uint32_t dst_chip = kernel_compile_time_args[ct.ring_chip_ids_base + dst_pos];
        const uint32_t this_out_d = continues ? out_d++ : 0;
        for (uint32_t j = 0; j < ct.experts_per_chip; j++) {
            const uint32_t e = c.chip_experts[dst_pos * ct.experts_per_chip + j];
            const uint32_t len = chunk_len(c, origin, e, idx, cnt);
            consumed += len;
            const uint32_t in_base = c.in_start[d * ct.experts_per_chip + j];
            const uint32_t out_base = continues ? c.out_start[this_out_d * ct.experts_per_chip + j] : 0;
            volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.fwd_sem_addr);
            // Pages are read a batch at a time so several DRAM reads are in flight. BATCH <= QUEUE_DEPTH is
            // enough to avoid deadlock, since claim_entry publishes before it blocks; BATCH <= QUEUE_DEPTH/2
            // lets the sender drain one batch while the next is read.
            for (uint32_t p = 0; p < len;) {
                const uint32_t n = (len - p < ct.batch) ? (len - p) : ct.batch;
                uint32_t claimed[dspf2d::BATCH];
                for (uint32_t i = 0; i < n; i++) {
                    // Upstream fills the section in order, so a page is ready once the arrival count
                    // passes its offset. An upstream chip increments the counter, so invalidate first.
                    while (true) {
                        invalidate_l1_cache();
                        if (*arrived > in_base + p + i) {
                            break;
                        }
                        queue.flush_publish();  // let our own sender work while we wait on upstream
                    }
                    claimed[i] = queue.claim_entry();
                    noc_async_read(
                        fwd_acc.get_noc_addr(my_fwd_section + in_base + p + i),
                        entry_addr(claimed[i]),
                        ct.token_size_bytes + dspf2d::FWD_EXTRA_BYTES);
                }
                noc_async_read_barrier();  // fwd_meta decides the next hop, so it must have landed
                // The reads bypass the data cache and these entries held other pages before, so a cached
                // fwd_meta could be stale.
                invalidate_l1_cache();

                for (uint32_t i = 0; i < n; i++) {
                    volatile tt_l1_ptr dspf2d::FwdMetadata* fwd_meta = entry_meta(claimed[i]);
                    // Every page of (origin, dst_pos) shares one destination, so whether this hop is the
                    // last comes from the descriptor. fwd_meta is only checked: an unwritten one reads as
                    // chip 0, which is also a valid chip id.
                    ASSERT(fwd_meta->dst_chip == (uint64_t)dst_chip);
                    if (!continues) {
                        fwd_meta->cmd = dspf2d::CMD_FINAL_WRITE;
                        fwd_meta->this_addr = fwd_meta->final_payload_addr;
                    } else {
                        fwd_meta->cmd = (p + i + 1 == len) ? dspf2d::CMD_FORWARD_END : dspf2d::CMD_FORWARD;
                        fwd_meta->this_addr = fwd_acc.get_noc_addr(my_fwd_section + out_base + p + i);
                    }
                    queue.mark_ready();
                }
                p += n;
            }
        }
    }
    return consumed;
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

    const auto fwd_acc =
        TensorAccessor(dspf2d::ReaderCtArgs::fwd_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kFwdAddr));
    const uint32_t my_fwd_section = ct.stream * ct.fwd_pages_per_stream;

    // The downstream chip's position on the axis. A page bound for it is delivered, not forwarded.
    uint32_t downstream_pos = 0;
    for (uint32_t pos = 0; pos < ct.extent; pos++) {
        if (kernel_compile_time_args[ct.ring_chip_ids_base + pos] == ct.downstream_chip_id) {
            downstream_pos = pos;
        }
    }

    TokenQueue queue;
    {
        DeviceZoneScopedN("dspf2d_starts");
        chunk_starts(c, ct.in_descriptors_base, c.in_start);
        chunk_starts(c, ct.out_descriptors_base, c.out_start);
    }
    wait_for_untilize();  // the own phase is the first thing that reads a token
    {
        DeviceZoneScopedN("dspf2d_own");
        own_phase(c, queue, in_acc, out_acc, meta_acc, fwd_acc, my_fwd_section);
        queue.flush_publish();
    }
    uint32_t consumed = 0;  // pages this stream took out of its fwd_section
    {
        DeviceZoneScopedN("dspf2d_forward");
        consumed = forward_phase(c, queue, fwd_acc, my_fwd_section, downstream_pos);
        queue.flush_publish();
    }
    // Local tokens go last because they never leave the chip, while earlier pages have chips downstream waiting.
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

        noc_async_atomic_barrier();
        // Subtract what this launch consumed instead of zeroing: the upstream chip may already be signalling
        // for the next launch, and zeroing would drop those signals and hang it. The NoC only has an atomic
        // add, so this adds the two's complement; both counts are bounded by the section, so the wrap is exact.
        noc_semaphore_inc(get_noc_addr(ct.fwd_sem_addr), (uint32_t)(0u - consumed));
        noc_async_atomic_barrier();
        if constexpr (ct.untilize_tile_rows > 0) {
            // Safe only here: the wait before the first token read saw all untilize_tile_rows increments,
            // so no writer still signals this counter.
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.untilize_sem_addr), 0);
        }
    }
}
