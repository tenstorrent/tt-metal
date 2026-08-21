// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel (reader RISC, NOC_0). Feeds the L1 ring the sender sends from and makes ALL routing
// decisions. The sender writes L1 -> eth over NOC_1, so the two do not contend.
//
// Every packet the sender sends travels exactly one hop, to the chip across its own cable, so a token
// bound further is staged into that neighbour's forwarding buffer and re-sent from there by the neighbour's
// reader on the same (plane, direction). Hence two phases:
//
//   A. Its OWN assignments — its plane's share of this chip's movements. Each destination is either our
//      immediate neighbour (=> CMD_FINAL_WRITE, the neighbour writes it straight into its output region) or
//      further away (=> CMD_FORWARD into the neighbour's forwarding buffer). A forwarding assignment ends
//      with a SENTINEL token so the downstream reader knows where the chunk stops.
//
//   B. The chunks that arrived in OUR quarter of the forwarding buffer, written by the upstream chip's
//      sender on the same (plane, direction). Each is pushed one more hop: final-write if its destination
//      is now our neighbour, re-forward otherwise. A sentinel is passed on only when re-forwarding — a final
//      write leaves no chunk downstream to terminate.
//
// The stream ends with a CMD_END slot, because its length is not knowable up front: how much this core
// re-forwards depends on chunk sizes decided upstream.
//
// The forwarding-buffer page layout is deliberately [token][final_addr][dst_chip], which is EXACTLY the
// first token_size+16 bytes of a ring slot. Re-forwarding is therefore a single DRAM read of
// token_size+16 bytes straight into the slot base: payload and both metadata words land where the sender
// already expects them, with no unpacking.
//
// The ring is hand-rolled rather than a metal CB because the packet headers live at fixed offsets from the
// L1 allocator base — where a CB would be allocated — and a sender on another chip addresses them there.
//
// Two monotonic single-writer counters, each bumped by a NoC atomic to our OWN core (the proven idiom for
// cross-RISC visibility on one core; a plain store can sit in a write buffer where the other RISC will not
// see it). `filled` is ours, `freed` is the sender's, and each side keeps its own local count and works
// on the difference, so there is no read-modify-write to race.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "combine_fabric2d_reader_ct_args.hpp"

constexpr cmbf2d::ReaderCtArgs ct{};

// Re-forwarding reads the token AND the two metadata words that follow it in the page.
constexpr uint32_t fwd_read_bytes = ct.token_size_bytes + cmbf2d::FWD_EXTRA_BYTES;

// Routing metadata is PREFETCHED a batch at a time into pads at the front of the control region. Pads,
// not one buffer, because a DRAM read needs a 64-byte-aligned L1 destination on Blackhole
// (LOG_BASE_2_OF_DRAM_ALIGNMENT = 6), and a 12-byte record per token would not keep that.
//
// Batching is what makes the per-token loop cheap. A token's destination page comes from its metadata,
// and turning that page into an address costs an interleaved-bank division — so if the metadata arrived
// per token, that arithmetic would sit AFTER the read barrier, serialised between tokens. With the batch
// already in L1 the loop is: compute the metadata, issue the token read, barrier. The arithmetic overlaps
// the 14 kB read instead of following it.
//
// The batch is capped rather than sized to the run: run lengths are data-dependent and a whole expert
// region could be tens of thousands of tokens, so a cap bounds L1 with no correctness risk. Reads are
// issued back to back and awaited once, so one batch costs one DRAM latency, not `cap` of them.
constexpr uint32_t meta_pads_addr = ct.control_addr;
constexpr uint32_t META_READ_BYTES = cmbf2d::META_PAD_STRIDE;  // fill the whole pad; the record is shorter
constexpr uint32_t control_tables_addr = ct.control_addr + ct.meta_prefetch_cap * cmbf2d::META_PAD_STRIDE;

// Page of a chunk within OUR quarter. The same formula serves both directions: the quarter index is ours
// either way, only the chip differs, and the buffer's device-local address is uniform across the mesh.
constexpr uint32_t fwd_page(uint32_t chunk, uint32_t token) {
    return ct.my_quarter * ct.fwd_chunks_per_quarter * ct.fwd_pages_per_chunk + chunk * ct.fwd_pages_per_chunk + token;
}

// Our slice of a run: of [begin, end) take [begin + n*idx/count, begin + n*(idx+1)/count). Integer
// arithmetic, so consecutive slices meet exactly — no gaps, no overlap, whatever n is.
constexpr uint32_t slice_begin(uint32_t begin, uint32_t n, uint32_t idx, uint32_t count) {
    return begin + (uint32_t)(((uint64_t)n * idx) / count);
}

uint32_t slot_addr_of(uint32_t slot) { return ct.ring_addr + slot * ct.slot_stride(); }

// A destination one hop away is written straight into its output region; anything further is staged into
// the neighbour's forwarding buffer and re-sent from there.
bool is_direct(uint32_t dst_chip_id) { return dst_chip_id == ct.nbr_chip_id; }

volatile tt_l1_ptr cmbf2d::FwdMetadata* slot_metadata(uint32_t slot) {
    return reinterpret_cast<volatile tt_l1_ptr cmbf2d::FwdMetadata*>(slot_addr_of(slot) + ct.token_size_bytes);
}

// Every DRAM buffer this kernel touches, so the phases take one argument instead of seven.
struct Dram {
    decltype(TensorAccessor(ct.dram_in_args, uint32_t{})) in;
    decltype(TensorAccessor(ct.dram_out_args, uint32_t{})) out;
    decltype(TensorAccessor(ct.dram_fwd_args, uint32_t{})) fwd;
    decltype(TensorAccessor(ct.dram_meta_args, uint32_t{})) meta;
    decltype(TensorAccessor(ct.dram_counts_args, uint32_t{})) counts;
    decltype(TensorAccessor(ct.dram_region_args, uint32_t{})) region;
    decltype(TensorAccessor(ct.dram_expert_offsets_args, uint32_t{})) expert_offsets;
};

Dram open_dram() {
    return Dram{
        TensorAccessor(ct.dram_in_args, ct.dram_in_base_addr),
        TensorAccessor(ct.dram_out_args, ct.dram_out_base_addr),
        TensorAccessor(ct.dram_fwd_args, ct.dram_fwd_base_addr),
        TensorAccessor(ct.dram_meta_args, ct.dram_meta_base_addr),
        TensorAccessor(ct.dram_counts_args, ct.dram_counts_base_addr),
        TensorAccessor(ct.dram_region_args, ct.dram_region_base_addr),
        TensorAccessor(ct.dram_expert_offsets_args, ct.dram_expert_offsets_base_addr)};
}

// Where origin chip `row`'s tokens for expert `e` start, and where they end. The runs are laid out in
// origin-chip order inside the expert's region, so one run ends where the next begins; the last one ends
// at the expert's total count past its region base, which is the only thing expert_offsets cannot say.
struct ControlTables {
    volatile tt_l1_ptr uint32_t* offsets;
    volatile tt_l1_ptr uint32_t* counts;
    volatile tt_l1_ptr uint32_t* region;

    uint32_t run_begin(uint32_t row, uint32_t e) const { return offsets[row * ct.num_routed_experts + e]; }
    uint32_t run_end(uint32_t row, uint32_t e) const {
        return row + 1 < ct.dispatch_group_size ? offsets[(row + 1) * ct.num_routed_experts + e]
                                                : region[e] + counts[e];
    }
};

// Control tensors, read once. All three are one row per page of `num_routed_experts` uint32: expert_offsets
// has one row per ORIGIN chip (it is replicated along that axis so every chip sees all of them), counts and
// region_offsets are a single row each. A few kB in total, so the whole slice comes in rather than
// cherry-picking this chip's columns — a strided gather of 4-byte words would cost more NoC transactions
// than the bytes saved.
ControlTables read_control_tables(const Dram& dram) {
    constexpr uint32_t row_bytes = ct.num_routed_experts * 4;
    ControlTables ctl{
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(control_tables_addr),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            control_tables_addr + ct.dispatch_group_size * ct.num_routed_experts * 4),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            control_tables_addr + (ct.dispatch_group_size + 1) * ct.num_routed_experts * 4)};
    for (uint32_t r = 0; r < ct.dispatch_group_size; r++) {
        noc_async_read(dram.expert_offsets.get_noc_addr(r), control_tables_addr + r * row_bytes, row_bytes);
    }
    noc_async_read(dram.counts.get_noc_addr(0), (uint32_t)ctl.counts, row_bytes);
    noc_async_read(dram.region.get_noc_addr(0), (uint32_t)ctl.region, row_bytes);
    noc_async_read_barrier();
    return ctl;
}

// Everything the phases share: the ring's flow control, where the next downstream chunk goes, and how far
// into our own quarter we have consumed.
//
// `published` counts slots ANNOUNCED to the sender; `claimed` counts slots we have taken. They differ
// because reads are issued in batches: a batch's slots are all claimed and their reads all issued before
// any of them is announced, which is what puts several 14 kB DRAM reads in flight at once instead of
// exposing one read's latency per token. The ring and both counters are continuous, so a phase or chunk
// boundary is invisible to the flow control.
//
// Deadlock is ruled out by batch <= num_l1_slots/2: at most `batch` slots are ever unannounced, so the
// sender always has at least as many announced ones to drain, and `freed` keeps advancing.
struct Reader {
    const Dram& dram;
    ControlTables ctl;
    uint32_t published = 0;
    uint32_t claimed = 0;
    // Publishing is batched so the atomic is amortised, matching the sender's batch.
    uint32_t pending_publish = 0;
    // Which chunk of the neighbour's quarter the next forwarding batch goes into. Our own forwarding
    // assignments take 0.., then our re-forwards continue from there — and the downstream reader walks its
    // chunks in exactly that order, which is what makes the two agree with no negotiation.
    uint32_t out_chunk = 0;
    // Forwarded pages (sentinels included) taken out of our quarter.
    uint32_t consumed = 0;

    void flush_publish() {
        if (pending_publish > 0) {
            noc_semaphore_inc(get_noc_addr(ct.filled_addr), pending_publish);
            pending_publish = 0;
        }
    }

    // Claim one ring slot, blocking until the sender frees one. Publishes anything pending FIRST: never
    // make the sender wait on slots we are holding back, which would deadlock us against each other.
    uint32_t claim_slot() {
        volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.freed_addr);
        invalidate_l1_cache();
        if (claimed - *freed >= ct.num_l1_slots) {
            flush_publish();
            while (true) {
                invalidate_l1_cache();
                if (claimed - *freed < ct.num_l1_slots) {
                    break;
                }
            }
        }
        return claimed++ % ct.num_l1_slots;
    }

    // Announce `n` slots whose data is known to be in L1.
    void publish_n(uint32_t n) {
        published += n;
        pending_publish += n;
        if (pending_publish >= ct.batch) {
            flush_publish();
        }
    }

    // Read the metadata for pages [lo, hi) into the pads, one page read each, awaited together: pad k then
    // holds page lo + k's record.
    void prefetch_metadata(uint32_t lo, uint32_t hi) const {
        for (uint32_t p = lo; p < hi; p++) {
            noc_async_read(
                dram.meta.get_noc_addr(p), meta_pads_addr + (p - lo) * cmbf2d::META_PAD_STRIDE, META_READ_BYTES);
        }
        noc_async_read_barrier();
    }

    // Issues one token's read and fills its metadata but does NOT barrier or announce it: the caller does
    // both once per batch, so several reads are in flight together. `chunk_page` is where the token goes in
    // the downstream forwarding chunk, and is ignored for a direct write.
    void issue_token(uint32_t dst_chip_id, uint32_t in_page, uint32_t pad, uint32_t chunk_page) {
        const uint32_t slot = claim_slot();
        volatile tt_l1_ptr cmbf2d::FwdMetadata* metadata = slot_metadata(slot);
        volatile tt_l1_ptr uint32_t* meta =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_pads_addr + pad * cmbf2d::META_PAD_STRIDE);

        noc_async_read(dram.in.get_noc_addr(in_page), slot_addr_of(slot), ct.token_size_bytes);
        // Everything below runs while that read is in flight, which is the point of prefetching the metadata.
        // Record layout: [0] linearized_coord, [1] token_idx, [2] topk_idx; the destination chip is already
        // known from the run this token came out of.
        const uint32_t out_page = meta[1] * ct.num_experts_per_tok + meta[2];
        // Computed once and then travelling with the token for however many hops it takes. Valid on any chip
        // because the output buffer's base address and interleaved bank mapping are uniform across the mesh.
        const uint64_t final_addr = dram.out.get_noc_addr(out_page);
        if (is_direct(dst_chip_id)) {
            metadata->cmd = cmbf2d::CMD_FINAL_WRITE;
            metadata->this_addr = final_addr;
        } else {
            metadata->final_addr = final_addr;
            metadata->dst_chip = (uint64_t)dst_chip_id;
            metadata->cmd = cmbf2d::CMD_FORWARD;
            metadata->this_addr = dram.fwd.get_noc_addr(fwd_page(out_chunk, chunk_page));
        }
    }

    // Stage [lo, hi) of one run: prefetch their metadata, then issue and announce them `batch` at a time.
    // Returns how many were staged, which is what advances the caller's position in the downstream chunk.
    uint32_t stage_run_slice(uint32_t dst_chip_id, uint32_t lo, uint32_t hi, uint32_t chunk_page) {
        uint32_t staged = 0;
        for (uint32_t base = lo; base < hi; base += ct.meta_prefetch_cap) {
            const uint32_t end = (hi - base) > ct.meta_prefetch_cap ? base + ct.meta_prefetch_cap : hi;
            prefetch_metadata(base, end);
            // `batch` tokens' reads in flight at a time, then one barrier and one announcement.
            for (uint32_t q = base; q < end;) {
                const uint32_t k = (end - q) > ct.batch ? ct.batch : (end - q);
                for (uint32_t j = 0; j < k; j++) {
                    issue_token(dst_chip_id, q + j, q + j - base, chunk_page + staged);
                    if (!is_direct(dst_chip_id)) {
                        staged++;
                    }
                }
                noc_async_read_barrier();  // every token of the ct.batch is in L1 before any is announced
                publish_n(k);
                q += k;
            }
        }
        return staged;
    }

    // Sentinel: terminates this chunk for the downstream reader. Carries no usable token, so no DRAM read —
    // only the metadata matters. Chunk lengths are data-dependent now, which is exactly what the sentinel
    // exists for: nothing downstream assumes a chunk is full.
    //
    // It carries the chunk's destination in the otherwise-unused FINAL_ADDR word, because from phase 10 a
    // chunk can be EMPTY — a sender's slice of a run may hold no tokens at all. The relay decides
    // final-write-vs-re-forward from the first token it sees, and an empty chunk has none, so without this
    // the decision would be undecidable and the chunk count downstream would drift. Both words travel with a
    // forwarded packet (token + 16 bytes), so it propagates hop to hop for free.
    void close_chunk(uint32_t dst_chip_id, uint32_t chunk_page) {
        volatile tt_l1_ptr cmbf2d::FwdMetadata* metadata = slot_metadata(claim_slot());
        metadata->final_addr = (uint64_t)dst_chip_id;
        metadata->dst_chip = cmbf2d::SENTINEL_DST_CHIP;
        metadata->cmd = cmbf2d::CMD_FORWARD;
        metadata->this_addr = dram.fwd.get_noc_addr(fwd_page(out_chunk, chunk_page));
        published++;
        pending_publish++;
        flush_publish();  // a chunk boundary: do not let it sit unpublished
        out_chunk++;
    }

    // One own assignment: everything this chip owes ONE destination chip, over all the experts it hosts,
    // narrowed to this sender's slice of each run. The experts are merged into a single stream — a chunk is
    // one (this chip -> that chip) term, and which expert a token sat under is not something the forwarding
    // protocol needs to know.
    void do_own_assignment(uint32_t a) {
        const uint32_t dst_chip_id = kernel_compile_time_args[ct.assignment_base + a * cmbf2d::ASSIGNMENT_WORDS + 0];
        const uint32_t dst_row = kernel_compile_time_args[ct.assignment_base + a * cmbf2d::ASSIGNMENT_WORDS + 1];
        const uint32_t split_idx = kernel_compile_time_args[ct.assignment_base + a * cmbf2d::ASSIGNMENT_WORDS + 2];
        const uint32_t split_count = kernel_compile_time_args[ct.assignment_base + a * cmbf2d::ASSIGNMENT_WORDS + 3];
        uint32_t chunk_page = 0;  // position in the downstream chunk, continuous across our experts
        for (uint32_t le = 0; le < ct.experts_per_chip; le++) {
            const uint32_t e = ct.my_expert_base + le;
            const uint32_t begin = ctl.run_begin(dst_row, e);
            const uint32_t n = ctl.run_end(dst_row, e) - begin;
            chunk_page += stage_run_slice(
                dst_chip_id,
                slice_begin(begin, n, split_idx, split_count),
                slice_begin(begin, n, split_idx + 1, split_count),
                chunk_page);
        }

        if (!is_direct(dst_chip_id)) {
            close_chunk(dst_chip_id, chunk_page);
        }
    }

    // One arrived chunk, pushed exactly one more hop.
    //
    // This is where most of a sender's traffic comes from, not its own assignments: on an 8-ring each
    // sender puts H*H/2 = 8 tok packets on its cable but only ~1.75 tok of those are its own, so ~78% is
    // relayed. Batching the reads here is therefore what the reader's throughput actually turns on.
    //
    // Batching has to be SPECULATIVE, because a chunk's length is only discovered by reading its sentinel.
    // Two facts bound the speculation:
    //   * having read a non-sentinel page, the next page of this chunk certainly exists;
    //   * the upstream watermark says how many pages upstream has written, though not how they divide
    //     between this chunk and later ones.
    // So we read up to `batch` pages at once and stop at the sentinel. Pages read past a sentinel are
    // garbage upstream never wrote — harmless, since they are discarded and their slots handed straight back
    // (nothing was announced for them, so the sender never saw them). The waste is at most batch-1 reads
    // per chunk against a mean chunk of tok pages, and it buys `batch` reads in flight.
    void do_forward_chunk(uint32_t chunk) {
        // Every token of a chunk shares one final destination (a chunk IS one (chip -> chip) term), so the
        // first page decides for the whole chunk whether this hop is the last one.
        bool reforward = false;
        bool decided = false;
        uint32_t i = 0;  // page index within the chunk
        bool at_end = false;

        while (!at_end) {
            const uint32_t k = read_arrived_pages(chunk, i);

            uint32_t used = 0;  // pages of this ct.batch that really belong to the chunk
            uint32_t pub = 0;   // of those, the ones we hand to the sender
            const uint32_t first_slot = claimed - k;
            for (uint32_t j = 0; j < k; j++) {
                volatile tt_l1_ptr cmbf2d::FwdMetadata* metadata = slot_metadata((first_slot + j) % ct.num_l1_slots);
                const uint64_t dst_chip = metadata->dst_chip;
                used++;

                if (dst_chip == cmbf2d::SENTINEL_DST_CHIP) {
                    // An EMPTY chunk reaches its sentinel with nothing having decided for it, so the
                    // sentinel's own destination word decides. The chunk still has to be passed on when it is
                    // not for our neighbour: the downstream reader counts chunks positionally, so silently
                    // dropping one would shift every chunk after it.
                    if (!decided) {
                        reforward = (metadata->final_addr != (uint64_t)ct.nbr_chip_id);
                        decided = true;
                    }
                    if (reforward) {
                        // Pass the terminator on, at the end of the chunk we have been writing downstream.
                        // Its destination word is already in place — it came in with the sentinel.
                        metadata->cmd = cmbf2d::CMD_FORWARD;
                        metadata->this_addr = dram.fwd.get_noc_addr(fwd_page(out_chunk, i + j));
                        pub++;
                    }
                    at_end = true;
                    break;
                }

                if (!decided) {
                    reforward = (dst_chip != (uint64_t)ct.nbr_chip_id);
                    decided = true;
                }
                if (reforward) {
                    metadata->cmd = cmbf2d::CMD_FORWARD;
                    metadata->this_addr = dram.fwd.get_noc_addr(fwd_page(out_chunk, i + j));
                } else {
                    metadata->cmd = cmbf2d::CMD_FINAL_WRITE;
                    metadata->this_addr = metadata->final_addr;
                }
                pub++;
            }

            // Announce only the pages we are passing on, in claim order. A sentinel we swallow (last hop) and
            // any speculative reads past it are not announced, so their slots go back by rewinding `claimed`
            // — legitimate precisely because the sender was never told about them.
            if (pub > 0) {
                publish_n(pub);
            }
            claimed -= (k - pub);
            consumed += used;
            i += used;
            if (at_end) {
                flush_publish();  // a chunk boundary: do not let it sit unannounced
            }
        }
        if (reforward) {
            out_chunk++;
        }
    }

    // Wait for upstream to have written past `consumed`, then read as many of chunk `chunk`'s pages from
    // page `i` as are both available and still inside the chunk. Returns how many slots were claimed.
    uint32_t read_arrived_pages(uint32_t chunk, uint32_t i) {
        volatile tt_l1_ptr uint32_t* fwd_arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.fwd_sem_addr);
        // Do not outpace the upstream sender. It bumps this counter every fwd_bump_every pages and ALWAYS
        // right after a sentinel, so a chunk boundary is never left unreachable.
        invalidate_l1_cache();
        if (*fwd_arrived <= consumed) {
            flush_publish();  // let the sender work while we wait on upstream
            while (true) {
                invalidate_l1_cache();
                if (*fwd_arrived > consumed) {
                    break;
                }
            }
        }
        uint32_t k = *fwd_arrived - consumed;
        if (k > ct.batch) {
            k = ct.batch;
        }
        if (k > ct.fwd_pages_per_chunk - i) {
            k = ct.fwd_pages_per_chunk - i;  // never read outside this chunk's own page range
        }

        for (uint32_t j = 0; j < k; j++) {
            const uint32_t slot = claim_slot();
            // ONE read brings the token AND the [final_addr][dst_chip] pair straight into the slot's
            // metadata, because the page layout was chosen to match the slot layout exactly.
            noc_async_read(dram.fwd.get_noc_addr(fwd_page(chunk, i + j)), slot_addr_of(slot), fwd_read_bytes);
        }
        noc_async_read_barrier();
        return k;
    }

    // Walk the schedule. Forwarding chunks always appear in increasing order, so `consumed` stays a valid
    // watermark into our quarter however the own assignments are interleaved around them.
    void run_schedule() {
        for (uint32_t si = 0; si < ct.schedule_len; si++) {
            const uint32_t entry = kernel_compile_time_args[ct.schedule_base + si];
            if (entry & cmbf2d::SCHED_FWD) {
                do_forward_chunk(entry & ~cmbf2d::SCHED_FWD);
            } else {
                do_own_assignment(entry);
            }
        }
        flush_publish();
    }

    // Same-chip tokens. An expert here also holds tokens that started here, and those need no cable at all —
    // they are a DRAM-to-DRAM copy on this chip. So they are not a sender's work and never appear in its
    // packet count; they go straight out over our own NoC.
    //
    // Done LAST and split `local_split_count` ways (one slice per core) for the same reason: it is pure local
    // bandwidth, so doing it while the fabric is still busy would compete with the token reads feeding it,
    // and leaving it to one core would make it a serial tail. Whether last is actually the best place for it,
    // and whether it belongs on separate cores entirely, is a stage-3 question.
    void run_local_phase() {
        // One slot serves as the staging buffer for the whole phase, and is deliberately NEVER published:
        // the sender must not see these tokens, or it would send them over the fabric as well. Slot
        // `published % num_l1_slots` is the next one the sender has yet to be told about, so it is ours to
        // scribble in; claim_slot() is still what proves at least one slot is free before we take it, and
        // `published` does not move during this phase so it stays free. The CMD_END slot below reclaims it.
        const uint32_t slot_addr = slot_addr_of(claim_slot());
        for (uint32_t le = 0; le < ct.experts_per_chip; le++) {
            const uint32_t e = ct.my_expert_base + le;
            const uint32_t begin = ctl.run_begin(ct.my_row, e);
            const uint32_t n = ctl.run_end(ct.my_row, e) - begin;
            const uint32_t lo = slice_begin(begin, n, ct.my_quarter, ct.local_split_count);
            const uint32_t hi = slice_begin(begin, n, ct.my_quarter + 1, ct.local_split_count);
            for (uint32_t base = lo; base < hi; base += ct.meta_prefetch_cap) {
                const uint32_t end = (hi - base) > ct.meta_prefetch_cap ? base + ct.meta_prefetch_cap : hi;
                prefetch_metadata(base, end);
                for (uint32_t p = base; p < end; p++) {
                    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        meta_pads_addr + (p - base) * cmbf2d::META_PAD_STRIDE);
                    const uint32_t out_page = meta[1] * ct.num_experts_per_tok + meta[2];
                    const uint64_t out_addr = dram.out.get_noc_addr(out_page);
                    noc_async_read(dram.in.get_noc_addr(p), slot_addr, ct.token_size_bytes);
                    noc_async_read_barrier();
                    noc_async_write(slot_addr, out_addr, ct.token_size_bytes);
                    // Serialised on purpose: the slot IS the buffer, so it cannot be refilled until the write
                    // has read it out. Overlapping these needs several buffers in flight, tracked separately
                    // from the sender's ring accounting.
                    noc_async_write_barrier();
                }
            }
        }
        claimed--;  // hand the staging slot back; nothing was ever announced for it
    }

    // End of stream. The sender cannot know the length up front, so it stops on this.
    void end_stream() {
        slot_metadata(claim_slot())->cmd = cmbf2d::CMD_END;
        published++;
        pending_publish++;
        flush_publish();
    }
};

void kernel_main() {
    const Dram dram = open_dram();
    Reader reader{dram, read_control_tables(dram)};

    reader.run_schedule();
    reader.run_local_phase();
    reader.end_stream();

    // Back to zero for the next launch, which starts its own count at zero. The upstream sender cannot bump
    // this again: we only got past the last chunk by reading its sentinel, so everything it owes us has
    // arrived, and its drain targets a sink address rather than this semaphore.
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.fwd_sem_addr), 0);
}
