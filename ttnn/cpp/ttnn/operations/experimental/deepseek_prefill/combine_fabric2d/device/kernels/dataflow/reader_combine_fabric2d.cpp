// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel (reader RISC, NOC_0). Feeds the L1 ring the sender sends from and makes ALL routing
// decisions. The sender writes L1 -> eth over NOC_1, so the two do not contend.
//
// Every packet the sender sends travels exactly one hop, to the chip across its own cable, so a token
// bound further is staged into that neighbour's forwarding buffer and re-sent from there by the neighbour's
// reader on the same (plane, direction). Hence two kinds of work:
//
//   A. Its OWN assignments — its plane's share of this chip's movements. Each destination is either our
//      immediate neighbour (=> CMD_FINAL_WRITE, the neighbour writes it straight into its output region) or
//      further away (=> CMD_FORWARD into the neighbour's forwarding buffer).
//
//   B. The chunks that arrived in OUR region of the forwarding buffer, written by the upstream chip's
//      sender on the same (plane, direction). Each is pushed one more hop: final-write if its destination
//      is now our neighbour, re-forward otherwise.
//
// Both are done one LOCAL EXPERT at a time: the whole schedule runs for expert 0, then again for expert 1,
// and so on, so a chunk is one (origin chip -> destination chip, expert) term. Every chip walks its experts
// in the same order, so a relay at iteration e is passing on chunks its upstream produced at iteration e —
// the ordering the dense forwarding buffer needs is unchanged, there are just more, smaller chunks. This is
// what lets an upstream stage produce one expert's tokens at a time.
//
// The forwarding buffer is DENSE: a chunk occupies exactly as many pages as it has tokens, and starts where
// the previous one ended. Nothing is exchanged to make the writer and the reader of a region agree on those
// boundaries — both compute every chunk's length from the same replicated expert_offsets, using the chunk
// descriptors the host packed in the same order the upstream sender emits them.
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
#include "combine_fabric2d_group_walk.hpp"

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

// Page of our region of the forwarding buffer. The same formula serves both directions: the stream index
// is ours either way, only the chip differs, and the buffer's device-local address is uniform across the
// mesh. Chunk boundaries do not appear here — a chunk is just a run of consecutive pages, so both sides
// address the region by a single running page count.
constexpr uint32_t fwd_page(uint32_t page_in_stream) { return ct.my_stream * ct.fwd_pages_per_stream + page_in_stream; }

// Our slice of a run: of [begin, end) take [begin + n*idx/count, begin + n*(idx+1)/count). Integer
// arithmetic, so consecutive slices meet exactly — no gaps, no overlap, whatever n is.
constexpr uint32_t slice_begin(uint32_t begin, uint32_t n, uint32_t idx, uint32_t count) {
    return begin + (uint32_t)(((uint64_t)n * idx) / count);
}

uint32_t slot_addr_of(uint32_t slot) { return ct.ring_addr + slot * ct.slot_stride(); }

// Step `i` of a prefetch window of `n` pages, as an offset from the window's base. Which IS the pad its
// metadata was prefetched into, whichever way the window is walked.
constexpr uint32_t pad_at(uint32_t n, uint32_t i) { return ct.walks_down ? n - 1 - i : i; }

// A destination one hop away is written straight into its output region; anything further is staged into
// the neighbour's forwarding buffer and re-sent from there.
bool is_direct(uint32_t dst_chip_id) { return dst_chip_id == ct.nbr_chip_id; }

volatile tt_l1_ptr cmbf2d::FwdMetadata* slot_metadata(uint32_t slot) {
    return reinterpret_cast<volatile tt_l1_ptr cmbf2d::FwdMetadata*>(slot_addr_of(slot) + ct.token_size_bytes);
}

// The expert is not part of the descriptor: the same one serves every iteration of the outer expert loop,
// which is what names the run.
cmbf2d::ChunkDescriptor forwarding_chunk(uint32_t chunk) {
    return cmbf2d::ChunkDescriptor::from_words(
        &kernel_compile_time_args[ct.forwarding_chunk_base + chunk * cmbf2d::CHUNK_WORDS]);
}

// Experts hosted by any chip on our ring. The dispatch group's experts are laid out in dispatch-group-index order, so
// our own base locates every other chip's — which is what lets us size a chunk we neither produced nor receive.
constexpr uint32_t group_expert_base = ct.my_expert_base - ct.my_dg_index * ct.experts_per_chip;
constexpr uint32_t expert_base(uint32_t dg_index) { return group_expert_base + dg_index * ct.experts_per_chip; }

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

// Control tensors, read once. All three are one row per page of `num_routed_experts` uint32: expert_offsets
// has one row per ORIGIN chip (it is replicated along that axis so every chip sees all of them), counts and
// region_offsets are a single row each. A few kB in total, so the whole slice comes in rather than
// cherry-picking this chip's columns — a strided gather of 4-byte words would cost more NoC transactions
// than the bytes saved.
cmbf2d::ControlTables read_control_tables(const Dram& dram) {
    constexpr uint32_t row_bytes = ct.num_routed_experts * 4;
    cmbf2d::ControlTables ctl{
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(control_tables_addr),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            control_tables_addr + ct.dispatch_group_size * ct.num_routed_experts * 4),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            control_tables_addr + (ct.dispatch_group_size + 1) * ct.num_routed_experts * 4),
        ct.num_routed_experts,
        ct.dispatch_group_size};
    for (uint32_t r = 0; r < ct.dispatch_group_size; r++) {
        noc_async_read(dram.expert_offsets.get_noc_addr(r), control_tables_addr + r * row_bytes, row_bytes);
    }
    noc_async_read(dram.counts.get_noc_addr(0), (uint32_t)ctl.counts, row_bytes);
    noc_async_read(dram.region.get_noc_addr(0), (uint32_t)ctl.region, row_bytes);
    noc_async_read_barrier();
    return ctl;
}

#if TILE
// Our end of the untilizer handshake. The group produces a fixed sequence of batches (see the group walk),
// this stream reads the rows it wants out of them, and the two exchange only counts.
//
// A stream takes half of each run, so it SKIPS the batches its partner is taking. Those have to be released
// on the way past or the group's producer stalls on a slot nobody wants, which is why releasing is a
// watermark rather than something only a batch we read gets.
struct Untilized {
    uint32_t expert_base = 0;  // batch index the current expert's walk starts at
    uint32_t released = 0;
    uint32_t open = ~0u;  // batch we are reading out of, or none

    static uint32_t owner(uint32_t batch) { return batch % ct.num_untilizers; }

    static uint64_t untilizer_noc(uint32_t j, uint32_t addr) {
        const uint32_t w = ct.untilizer_base + j * cmbf2d::UNT_PEER_WORDS;
        return get_noc_addr(kernel_compile_time_args[w + 0], kernel_compile_time_args[w + 1], addr);
    }

    // Where untilizer j's produced count sits on OUR core.
    static volatile tt_l1_ptr uint32_t* produced_by(uint32_t j) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            kernel_compile_time_args[ct.untilizer_base + j * cmbf2d::UNT_PEER_WORDS + 2]);
    }

    void release_through(uint32_t upto) {
        if (open != ~0u && upto > open) {
            noc_async_read_barrier();  // rows of the batch we are leaving may still be in flight
            open = ~0u;
        }
        for (; released < upto; released++) {
            noc_semaphore_inc(untilizer_noc(owner(released), ct.unt_freed_addr), 1);
        }
    }

    // Where `page`'s row sits, blocking until the batch holding it has been staged. Hands back everything
    // before it FIRST: the producer reuses those slots to build this one, so holding them back would be
    // waiting on a core that is waiting on us.
    uint64_t take(const cmbf2d::GroupWalk& walk, uint32_t page) {
        const uint32_t batch = expert_base + walk.batch_of(page);
        if (batch != open) {
            release_through(batch);
            volatile tt_l1_ptr uint32_t* produced = produced_by(owner(batch));
            invalidate_l1_cache();
            while (*produced < batch / ct.num_untilizers + 1) {
                invalidate_l1_cache();
            }
            open = batch;
        }
        const uint32_t row =
            (batch / ct.num_untilizers % ct.unt_ring_batches) * cmbf2d::UNT_BATCH_ROWS + page % cmbf2d::UNT_BATCH_ROWS;
        return untilizer_noc(owner(batch), ct.unt_ring_addr + row * ct.token_size_bytes);
    }

    void finish_expert(const cmbf2d::GroupWalk& walk) {
        expert_base += walk.num_batches();
        release_through(expert_base);
    }

    // End of stream. Every batch has been released, so each untilizer has stopped bumping and its count can
    // go back to zero for the next launch -- but only once its last bump has actually landed, which is what
    // the wait is for.
    void reset_counters() const {
        for (uint32_t j = 0; j < ct.num_untilizers; j++) {
            const uint32_t mine = expert_base > j ? (expert_base - j - 1) / ct.num_untilizers + 1 : 0;
            volatile tt_l1_ptr uint32_t* produced = produced_by(j);
            invalidate_l1_cache();
            while (*produced < mine) {
                invalidate_l1_cache();
            }
            noc_semaphore_set(produced, 0);
        }
    }
};
#endif

// Everything the phases share: the ring's flow control, where the next downstream chunk goes, and how far
// into our own region we have consumed.
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
    cmbf2d::ControlTables ctl;
#if TILE
    Untilized untilized;
    // The group's production for the expert in progress. Rebuilt every iteration, identically on every core
    // of the group, which is what makes a batch index mean the same thing to all of them.
    cmbf2d::GroupWalk walk{ct.walks_down != 0};
#endif

    // Where page `p` of our region is read from: an untilizer's staging ring, or the buffer itself when the
    // tokens are already rows.
    uint64_t token_source(uint32_t p) {
#if TILE
        return untilized.take(walk, p);
#else
        return dram.in.get_noc_addr(p);
#endif
    }
    uint32_t published = 0;
    uint32_t claimed = 0;
    // Publishing is batched so the atomic is amortised, matching the sender's batch.
    uint32_t pending_publish = 0;
    // How far into the downstream chip's region we have written, and where the chunk in progress ends. That
    // reader walks its chunks in exactly the order we emit them, sizing each the same way we do, so the page
    // counter alone is what the two chips agree on; the end marks where the sender owes it a bump.
    uint32_t out_page = 0;
    uint32_t out_chunk_end = 0;
    // Pages taken out of our own region, which for the same reason IS the page the next chunk starts at.
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

    // The token count of one forwarding chunk: the destination chip's tokens sitting under one of the origin
    // chip's experts, narrowed to the share the two chips agreed on. Every term is readable on any chip of
    // the ring, which is what makes a chunk's length agree between the chip that writes it and the chip that
    // reads it without either telling the other.
    uint32_t chunk_tokens(const cmbf2d::ChunkDescriptor& desc, uint32_t local_expert) const {
        const uint32_t e = expert_base(desc.origin_dg_index) + local_expert;
        const uint32_t begin = ctl.run_begin(desc.dst_dg_index, e);
        const uint32_t n = ctl.run_end(desc.dst_dg_index, e) - begin;
        return slice_begin(begin, n, desc.split_idx + 1, desc.split_count) -
               slice_begin(begin, n, desc.split_idx, desc.split_count);
    }

    // Point one slot at `page` of the downstream chip's region. The chunk's last page is marked so the
    // sender bumps that chip's arrival counter right there: that reader consumes a whole chunk before moving
    // on, so a tail left inside a partial bump batch would strand it.
    void aim_at_downstream(volatile tt_l1_ptr cmbf2d::FwdMetadata* metadata, uint32_t page) const {
        metadata->cmd = (page + 1 == out_chunk_end) ? cmbf2d::CMD_FORWARD_END : cmbf2d::CMD_FORWARD;
        metadata->this_addr = dram.fwd.get_noc_addr(fwd_page(page));
    }

    // Issues one token's read and fills its metadata but does NOT barrier or announce it: the caller does
    // both once per batch, so several reads are in flight together.
    void issue_token(uint32_t dst_chip_id, uint32_t in_page, uint32_t pad) {
        const uint32_t slot = claim_slot();
        volatile tt_l1_ptr cmbf2d::FwdMetadata* metadata = slot_metadata(slot);
        volatile tt_l1_ptr uint32_t* meta =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_pads_addr + pad * cmbf2d::META_PAD_STRIDE);

        noc_async_read(token_source(in_page), slot_addr_of(slot), ct.token_size_bytes);
        // Everything below runs while that read is in flight, which is the point of prefetching the metadata.
        // Record layout: [0] linearized_coord, [1] token_idx, [2] topk_idx; the destination chip is already
        // known from the run this token came out of.
        const uint32_t token_out_page = meta[1] * ct.num_experts_per_tok + meta[2];
        // Computed once and then travelling with the token for however many hops it takes. Valid on any chip
        // because the output buffer's base address and interleaved bank mapping are uniform across the mesh.
        const uint64_t final_addr = dram.out.get_noc_addr(token_out_page);
        if (is_direct(dst_chip_id)) {
            metadata->cmd = cmbf2d::CMD_FINAL_WRITE;
            metadata->this_addr = final_addr;
        } else {
            metadata->final_addr = final_addr;
            metadata->dst_chip = (uint64_t)dst_chip_id;
            aim_at_downstream(metadata, out_page++);
        }
    }

    // Pages [lo, hi) in the order this stream walks them, a prefetch window at a time: `body(base, n)` gets
    // a window whose metadata is already in the pads, and steps it with pad_at.
    template <typename Body>
    void walk_pages(uint32_t lo, uint32_t hi, Body body) {
        for (uint32_t done = 0; done < hi - lo;) {
            const uint32_t n = (hi - lo - done) > ct.meta_prefetch_cap ? ct.meta_prefetch_cap : hi - lo - done;
            const uint32_t base = ct.walks_down ? hi - done - n : lo + done;
            prefetch_metadata(base, base + n);
            body(base, n);
            done += n;
        }
    }

    // Stage [lo, hi) of one run: issue and announce its pages `batch` at a time.
    void stage_run_slice(uint32_t dst_chip_id, uint32_t lo, uint32_t hi) {
        walk_pages(lo, hi, [&](uint32_t base, uint32_t n) {
            // `batch` tokens' reads in flight at a time, then one barrier and one announcement.
            for (uint32_t q = 0; q < n;) {
                const uint32_t k = (n - q) > ct.batch ? ct.batch : (n - q);
                for (uint32_t j = 0; j < k; j++) {
                    issue_token(dst_chip_id, base + pad_at(n, q + j), pad_at(n, q + j));
                }
                noc_async_read_barrier();  // every token of the ct.batch is in L1 before any is announced
                publish_n(k);
                q += k;
            }
        });
    }

    // One own assignment for one local expert: what this chip owes ONE destination chip out of that
    // expert's region, narrowed to this sender's slice of the run.
    void do_own_assignment(uint32_t a, uint32_t local_expert) {
        const uint32_t w = ct.assignment_base + a * cmbf2d::ASSIGNMENT_WORDS;
        const uint32_t dst_chip_id = kernel_compile_time_args[w + 0];
        const cmbf2d::ChunkDescriptor desc{
            ct.my_dg_index,
            kernel_compile_time_args[w + 1],
            kernel_compile_time_args[w + 2],
            kernel_compile_time_args[w + 3]};
        // Sized before the first page is placed, because the last page has to be recognisable when it comes.
        out_chunk_end = out_page + (is_direct(dst_chip_id) ? 0 : chunk_tokens(desc, local_expert));
        const uint32_t e = ct.my_expert_base + local_expert;
        const uint32_t begin = ctl.run_begin(desc.dst_dg_index, e);
        const uint32_t n = ctl.run_end(desc.dst_dg_index, e) - begin;
        stage_run_slice(
            dst_chip_id,
            slice_begin(begin, n, desc.split_idx, desc.split_count),
            slice_begin(begin, n, desc.split_idx + 1, desc.split_count));
    }

    // One arrived chunk, pushed exactly one more hop.
    //
    // This is where most of a sender's traffic comes from, not its own assignments: on an 8-ring each
    // sender puts H*H/2 = 8 tok packets on its cable but only ~1.75 tok of those are its own, so ~78% is
    // relayed. Batching the reads here is therefore what the reader's throughput actually turns on.
    //
    // The chunk's length is computed rather than discovered, so the reads are batched with nothing
    // speculative about them: `batch` pages at a time, capped by what is left of the chunk and by what
    // upstream says has arrived. An EMPTY chunk occupies no pages at all and so is simply not there.
    void do_forward_chunk(uint32_t chunk, uint32_t local_expert) {
        uint32_t remaining = chunk_tokens(forwarding_chunk(chunk), local_expert);
        if (remaining == 0) {
            return;
        }
        out_chunk_end = out_page + remaining;  // where the marker goes, should this chunk be re-forwarded

        // Every token of a chunk shares one final destination (a chunk IS one (chip -> chip, expert) term),
        // so the first page decides for the whole chunk whether this hop is the last one.
        bool reforward = false;
        bool decided = false;
        while (remaining > 0) {
            const uint32_t k = read_arrived_pages(remaining);
            const uint32_t first_slot = claimed - k;
            for (uint32_t j = 0; j < k; j++) {
                volatile tt_l1_ptr cmbf2d::FwdMetadata* metadata = slot_metadata((first_slot + j) % ct.num_l1_slots);
                if (!decided) {
                    reforward = (metadata->dst_chip != (uint64_t)ct.nbr_chip_id);
                    decided = true;
                }
                if (reforward) {
                    aim_at_downstream(metadata, out_page++);
                } else {
                    metadata->cmd = cmbf2d::CMD_FINAL_WRITE;
                    metadata->this_addr = metadata->final_addr;
                }
            }
            publish_n(k);
            consumed += k;
            remaining -= k;
        }
    }

    // Wait for upstream to have written past `consumed`, then read as many of the chunk in progress as are
    // both available and still inside it — `remaining` is what is left of it. Returns how many slots were
    // claimed, all of which belong to the chunk.
    uint32_t read_arrived_pages(uint32_t remaining) {
        volatile tt_l1_ptr uint32_t* fwd_arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.fwd_sem_addr);
        // Do not outpace the upstream sender. It bumps this counter every fwd_bump_every pages and ALWAYS on
        // the last page of a chunk, so a chunk boundary is never left unreachable.
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
        if (k > remaining) {
            k = remaining;
        }

        for (uint32_t j = 0; j < k; j++) {
            const uint32_t slot = claim_slot();
            // ONE read brings the token AND the [final_addr][dst_chip] pair straight into the slot's
            // metadata, because the page layout was chosen to match the slot layout exactly.
            noc_async_read(dram.fwd.get_noc_addr(fwd_page(consumed + j)), slot_addr_of(slot), fwd_read_bytes);
        }
        noc_async_read_barrier();
        return k;
    }

    // Walk the schedule for one local expert. The schedule is the same for every expert — which chip pairs
    // this stream serves and in what order does not depend on the expert — so the expert is only the
    // caller's loop variable. Forwarding chunks always appear in increasing order, so `consumed` stays a
    // valid watermark into our region however the own assignments are interleaved around them.
    //
    // Publishing at the end leaves nothing claimed but unannounced, which is what lets the local phase
    // treat the next slot as private.
    void run_schedule(uint32_t local_expert) {
        for (uint32_t si = 0; si < ct.schedule_len; si++) {
            const uint32_t entry = kernel_compile_time_args[ct.schedule_base + si];
            if (entry & cmbf2d::SCHED_FWD) {
                do_forward_chunk(entry & ~cmbf2d::SCHED_FWD, local_expert);
            } else {
                do_own_assignment(entry, local_expert);
            }
        }
        flush_publish();
    }

    // Same-chip tokens for one local expert. An expert here also holds tokens that started here, and those
    // need no cable at all — they are a DRAM-to-DRAM copy on this chip. So they are not a sender's work and
    // never appear in its packet count; they go straight out over our own NoC.
    //
    // Split `local_split_count` ways, one slice per core, so it is not a serial tail on one of them. It runs
    // inside the expert loop rather than once at the end, so an expert is wholly done — fabric and local
    // alike — before the next one starts; that costs some overlap with the fabric work still draining, and
    // whether these copies belong on separate cores entirely is a stage-3 question.
    void run_local_phase(uint32_t local_expert) {
        // One slot serves as the staging buffer for the whole phase, and is deliberately NEVER published:
        // the sender must not see these tokens, or it would send them over the fabric as well. Slot
        // `published % num_l1_slots` is the next one the sender has yet to be told about, so it is ours to
        // scribble in; claim_slot() is still what proves at least one slot is free before we take it, and
        // `published` does not move during this phase so it stays free. The CMD_END slot below reclaims it.
        const uint32_t slot_addr = slot_addr_of(claim_slot());
        const uint32_t e = ct.my_expert_base + local_expert;
        const uint32_t begin = ctl.run_begin(ct.my_dg_index, e);
        const uint32_t n = ctl.run_end(ct.my_dg_index, e) - begin;
        const uint32_t lo = slice_begin(begin, n, ct.my_stream, ct.local_split_count);
        const uint32_t hi = slice_begin(begin, n, ct.my_stream + 1, ct.local_split_count);
        walk_pages(lo, hi, [&](uint32_t base, uint32_t window) {
            for (uint32_t i = 0; i < window; i++) {
                const uint32_t pad = pad_at(window, i);
                volatile tt_l1_ptr uint32_t* meta =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_pads_addr + pad * cmbf2d::META_PAD_STRIDE);
                const uint32_t out_page = meta[1] * ct.num_experts_per_tok + meta[2];
                const uint64_t out_addr = dram.out.get_noc_addr(out_page);
                noc_async_read(token_source(base + pad), slot_addr, ct.token_size_bytes);
                noc_async_read_barrier();
                noc_async_write(slot_addr, out_addr, ct.token_size_bytes);
                // Serialised on purpose: the slot IS the buffer, so it cannot be refilled until the write
                // has read it out. Overlapping these needs several buffers in flight, tracked separately
                // from the sender's ring accounting.
                noc_async_write_barrier();
            }
        });
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

    // One pass per local expert, fabric then local, so every token of expert e is placed before expert
    // e + 1 is touched.
    for (uint32_t local_expert = 0; local_expert < ct.experts_per_chip; local_expert++) {
#if TILE
        reader.walk = cmbf2d::group_walk(
            reader.ctl,
            ct.walks_down != 0,
            ct.my_expert_base + local_expert,
            ct.my_dg_index,
            ct.num_assignments,
            [](uint32_t k) { return kernel_compile_time_args[ct.assignment_base + k * cmbf2d::ASSIGNMENT_WORDS + 1]; });
#endif
        reader.run_schedule(local_expert);
        reader.run_local_phase(local_expert);
#if TILE
        reader.untilized.finish_expert(reader.walk);
#endif
    }
    reader.end_stream();
#if TILE
    reader.untilized.reset_counters();
#endif

    // Back to zero for the next launch, which starts its own count at zero. The upstream sender cannot bump
    // this again: its bumps sum to exactly the pages of our region and we consumed all of them, so the last
    // one has already landed — and its drain targets a sink address rather than this semaphore.
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.fwd_sem_addr), 0);
}
