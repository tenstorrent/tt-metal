// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel (reader RISC, NOC_0). Builds this chip's routing index, then fills the L1 ring the sender
// on this same core drains.
//
// The routing index is the piece with no counterpart in combine. Combine's input is already grouped by
// origin chip, so a chunk is four words out of a control table. Dispatch's input is token order and a
// token's destination is data-dependent (indices -> dispatch table), so the destination-grouped runs the
// protocol needs have to be manufactured here.
//
// Replaying the production op's allocator EXACTLY is what makes the pages byte-identical to it, including
// the rule that a token past the buffer's capacity is dropped while its counter still advances. Every
// stream core replays the whole walk independently and identically, which is what lets the baton
// semaphore the production op passes between its workers disappear.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/debug/assert.h"
#include "dispatch_fabric2d_reader_ct_args.hpp"

constexpr dspf2d::ReaderCtArgs ct{};

namespace {

// The reader's L1 working set, carved out of the control region in one fixed order so every chip lays it
// out identically. `indices` comes first because it is the only part read straight from DRAM per token,
// and its records must stay 64-byte aligned.
struct Control {
    volatile tt_l1_ptr uint16_t* indices;       // seq_len records, each padded to indices_pad_stride
    volatile tt_l1_ptr uint32_t* offsets;       // extent x num_routed_experts: every source chip's row
    volatile tt_l1_ptr uint32_t* counts;        // num_routed_experts, summed over source chips
    volatile tt_l1_ptr uint32_t* region;        // num_routed_experts
    volatile tt_l1_ptr int32_t* table;          // num_routed_experts (+1 sentinel), expert -> chip in group
    volatile tt_l1_ptr uint32_t* alloc;         // num_routed_experts, the running per-expert allocator
    volatile tt_l1_ptr uint32_t* chip_experts;  // extent x experts_per_chip, ascending global expert id
    volatile tt_l1_ptr uint32_t* bucket_len;    // extent x experts_per_chip
    volatile tt_l1_ptr uint32_t* bucket_start;  // extent x experts_per_chip
    volatile tt_l1_ptr uint32_t* entries;       // 3 words per surviving (token, top-k slot)
    uint32_t end;
};

Control carve_control() {
    uint32_t a = ct.control_addr;
    const auto take = [&](uint32_t bytes) {
        const uint32_t at = a;
        a += bytes;
        return at;
    };
    const auto words = [&](uint32_t n) { return take(n * 4u); };

    Control c;
    c.indices = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(take(ct.seq_len * ct.indices_pad_stride));
    c.offsets = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words(ct.extent * ct.num_routed_experts));
    c.counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words(ct.num_routed_experts));
    c.region = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words(ct.num_routed_experts));
    c.table = reinterpret_cast<volatile tt_l1_ptr int32_t*>(words(ct.num_routed_experts + 1));
    c.alloc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words(ct.num_routed_experts));
    c.chip_experts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words(ct.extent * ct.experts_per_chip));
    c.bucket_len = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words(ct.extent * ct.experts_per_chip));
    c.bucket_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words(ct.extent * ct.experts_per_chip));
    c.entries = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words(3 * ct.seq_len * ct.topk));
    c.end = a;
    return c;
}

// Every source chip's row of the offsets table, plus the two tensors that close the last row, plus the
// dispatch table. A few kB, read once and indexed from L1 thereafter.
void read_control_tables(const Control& c) {
    const auto offsets_acc = TensorAccessor(dspf2d::ReaderCtArgs::offsets_args, get_arg_val<uint32_t>(2));
    const auto table_acc = TensorAccessor(dspf2d::ReaderCtArgs::table_args, get_arg_val<uint32_t>(3));
    const auto counts_acc = TensorAccessor(dspf2d::ReaderCtArgs::counts_args, get_arg_val<uint32_t>(4));
    const auto region_acc = TensorAccessor(dspf2d::ReaderCtArgs::region_args, get_arg_val<uint32_t>(5));

    const uint32_t row_bytes = ct.num_routed_experts * 4u;
    for (uint32_t r = 0; r < ct.extent; r++) {
        noc_async_read(offsets_acc.get_noc_addr(r), (uint32_t)(c.offsets + r * ct.num_routed_experts), row_bytes);
    }
    noc_async_read(counts_acc.get_noc_addr(0), (uint32_t)c.counts, row_bytes);
    noc_async_read(region_acc.get_noc_addr(0), (uint32_t)c.region, row_bytes);
    // The table carries a trailing sentinel column so a padded token's unguarded lookup maps to -1.
    noc_async_read(table_acc.get_noc_addr(0), (uint32_t)c.table, (ct.num_routed_experts + 1) * 4u);
    noc_async_read_barrier();
}

// One 64-byte-padded record per token. The pad is not a convenience: a DRAM read needs a 64-byte-aligned
// L1 destination on Blackhole, so reading topk uint16 per token into a packed array would put every token
// after the first at a wrong address and build the whole index out of garbage.
void read_indices(const Control& c) {
    const auto indices_acc = TensorAccessor(dspf2d::ReaderCtArgs::indices_args, get_arg_val<uint32_t>(1));
    const uint32_t record_bytes = ct.topk * 2u;
    for (uint32_t t = 0; t < ct.seq_len; t++) {
        noc_async_read(indices_acc.get_noc_addr(t), (uint32_t)c.indices + t * ct.indices_pad_stride, record_bytes);
    }
    noc_async_read_barrier();
}

// chip -> its experts, ascending. Every chip on the axis builds the same inverse because the dispatch
// table is replicated along it, which is what lets a relay expand a (origin, destination) descriptor into
// the same experts_per_chip chunks the writer expanded it into.
void build_chip_experts(const Control& c) {
    for (uint32_t i = 0; i < ct.extent * ct.experts_per_chip; i++) {
        c.chip_experts[i] = 0;
        c.bucket_len[i] = 0;
    }
    for (uint32_t row = 0; row < ct.extent; row++) {
        uint32_t n = 0;
        for (uint32_t e = 0; e < ct.num_routed_experts; e++) {
            if (c.table[e] != (int32_t)row) {
                continue;
            }
            ASSERT(n < ct.experts_per_chip);
            c.chip_experts[row * ct.experts_per_chip + n] = e;
            n++;
        }
        // The whole protocol sizes a relayed chunk group as experts_per_chip terms, so a chip hosting a
        // different number would desynchronise the writer and the reader of a forwarding region.
        ASSERT(n == ct.experts_per_chip);
    }
}

// This chip's tokens per (destination chip, expert), replaying the production allocator so the page a
// token lands on is the page that op would have given it.
//
// Returns the number of tokens dropped for want of capacity. The production op advances the counter for a
// dropped token and emits nothing, and so does this: the pages of every later token depend on it.
uint32_t count_buckets(const Control& c) {
    for (uint32_t e = 0; e < ct.num_routed_experts; e++) {
        c.alloc[e] = c.offsets[ct.my_row * ct.num_routed_experts + e];
    }
    uint32_t dropped = 0;
    for (uint32_t t = 0; t < ct.seq_len; t++) {
        volatile tt_l1_ptr uint16_t* idx =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>((uint32_t)c.indices + t * ct.indices_pad_stride);
        for (uint32_t k = 0; k < ct.topk; k++) {
            const uint32_t e = idx[k];
            const int32_t row = c.table[e];
            if (row == -1) {
                continue;  // the expert is not in this dispatch group
            }
            if (c.alloc[e] >= ct.max_dispatch_buf_tokens) {
                c.alloc[e]++;
                dropped++;
                continue;
            }
            c.alloc[e]++;
            // Which of that chip's experts this is. Linear over experts_per_chip, which is 8 at
            // production geometry.
            const uint32_t base = (uint32_t)row * ct.experts_per_chip;
            for (uint32_t j = 0; j < ct.experts_per_chip; j++) {
                if (c.chip_experts[base + j] == e) {
                    c.bucket_len[base + j]++;
                    break;
                }
            }
        }
    }
    return dropped;
}

// The free consistency check, and the reason to build the index before moving a byte: this chip owes
// expert e exactly as many tokens as its own row of the offsets table says, because that table was
// derived from the same routing. Rows are absolute buffer positions, so the last row closes against
// counts + region rather than counts alone.
//
// A mismatch means the replay diverged from the production allocator -- which would otherwise surface as
// wrong pages or, worse, as a deadlock once chunk lengths are computed from these same numbers.
void check_buckets(const Control& c, uint32_t dropped) {
    if (dropped != 0) {
        return;  // capacity was exceeded, so the buckets legitimately fall short of the table
    }
    for (uint32_t row = 0; row < ct.extent; row++) {
        for (uint32_t j = 0; j < ct.experts_per_chip; j++) {
            const uint32_t e = c.chip_experts[row * ct.experts_per_chip + j];
            const uint32_t mine = c.offsets[ct.my_row * ct.num_routed_experts + e];
            const uint32_t expected = (ct.my_row + 1 < ct.extent)
                                          ? c.offsets[(ct.my_row + 1) * ct.num_routed_experts + e] - mine
                                          : c.counts[e] + c.region[e] - mine;
            ASSERT(c.bucket_len[row * ct.experts_per_chip + j] == expected);
        }
    }
}

// Words per entry in a bucket: the token's index on this chip, the page it lands on at the
// destination, and which top-k slot it came from. The last two both travel to the destination -- the
// page as an address, the slot as metadata field 2.
constexpr uint32_t ENTRY_WORDS = 3;

// Fill each bucket with its (token, page, slot) triples, in token order. Replays the same allocator
// count_buckets did, which is what makes the page a token lands on match the production op's.
void fill_entries(const Control& c) {
    for (uint32_t e = 0; e < ct.num_routed_experts; e++) {
        c.alloc[e] = c.offsets[ct.my_row * ct.num_routed_experts + e];
    }
    for (uint32_t i = 0; i < ct.extent * ct.experts_per_chip; i++) {
        c.bucket_len[i] = 0;  // reused as the running cursor into each bucket
    }
    for (uint32_t t = 0; t < ct.seq_len; t++) {
        volatile tt_l1_ptr uint16_t* idx =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>((uint32_t)c.indices + t * ct.indices_pad_stride);
        for (uint32_t k = 0; k < ct.topk; k++) {
            const uint32_t e = idx[k];
            const int32_t row = c.table[e];
            if (row == -1) {
                continue;
            }
            if (c.alloc[e] >= ct.max_dispatch_buf_tokens) {
                c.alloc[e]++;
                continue;
            }
            const uint32_t page = c.alloc[e]++;
            const uint32_t base = (uint32_t)row * ct.experts_per_chip;
            for (uint32_t j = 0; j < ct.experts_per_chip; j++) {
                if (c.chip_experts[base + j] != e) {
                    continue;
                }
                const uint32_t b = base + j;
                const uint32_t at = (c.bucket_start[b] + c.bucket_len[b]++) * ENTRY_WORDS;
                c.entries[at + 0] = t;
                c.entries[at + 1] = page;
                c.entries[at + 2] = k;
                break;
            }
        }
    }
}

// A share of a run, by fraction rather than count: token counts are data-dependent and unknown to the
// host, and integer arithmetic makes consecutive slices meet exactly whatever the count turns out to be.
uint32_t slice_begin(uint32_t n, uint32_t idx, uint32_t count) { return (n * idx) / count; }

// The reader -> sender ring. Two monotonic single-writer counters: this side owns `filled`, the sender
// owns `freed`, and each works on the difference, so neither needs a read-modify-write the other could
// race. `claimed` runs ahead of `published` so several token reads are in flight at once; BATCH <=
// NUM_L1_SLOTS/2 is what proves the reader cannot take every slot before announcing any.
struct Ring {
    uint32_t claimed = 0;
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

    void flush_publish() {
        if (published == claimed) {
            return;
        }
        // The tokens have to be in L1 before the sender is told the slots are filled.
        noc_async_read_barrier();
        noc_semaphore_inc(get_noc_addr(ct.filled_addr), claimed - published);
        published = claimed;
    }
};

uint32_t slot_addr(uint32_t slot) { return ct.ring_addr + slot * ct.slot_stride(); }

volatile tt_l1_ptr dspf2d::FwdMetadata* slot_tail(uint32_t slot) {
    return reinterpret_cast<volatile tt_l1_ptr dspf2d::FwdMetadata*>(slot_addr(slot) + ct.token_size_bytes);
}

}  // namespace

void kernel_main() {
    const Control c = carve_control();
    ASSERT(c.end <= ct.filled_addr);  // the control region must not run into the semaphores

    read_control_tables(c);
    read_indices(c);
    build_chip_experts(c);
    const uint32_t dropped = count_buckets(c);
    check_buckets(c, dropped);

    // Exclusive prefix sum, so the next increment can place each bucket's (token, page) pairs without
    // moving anything already counted.
    uint32_t at = 0;
    for (uint32_t i = 0; i < ct.extent * ct.experts_per_chip; i++) {
        c.bucket_start[i] = at;
        at += c.bucket_len[i];
    }

    fill_entries(c);

    const auto in_acc = TensorAccessor(dspf2d::ReaderCtArgs::in_args, get_arg_val<uint32_t>(0));
    const auto out_acc = TensorAccessor(dspf2d::ReaderCtArgs::out_payload_args, get_arg_val<uint32_t>(6));
    const auto meta_acc = TensorAccessor(dspf2d::ReaderCtArgs::out_meta_args, get_arg_val<uint32_t>(7));

    Ring ring;
    // Own assignments bound for the chip across this cable. Those are single hops that land straight in
    // the destination's output, so they need nothing from the forwarding region -- destinations further
    // round do, and arrive with the relay.
    for (uint32_t a = 0; a < ct.num_own; a++) {
        const uint32_t base = ct.assignment_base + a * dspf2d::ASSIGNMENT_WORDS;
        const uint32_t dst_chip = kernel_compile_time_args[base + 0];
        const uint32_t dst_row = kernel_compile_time_args[base + 1];
        const uint32_t split_idx = kernel_compile_time_args[base + 2];
        const uint32_t split_count = kernel_compile_time_args[base + 3];
        if (dst_chip != ct.nbr_chip_id) {
            continue;
        }
        for (uint32_t j = 0; j < ct.experts_per_chip; j++) {
            const uint32_t b = dst_row * ct.experts_per_chip + j;
            const uint32_t n = c.bucket_len[b];
            const uint32_t from = slice_begin(n, split_idx, split_count);
            const uint32_t to = slice_begin(n, split_idx + 1, split_count);
            for (uint32_t i = from; i < to; i++) {
                const uint32_t at = (c.bucket_start[b] + i) * ENTRY_WORDS;
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
                tail->cmd = dspf2d::CMD_FINAL_WRITE;
                tail->this_addr = tail->final_payload_addr;
            }
        }
    }
    ring.flush_publish();

    // Relays and the local same-chip phase land next.
    const uint32_t end_slot = ring.claim_slot();
    slot_tail(end_slot)->cmd = dspf2d::CMD_END;
    ring.flush_publish();

    noc_async_atomic_barrier();
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.fwd_sem_addr), 0);
}
