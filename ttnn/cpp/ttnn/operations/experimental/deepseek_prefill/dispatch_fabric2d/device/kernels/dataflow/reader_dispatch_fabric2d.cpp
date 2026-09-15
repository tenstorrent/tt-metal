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
    volatile tt_l1_ptr uint32_t* mc_meta;       // fanout: 4 words per destination of one page
    volatile tt_l1_ptr uint32_t* mc_count;      // fanout: entries emitted per direction
    // fanout: one reach row per (origin, direction), each padded to 64 bytes. An address rather than a
    // pointer because the pad makes the stride wider than the row.
    uint32_t reach;
    volatile tt_l1_ptr uint32_t* in_start;   // page offset of each chunk this stream reads
    volatile tt_l1_ptr uint32_t* out_start;  // page offset of each chunk it writes downstream
    uint32_t end;
};

// The geometry the control region is sized from. The host builds the same struct and reserves
// control_region_bytes of it; carve_control below walks the same block list in the same order, so the
// two cannot drift apart the way an independently maintained sum did, twice.
dspf2d::ControlGeometry control_geometry() {
    dspf2d::ControlGeometry g;
    g.seq_len = ct.seq_len;
    g.indices_pad_stride = ct.indices_pad_stride;
    g.extent = ct.extent;
    g.num_routed_experts = ct.num_routed_experts;
    g.experts_per_chip = ct.experts_per_chip;
    g.topk = ct.topk;
    g.num_relay = ct.num_relay;
    return g;
}

Control carve_control() {
    const dspf2d::ControlGeometry g = control_geometry();
    uint32_t a = ct.control_addr;
    const auto take = [&](uint32_t block) {
        const uint32_t at = a;
        a += dspf2d::control_block_bytes(g, block);
        return at;
    };

    Control c;
    c.indices = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(take(dspf2d::kCbIndices));
    c.offsets = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbOffsets));
    c.counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbCounts));
    c.region = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbRegion));
    c.table = reinterpret_cast<volatile tt_l1_ptr int32_t*>(take(dspf2d::kCbTable));
    c.alloc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbAlloc));
    c.chip_experts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbChipExperts));
    c.bucket_len = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbBucketLen));
    c.bucket_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbBucketStart));
    c.entries = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbEntries));
    c.mc_meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbMcMeta));
    c.mc_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbMcCount));
    c.reach = take(dspf2d::kCbReach);
    c.in_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbInStart));
    c.out_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbOutStart));
    c.end = a;
    // The host reserved exactly this, from the same list. A carve that outgrew the reservation would
    // run into the global semaphores, so say so here rather than corrupting them.
    ASSERT(c.end - ct.control_addr == dspf2d::control_region_bytes(g));
    return c;
}

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
    if (ct.fanout) {
        const auto reach_acc = TensorAccessor(
            dspf2d::ReaderCtArgs::reach_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kFanoutReachAddr));
        const uint32_t hops = dspf2d::mc_reach_hops(ct.extent);
        const uint32_t stride = dspf2d::mc_reach_row_bytes(ct.extent);
        // Row by row into 64-byte-padded slots. A row is hops * 4 bytes, which is never a multiple of
        // 64, and a DRAM read needs a 64-byte-aligned L1 destination on Blackhole: packed, every row
        // after the first lands at a wrong address and every chunk length downstream is garbage.
        for (uint32_t o = 0; o < ct.extent; o++) {
            for (uint32_t d = 0; d < 2u; d++) {
                noc_async_read(reach_acc.get_noc_addr(o * 2u + d), c.reach + (o * 2u + d) * stride, hops * 4u);
            }
        }
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
// The production op advances the counter for a token it drops for want of capacity and emits nothing,
// and so does this: the pages of every later token depend on it.
void count_buckets(const Control& c) {
    for (uint32_t e = 0; e < ct.num_routed_experts; e++) {
        c.alloc[e] = c.offsets[ct.my_row * ct.num_routed_experts + e];
    }
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
}

// Tokens one origin chip owes one expert. Every chip on the axis computes this identically from the
// replicated table, which is what lets a chip size a run it neither wrote nor receives. Rows are
// absolute buffer positions, so the last origin closes against counts + region, not counts alone.
uint32_t run_len(const Control& c, uint32_t origin_row, uint32_t e) {
    const uint32_t at = c.offsets[origin_row * ct.num_routed_experts + e];
    const uint32_t routed = (origin_row + 1 < ct.extent) ? c.offsets[(origin_row + 1) * ct.num_routed_experts + e] - at
                                                         : c.counts[e] + c.region[e] - at;
    // The table counts every token routed to e, but the origin drops the ones past the expert's
    // capacity while still advancing the counter. Its allocator starts at `at` and walks up, so the
    // survivors are exactly the first `cap - at`. Sizing a chunk by `routed` would make every reader
    // downstream wait for pages the origin never sent.
    const uint32_t room = ct.max_dispatch_buf_tokens > at ? ct.max_dispatch_buf_tokens - at : 0;
    return routed < room ? routed : room;
}

// The free consistency check, and the reason to build the index before moving a byte: this chip owes
// expert e exactly as many tokens as its own row of the offsets table says, because that table was
// derived from the same routing. Rows are absolute buffer positions, so the last row closes against
// counts + region rather than counts alone.
//
// A mismatch means the replay diverged from the production allocator -- which would otherwise surface as
// wrong pages or, worse, as a deadlock once chunk lengths are computed from these same numbers.
void check_buckets(const Control& c) {
    for (uint32_t row = 0; row < ct.extent; row++) {
        for (uint32_t j = 0; j < ct.experts_per_chip; j++) {
            const uint32_t e = c.chip_experts[row * ct.experts_per_chip + j];
            ASSERT(c.bucket_len[row * ct.experts_per_chip + j] == run_len(c, ct.my_row, e));
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

// Which way round the ring a destination lies, and how far. A tie at exactly half the ring goes
// clockwise, matching how the reach table was built -- pick the other way here and a chunk's length
// stops agreeing with what the origin actually sends.
uint32_t mc_dir_of(uint32_t my_row, uint32_t dst_row, uint32_t* hop_out) {
    const uint32_t cw = (dst_row + ct.extent - my_row) % ct.extent;
    const uint32_t ccw = ct.extent - cw;
    if (cw <= ccw) {
        *hop_out = cw;
        return 0;
    }
    *hop_out = ccw;
    return 1;
}

// Collapse each token to at most one entry per direction: the token, how many destinations it carries
// that way, then one packed word per destination. Replays the same allocator the unicast path does --
// same drop rule, same page numbering -- so both modes land a token on the page `dispatch` would.
void build_multicast_entries(const Control& c) {
    const uint32_t stride = dspf2d::fo_entry_words(ct.topk);
    c.mc_count[0] = 0;
    c.mc_count[1] = 0;
    for (uint32_t e = 0; e < ct.num_routed_experts; e++) {
        c.alloc[e] = c.offsets[ct.my_row * ct.num_routed_experts + e];
    }
    for (uint32_t t = 0; t < ct.seq_len; t++) {
        volatile tt_l1_ptr uint16_t* idx =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>((uint32_t)c.indices + t * ct.indices_pad_stride);
        uint32_t n_dir[2] = {0, 0};
        uint32_t packed[2][dspf2d::FO_MAX_DESTS];
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
            if ((uint32_t)row == ct.my_row) {
                continue;  // the local phase owns these; they never touch a cable
            }
            uint32_t hop = 0;
            const uint32_t d = mc_dir_of(ct.my_row, (uint32_t)row, &hop);
            if (n_dir[d] < dspf2d::FO_MAX_DESTS) {
                packed[d][n_dir[d]++] = (page & dspf2d::FO_PAGE_MASK) |
                                        ((hop & dspf2d::FO_HOP_MASK) << dspf2d::FO_HOP_SHIFT) |
                                        (k << dspf2d::FO_SLOT_SHIFT);
            }
        }
        for (uint32_t d = 0; d < 2; d++) {
            if (n_dir[d] == 0) {
                continue;
            }
            const uint32_t at = c.mc_count[d]++;
            volatile tt_l1_ptr uint32_t* ent = c.entries + (d * ct.seq_len + at) * stride;
            ent[0] = t;
            ent[1] = n_dir[d];
            for (uint32_t i = 0; i < n_dir[d]; i++) {
                ent[2 + i] = packed[d][i];
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

    // Hand back a slot claimed as scratch. Only valid while nothing has been published for it, which
    // is what keeps the sender from ever seeing it.
    void release_slot() { claimed--; }

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

// Under fan-out a chunk is (origin, hop) and its length is how many of that origin's tokens are still
// in flight at that hop -- not a per-expert count, which is a marginal and cannot express it.
uint32_t mc_reach(const Control& c, uint32_t origin_row, uint32_t dir_idx, uint32_t hop) {
    const uint32_t stride = dspf2d::mc_reach_row_bytes(ct.extent);
    volatile tt_l1_ptr uint32_t* row =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c.reach + (origin_row * 2u + dir_idx) * stride);
    return row[hop];
}

// Tokens from one origin whose farthest destination this way is EXACTLY `far` hops off. reach is
// cumulative and terminated by a zero at m + 1, so the classes partition its tokens.
uint32_t mc_class_size(const Control& c, uint32_t origin_row, uint32_t dir_idx, uint32_t far) {
    return mc_reach(c, origin_row, dir_idx, far) - mc_reach(c, origin_row, dir_idx, far + 1);
}

// Which link carries a token: its rank within its farthest-hop class, sliced the usual way.
//
// The obvious rule -- slice the whole hop-1 list -- is WRONG for more than one link, and wrong as a
// deadlock rather than as wrong data. A multicast chunk shrinks as it travels, so a link's contiguous
// share of what remains is not the share the next hop's list would hand it: with four tokens whose
// farthest hops are (2, 2, 1, 1), link 0 takes both far-2 tokens and forwards two pages where the
// downstream, slicing reach[2] = 2 in half, waits for one from each link. Nothing about the data
// prevents that, and no rule over the hop-1 list avoids it.
//
// Splitting each farthest-hop class instead is stable: a token keeps its link for the whole journey,
// so a link's pages at hop h are exactly its share of the classes with far >= h. Class sizes come out
// of the same reach table both sides already hold, so this still needs no communication -- and with
// one link it telescopes back to reach[hop].
uint32_t mc_link_of(uint32_t rank, uint32_t class_size) {
    for (uint32_t link = 0; link + 1 < ct.num_links; link++) {
        if (rank < slice_begin(class_size, link + 1, ct.num_links)) {
            return link;
        }
    }
    return ct.num_links - 1;
}

uint32_t mc_chunk_len(const Control& c, uint32_t origin_row, uint32_t dir_idx, uint32_t hop, uint32_t link) {
    const uint32_t m = ct.extent / 2u;
    uint32_t len = 0;
    for (uint32_t far = hop; far <= m; far++) {
        const uint32_t n = mc_class_size(c, origin_row, dir_idx, far);
        len += slice_begin(n, link + 1, ct.num_links) - slice_begin(n, link, ct.num_links);
    }
    return len;
}

// Farthest hop of a staged multicast entry, which is the class it is split by. Derived from the
// entry rather than stored, so there is one definition of a destination's hop.
uint32_t mc_entry_far(volatile tt_l1_ptr uint32_t* ent) {
    uint32_t far = 0;
    for (uint32_t i = 0; i < ent[1]; i++) {
        const uint32_t hop = (ent[2 + i] >> dspf2d::FO_HOP_SHIFT) & dspf2d::FO_HOP_MASK;
        if (hop > far) {
            far = hop;
        }
    }
    return far;
}

// Where this chip sits relative to an origin `j` hops upstream along the stream's direction.
uint32_t mc_row_back(uint32_t j, int32_t travel) {
    const int32_t e = static_cast<int32_t>(ct.extent);
    return static_cast<uint32_t>(((static_cast<int32_t>(ct.my_row) - static_cast<int32_t>(j) * travel) % e + e) % e);
}

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

// The same, for multicast, where a chunk is (origin, hop) and there are extent/2 of them either way.
//
// Outgoing is this chip's own tokens followed by each upstream origin pushed one hop further;
// incoming is the same origins one hop back. Those are the same m numbers offset by one hop, which is
// what makes a region both sides derive alone still agree -- and the reason the two lists are
// generated by one function rather than two.
uint32_t mc_chunk_starts(
    const Control& c,
    uint32_t dir_idx,
    uint32_t link,
    int32_t travel,
    bool outgoing,
    volatile tt_l1_ptr uint32_t* start) {
    const uint32_t m = ct.extent / 2u;
    uint32_t at = 0;
    for (uint32_t i = 0; i < m; i++) {
        const uint32_t origin = mc_row_back(i + (outgoing ? 0u : 1u), travel);
        start[i] = at;
        at += mc_chunk_len(c, origin, dir_idx, i + 1u, link);
    }
    ASSERT(at <= ct.fwd_pages_per_stream);
    return at;
}

uint32_t slot_addr(uint32_t slot) { return ct.ring_addr + slot * ct.slot_stride(); }

volatile tt_l1_ptr dspf2d::FwdMetadata* slot_tail(uint32_t slot) {
    return reinterpret_cast<volatile tt_l1_ptr dspf2d::FwdMetadata*>(slot_addr(slot) + ct.token_size_bytes);
}

// The fan-out tail of the same slot. A different view of the same 64 bytes: only one mode runs, and
// the two layouts agree on where `cmd` and `this_addr` sit so the sender need not know which.
volatile tt_l1_ptr dspf2d::FanoutMetadata* slot_mc_tail(uint32_t slot) {
    return reinterpret_cast<volatile tt_l1_ptr dspf2d::FanoutMetadata*>(slot_addr(slot) + ct.token_size_bytes);
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
            const uint32_t n = c.bucket_len[b];
            const uint32_t from = slice_begin(n, split_idx, split_count);
            const uint32_t to = slice_begin(n, split_idx + 1, split_count);
            const uint32_t out_base = direct ? 0 : c.out_start[a * ct.experts_per_chip + j];
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
            }
        }
    }
}

// Pages this stream relays: read out of its own forwarding region and pushed one hop further, or
// delivered if the chip across the cable is where they were going.
template <typename FwdAcc>
void relay_phase(const Control& c, Ring& ring, const FwdAcc& fwd_acc, uint32_t my_region, uint32_t nbr_row) {
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
            const uint32_t in_base = c.in_start[d * ct.experts_per_chip + j];
            const uint32_t out_base = continues ? c.out_start[this_out_d * ct.experts_per_chip + j] : 0;
            volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.fwd_sem_addr);
            for (uint32_t p = 0; p < len; p++) {
                // Upstream fills the region strictly left to right, so its page count is the high-water
                // offset and a page is ready once that count passes it.
                // The invalidate has to precede the read: an upstream chip owns this counter, so a
                // cached line would never show its increments.
                while (true) {
                    invalidate_l1_cache();
                    if (*arrived > in_base + p) {
                        break;
                    }
                    ring.flush_publish();  // let our own sender work while we wait on upstream
                }
                const uint32_t slot = ring.claim_slot();
                noc_async_read(
                    fwd_acc.get_noc_addr(my_region + in_base + p),
                    slot_addr(slot),
                    ct.token_size_bytes + dspf2d::FWD_EXTRA_BYTES);
                noc_async_read_barrier();  // the routing tail decides the next hop, so it must be here
                // The read landed behind the data cache, and this slot carried a different page eight
                // iterations ago: without this, the tail's addresses can still be that page's.
                invalidate_l1_cache();

                volatile tt_l1_ptr dspf2d::FwdMetadata* tail = slot_tail(slot);
                // Whether this hop is the last is a property of the CHUNK, so it comes from the
                // descriptor rather than from the arriving tail: every page of (origin, dst_row) shares
                // one destination, and the descriptor lists are the pair validate_chunk_agreement
                // proved the two chips agree on. Reading it back out of the tail would instead make
                // control flow depend on a DRAM round-trip, and an unwritten tail reads as chip 0 --
                // indistinguishable from a genuine destination on the chip whose id is 0.
                ASSERT(tail->dst_chip == (uint64_t)dst_chip);
                if (!continues) {
                    tail->cmd = dspf2d::CMD_FINAL_WRITE;
                    tail->this_addr = tail->final_payload_addr;
                } else {
                    tail->cmd = (p + 1 == len) ? dspf2d::CMD_FORWARD_END : dspf2d::CMD_FORWARD;
                    tail->this_addr = fwd_acc.get_noc_addr(my_region + out_base + p);
                }
            }
        }
    }
}

template <typename InAcc, typename OutAcc, typename MetaAcc>
void local_phase(const Control& c, Ring& ring, const InAcc& in_acc, const OutAcc& out_acc, const MetaAcc& meta_acc) {
    // Tokens routed to an expert this chip hosts. They never touch the fabric, so this is a DRAM read
    // and two DRAM writes, and the slot is scratch that is deliberately never published -- publishing
    // it would put these tokens on the cable as well. Every stream runs this over its own fraction, so
    // the split has to cover exactly once, the same rule the remote assignments follow.
    {
        const uint32_t scratch = ring.claim_slot();
        const uint32_t scratch_addr = slot_addr(scratch);
        volatile tt_l1_ptr dspf2d::FwdMetadata* tail = slot_tail(scratch);
        tail->pad = 0;
        for (uint32_t j = 0; j < ct.experts_per_chip; j++) {
            const uint32_t b = ct.my_row * ct.experts_per_chip + j;
            const uint32_t n = c.bucket_len[b];
            const uint32_t from = slice_begin(n, ct.stream, 2 * ct.num_links);
            const uint32_t to = slice_begin(n, ct.stream + 1, 2 * ct.num_links);
            for (uint32_t i = from; i < to; i++) {
                const uint32_t at = (c.bucket_start[b] + i) * ENTRY_WORDS;
                const uint32_t token = c.entries[at + 0];
                const uint32_t page = c.entries[at + 1];
                noc_async_read(in_acc.get_noc_addr(token), scratch_addr, ct.token_size_bytes);
                tail->meta[0] = ct.linearized_coord;
                tail->meta[1] = token;
                tail->meta[2] = c.entries[at + 2];
                noc_async_read_barrier();
                noc_async_write(scratch_addr, out_acc.get_noc_addr(page), ct.token_size_bytes);
                noc_async_write(
                    scratch_addr + ct.token_size_bytes, meta_acc.get_noc_addr(page), dspf2d::METADATA_WIRE_BYTES);
                // The slot IS the buffer, so it cannot be refilled until both writes have read it out.
                noc_async_write_barrier();
            }
        }
        ring.release_slot();
    }
}

// --- Fan-out -------------------------------------------------------------------------------------
//
// One page per token per DIRECTION, carrying its own destination list. It travels to the farthest
// destination that way; every chip en route keeps the pages addressed to it and passes the rest on.
// There is no final write on the cable: the neighbour consumes out of its own forwarding region, so
// every slot this mode publishes is a CMD_FORWARD.

// This chip's own tokens, one entry per (token, direction), narrowed to this link's share.
//
// The share is by farthest-hop class, which is what makes a link's count at every later hop derivable
// from the same reach table -- see mc_link_of.
template <typename InAcc, typename FwdAcc>
void mc_own_phase(
    const Control& c,
    Ring& ring,
    const InAcc& in_acc,
    const FwdAcc& fwd_acc,
    uint32_t my_region,
    uint32_t dir_idx,
    uint32_t link) {
    const uint32_t m = ct.extent / 2u;
    const uint32_t stride = dspf2d::fo_entry_words(ct.topk);
    const uint32_t len = mc_chunk_len(c, ct.my_row, dir_idx, 1u, link);
    // The table is built from this chip's own routing, so its first hop has to be the entries just
    // staged. If it is not, every chunk length downstream is wrong and the axis deadlocks.
    ASSERT(c.mc_count[dir_idx] == mc_reach(c, ct.my_row, dir_idx, 1u));
    ASSERT(m + 1u <= dspf2d::MC_MAX_HOPS);

    uint32_t rank[dspf2d::MC_MAX_HOPS];
    for (uint32_t h = 0; h < dspf2d::MC_MAX_HOPS; h++) {
        rank[h] = 0;
    }

    uint32_t q = 0;
    for (uint32_t i = 0; i < c.mc_count[dir_idx]; i++) {
        volatile tt_l1_ptr uint32_t* ent = c.entries + (dir_idx * ct.seq_len + i) * stride;
        const uint32_t far = mc_entry_far(ent);
        // Counted for every entry, not just this link's: the rank is a position in the whole class.
        const uint32_t r = rank[far]++;
        if (mc_link_of(r, mc_class_size(c, ct.my_row, dir_idx, far)) != link) {
            continue;
        }
        const uint32_t token = ent[0];
        const uint32_t n_dests = ent[1];
        const uint32_t slot = ring.claim_slot();
        noc_async_read(in_acc.get_noc_addr(token), slot_addr(slot), ct.token_size_bytes);

        // Hops are measured from HERE and never rewritten, which is what lets a page be immutable in
        // flight: a chip j hops along takes the destinations with hop == j and forwards the rest.
        volatile tt_l1_ptr dspf2d::FanoutMetadata* tail = slot_mc_tail(slot);
        tail->src_chip = ct.linearized_coord;
        tail->token = token;
        for (uint32_t d = 0; d < dspf2d::FO_MAX_DESTS; d++) {
            tail->dests[d] = (d < n_dests) ? ent[2 + d] : 0u;
        }
        // The last page of a chunk forces the downstream bump, which is the boundary that reader
        // switches on: leave it uncounted and the whole axis waits.
        tail->cmd = (q + 1 == len) ? dspf2d::CMD_FORWARD_END : dspf2d::CMD_FORWARD;
        tail->this_addr = fwd_acc.get_noc_addr(my_region + c.out_start[0] + q);
        q++;
    }
    ASSERT(q == len);
}

// Arrivals. Chunk j came from the origin j hops upstream, so its destinations at hop j are THIS chip:
// they are written locally, out of the slot the page was read into. What is left travels on.
//
// The read count and the forward count are different numbers -- reach[j] against reach[j + 1] -- and
// that is the point of the mode rather than a rounding artefact. Both sides derive their own, order is
// preserved, and a page carries its own destinations, so a dense region still lines up.
template <typename OutAcc, typename MetaAcc, typename FwdAcc>
void mc_relay_phase(
    const Control& c,
    Ring& ring,
    const OutAcc& out_acc,
    const MetaAcc& meta_acc,
    const FwdAcc& fwd_acc,
    uint32_t my_region,
    uint32_t dir_idx,
    uint32_t link,
    int32_t travel) {
    const uint32_t m = ct.extent / 2u;
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.fwd_sem_addr);

    for (uint32_t j = 1; j <= m; j++) {
        const uint32_t origin = mc_row_back(j, travel);
        const uint32_t len = mc_chunk_len(c, origin, dir_idx, j, link);
        // Nothing travels past half the ring, so the last chunk is consumed whole.
        const uint32_t fwd_len = (j < m) ? mc_chunk_len(c, origin, dir_idx, j + 1u, link) : 0u;
        const uint32_t in_base = c.in_start[j - 1u];
        const uint32_t out_base = (j < m) ? c.out_start[j] : 0u;
        uint32_t q = 0;
        for (uint32_t p = 0; p < len; p++) {
            // Upstream fills the region strictly left to right, so its page count is the high-water
            // offset and a page is ready once that count passes it.
            // The invalidate has to precede the read: an upstream chip owns this counter, so a
            // cached line would never show its increments.
            while (true) {
                invalidate_l1_cache();
                if (*arrived > in_base + p) {
                    break;
                }
                ring.flush_publish();  // let our own sender work while we wait on upstream
            }
            const uint32_t slot = ring.claim_slot();
            noc_async_read(
                fwd_acc.get_noc_addr(my_region + in_base + p),
                slot_addr(slot),
                ct.token_size_bytes + dspf2d::FWD_EXTRA_BYTES);
            noc_async_read_barrier();  // the destination list decides the next hop, so it must be here
            // The read landed behind the data cache, and this slot carried a different page eight
            // iterations ago: without this, the tail can still be that page's.
            invalidate_l1_cache();

            volatile tt_l1_ptr dspf2d::FanoutMetadata* tail = slot_mc_tail(slot);
            const uint32_t src_chip = tail->src_chip;
            const uint32_t token = tail->token;
            uint32_t kept = 0;
            bool travels_on = false;
            for (uint32_t d = 0; d < dspf2d::FO_MAX_DESTS; d++) {
                const uint32_t packed = tail->dests[d];
                const uint32_t hop = (packed >> dspf2d::FO_HOP_SHIFT) & dspf2d::FO_HOP_MASK;
                if (hop > j) {
                    travels_on = true;
                    continue;
                }
                // Hop 0 is an unused slot, and a hop below j was consumed by a chip behind us: the
                // page is never rewritten in flight, so both are still here and neither is ours.
                if (hop != j) {
                    continue;
                }
                // A token can hold several of this chip's experts, so each destination needs its own
                // metadata words -- one shared scratch would be overwritten under the write still
                // reading it.
                volatile tt_l1_ptr uint32_t* meta = c.mc_meta + kept * (dspf2d::MC_META_SLOT_BYTES / 4u);
                meta[0] = src_chip;
                meta[1] = token;
                meta[2] = packed >> dspf2d::FO_SLOT_SHIFT;
                meta[3] = 0;
                noc_async_write(
                    slot_addr(slot), out_acc.get_noc_addr(packed & dspf2d::FO_PAGE_MASK), ct.token_size_bytes);
                noc_async_write(
                    (uint32_t)meta, meta_acc.get_noc_addr(packed & dspf2d::FO_PAGE_MASK), dspf2d::METADATA_WIRE_BYTES);
                kept++;
            }
            // Every page in chunk j has a destination at hop j or beyond, or it would not be here.
            ASSERT(kept > 0 || travels_on);
            if (kept > 0) {
                // The slot is the source of those writes and is about to be either forwarded from or
                // handed back, so they have to have read it out first.
                noc_async_write_barrier();
            }
            if (travels_on) {
                tail->cmd = (q + 1 == fwd_len) ? dspf2d::CMD_FORWARD_END : dspf2d::CMD_FORWARD;
                tail->this_addr = fwd_acc.get_noc_addr(my_region + out_base + q);
                q++;
            } else {
                // Consumed here. Nothing was published for this slot, so handing it straight back is
                // what keeps a fully consumed page off the cable -- there is no drop command, and the
                // sender walks the ring in order.
                ring.release_slot();
            }
        }
        ASSERT(q == fwd_len);
    }
}

}  // namespace

void kernel_main() {
    const Control c = carve_control();
    ASSERT(c.end <= ct.filled_addr);  // the control region must not run into the semaphores

    read_control_tables(c);
    read_indices(c);
    build_chip_experts(c);
    count_buckets(c);
    check_buckets(c);

    // Exclusive prefix sum, so the next increment can place each bucket's (token, page) pairs without
    // moving anything already counted.
    uint32_t at = 0;
    for (uint32_t i = 0; i < ct.extent * ct.experts_per_chip; i++) {
        c.bucket_start[i] = at;
        at += c.bucket_len[i];
    }

    fill_entries(c);

    const auto in_acc =
        TensorAccessor(dspf2d::ReaderCtArgs::in_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kInputAddr));
    const auto out_acc = TensorAccessor(
        dspf2d::ReaderCtArgs::out_payload_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kOutPayloadAddr));
    const auto meta_acc =
        TensorAccessor(dspf2d::ReaderCtArgs::out_meta_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kOutMetaAddr));

    const auto fwd_acc =
        TensorAccessor(dspf2d::ReaderCtArgs::fwd_args, get_arg_val<uint32_t>(dspf2d::ReaderRtArg::kFwdAddr));
    const uint32_t my_region = ct.stream * ct.fwd_pages_per_stream;

    chunk_starts(c, ct.in_chunks_base, c.in_start);
    chunk_starts(c, ct.out_chunks_base, c.out_start);

    // Which position on the axis the chip across this cable holds. The outgoing list is ordered by it,
    // and a page bound for it is delivered rather than forwarded.
    uint32_t nbr_row = 0;
    for (uint32_t r = 0; r < ct.extent; r++) {
        if (kernel_compile_time_args[ct.ring_chip_ids_base + r] == ct.nbr_chip_id) {
            nbr_row = r;
        }
    }

    Ring ring;
    if (ct.fanout) {
        // The local phase reads the unicast routing index, and build_multicast_entries overwrites it:
        // the two layouts share one region because only one mode ever runs. So this chip's own pages
        // are placed before the index they were built from is replaced.
        local_phase(c, ring, in_acc, out_acc, meta_acc);
        build_multicast_entries(c);

        // Clockwise is direction 0 on both sides -- mc_dir_of resolves a tie at exactly half the ring
        // the same way the reach table was built. Differ in one place and the lengths silently
        // disagree.
        const uint32_t dir_idx = ct.stream % 2u;
        const uint32_t link = ct.stream / 2u;
        const int32_t travel = (dir_idx == 0u) ? 1 : -1;
        mc_chunk_starts(c, dir_idx, link, travel, /*outgoing=*/false, c.in_start);
        mc_chunk_starts(c, dir_idx, link, travel, /*outgoing=*/true, c.out_start);

        mc_own_phase(c, ring, in_acc, fwd_acc, my_region, dir_idx, link);
        ring.flush_publish();
        mc_relay_phase(c, ring, out_acc, meta_acc, fwd_acc, my_region, dir_idx, link, travel);
        ring.flush_publish();
    } else {
        chunk_starts(c, ct.in_chunks_base, c.in_start);
        chunk_starts(c, ct.out_chunks_base, c.out_start);

        own_phase(c, ring, in_acc, out_acc, meta_acc, fwd_acc, my_region);
        ring.flush_publish();
        relay_phase(c, ring, fwd_acc, my_region, nbr_row);
        ring.flush_publish();
        local_phase(c, ring, in_acc, out_acc, meta_acc);
    }

    const uint32_t end_slot = ring.claim_slot();
    slot_tail(end_slot)->cmd = dspf2d::CMD_END;
    ring.flush_publish();

    noc_async_atomic_barrier();
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.fwd_sem_addr), 0);
}
