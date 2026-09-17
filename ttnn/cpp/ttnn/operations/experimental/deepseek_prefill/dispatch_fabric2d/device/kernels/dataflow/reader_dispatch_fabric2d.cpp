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
    volatile tt_l1_ptr uint32_t* expert_slot;   // the same domain, packed as ES_* for the routing pass
    volatile tt_l1_ptr uint32_t* alloc;         // extent x experts_per_chip, the running allocator by slot
    volatile tt_l1_ptr uint32_t* chip_experts;  // extent x experts_per_chip, ascending global expert id
    volatile tt_l1_ptr uint32_t* row_fill;      // extent, while the chip -> experts inverse is built
    volatile tt_l1_ptr uint32_t* bucket_fill;   // extent x experts_per_chip, the fill cursor
    volatile tt_l1_ptr uint32_t* bucket_start;  // extent x experts_per_chip + 1, inclusive prefix sums
    volatile tt_l1_ptr uint32_t* entries;       // 3 words per surviving (token, top-k slot)
    volatile tt_l1_ptr uint32_t* mc_entries;    // fanout: one entry per (token, direction)
    volatile tt_l1_ptr uint32_t* mc_count;      // fanout: entries emitted per direction
    // fanout: one reach row per (origin, direction), each padded to 64 bytes. An address rather than a
    // pointer because the pad makes the stride wider than the row.
    uint32_t reach;
    volatile tt_l1_ptr uint32_t* padding;    // [real_token_count, pad_side], when one was supplied
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
    g.fanout = ct.fanout;
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
    c.expert_slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbExpertSlot));
    c.alloc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbAlloc));
    c.chip_experts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbChipExperts));
    c.row_fill = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbRowFill));
    c.bucket_fill = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbBucketFill));
    c.bucket_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbBucketStart));
    c.entries = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbEntries));
    c.mc_entries = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbMcEntries));
    c.mc_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbMcCount));
    c.reach = take(dspf2d::kCbReach);
    c.padding = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbPadding));
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
    // capacity while still advancing the counter. Its allocator starts at `at` and walks up, so the
    // survivors are exactly the first `cap - at`. Sizing a chunk by `routed` would make every reader
    // downstream wait for pages the origin never sent.
    const uint32_t room = ct.max_dispatch_buf_tokens > at ? ct.max_dispatch_buf_tokens - at : 0;
    return routed < room ? routed : room;
}

// The token's index on this chip, the page it lands on at the destination, and which top-k slot it
// came from. The last two both travel to the destination -- the page as an address, the slot as
// metadata field 2. Sized from the same expression the control block is reserved with.
constexpr uint32_t ENTRY_WORDS = dspf2d::entry_words();

// Which way round the ring a destination lies, and how far. A tie at exactly half the ring goes
// clockwise, matching how the reach table was built -- pick the other way here and a chunk's length
// stops agreeing with what the origin actually sends.
uint32_t mc_dir_of(uint32_t my_row, uint32_t dst_row, uint32_t* hop_out) {
    // Both rows are below extent, so the wrap is one conditional subtract rather than a modulo by a
    // value only known at run time, which on this RISC is a called division.
    uint32_t cw = dst_row + ct.extent - my_row;
    if (cw >= ct.extent) {
        cw -= ct.extent;
    }
    const uint32_t ccw = ct.extent - cw;
    if (cw <= ccw) {
        *hop_out = cw;
        return 0;
    }
    *hop_out = ccw;
    return 1;
}

// expert -> everything resolving a pick needs, in ONE word, and the chip -> experts inverse beside it.
//
// Every chip on the axis builds the same inverse because the dispatch table is replicated along it,
// which is what lets a relay expand a (origin, destination) descriptor into the same experts_per_chip
// chunks the writer expanded it into. The packed word is per-chip on top of that -- it carries the hop
// and the direction as measured from HERE -- so that the routing pass never looks a destination up.
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
        uint32_t w = slot;
        if constexpr (ct.fanout) {
            if (r == ct.my_row) {
                w |= dspf2d::ES_LOCAL_BIT;
            } else {
                uint32_t hop = 0;
                const uint32_t d = mc_dir_of(ct.my_row, r, &hop);
                w |= (d ? dspf2d::ES_DIR_BIT : 0u) | ((hop & dspf2d::FO_HOP_MASK) << dspf2d::FO_HOP_SHIFT);
            }
        }
        c.expert_slot[e] = w;
    }
    for (uint32_t r = 0; r < ct.extent; r++) {
        // The whole protocol sizes a relayed chunk group as experts_per_chip terms, so a chip hosting a
        // different number would desynchronise the writer and the reader of a forwarding region.
        ASSERT(c.row_fill[r] == ct.experts_per_chip);
    }
}

// Where each bucket sits and how long it is, plus the per-expert allocator's starting point.
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
    const uint32_t local_first = ct.my_row * ct.experts_per_chip;
    // One entry per (token, pick) is what the block holds, and the total cannot exceed it. The lengths
    // come from host tensors that nothing ties to seq_len, so an offsets table inconsistent with the
    // input would otherwise run the fill -- and the tail zeroing behind it -- through every block after
    // this one and into the global semaphores. A consistent table never reaches the cap.
    const uint32_t max_entries = ct.seq_len * ct.topk;
    uint32_t at = 0;
    for (uint32_t b = 0; b < n_slots; b++) {
        const uint32_t e = c.chip_experts[b];
        c.alloc[b] = c.offsets[ct.my_row * ct.num_routed_experts + e];
        // Under fan-out every remote destination is collapsed into a multicast entry instead, so only
        // this chip's own experts get a bucket -- the local phase is their only reader.
        const bool bucketed = !ct.fanout || (b >= local_first && b < local_first + ct.experts_per_chip);
        c.bucket_start[b] = at;
        c.bucket_fill[b] = at;
        const uint32_t n = bucketed ? run_len(c, ct.my_row, e) : 0u;
        at = (at + n > max_entries) ? max_entries : at + n;
    }
    c.bucket_start[n_slots] = at;
}

// The routing index, built in ONE pass over every (token, pick).
//
// Replaying the production op's allocator EXACTLY is what makes the pages byte-identical to it,
// including the rule that a token past the buffer's capacity is dropped while its counter still
// advances: the pages of every later token depend on it. A pick costs one indexed load to resolve --
// bucket, hop and direction together -- and a survivor goes straight into the layout the running mode
// consumes.
//
// Under fan-out that is one entry per (token, direction) carrying the packed destinations and their
// farthest hop, plus a bucket entry for each destination on THIS chip. The two layouts have separate
// blocks, so one pass fills both.
// Tokens the routing pass walks. A padding_config shortens it to the real ones, exactly as the
// production op shortens its batch loop, and for the same reason: with right padding the real tokens
// hold the low indices, so the allocator reaches all of them before the first padded one.
//
// This cannot desynchronise the ring. Every chunk length comes from the offsets table, never from how
// far this loop ran, and supplying the config asserts that padded tokens are sentinel-marked -- their
// picks resolve to ES_NOT_HERE and contribute no page. Skipping them is skipping no-ops.
uint32_t routed_token_count(const Control& c) {
    if constexpr (ct.has_padding_config) {
        const uint32_t real = c.padding[0];
        const uint32_t pad_side = c.padding[1];
        if (pad_side == 0u && real < ct.seq_len) {
            return real;
        }
    }
    return ct.seq_len;
}

void build_routing_index(const Control& c) {
    const uint32_t cap = ct.max_dispatch_buf_tokens;
    const uint32_t mc_stride = dspf2d::fo_entry_words(ct.topk);
    [[maybe_unused]] uint32_t mc_n[2] = {0, 0};
    const uint32_t tokens = routed_token_count(c);
    uint32_t idx_addr = (uint32_t)c.indices;
    for (uint32_t t = 0; t < tokens; t++, idx_addr += ct.indices_pad_stride) {
        volatile tt_l1_ptr uint16_t* idx = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_addr);
        [[maybe_unused]] uint32_t n_dir[2] = {0, 0};
        [[maybe_unused]] uint32_t far_dir[2] = {0, 0};
        [[maybe_unused]] uint32_t packed[2][dspf2d::FO_MAX_DESTS];
        for (uint32_t k = 0; k < ct.topk; k++) {
            const uint32_t w = c.expert_slot[idx[k]];
            if (w == dspf2d::ES_NOT_HERE) {
                continue;  // the expert is not in this dispatch group
            }
            const uint32_t slot = w & dspf2d::ES_SLOT_MASK;
            const uint32_t page = c.alloc[slot];
            c.alloc[slot] = page + 1u;
            if (page >= cap) {
                continue;  // dropped for want of capacity, with the counter already advanced
            }
            if constexpr (ct.fanout) {
                if ((w & dspf2d::ES_LOCAL_BIT) == 0u) {
                    const uint32_t d = (w >> dspf2d::ES_DIR_SHIFT) & 1u;
                    if (n_dir[d] < dspf2d::FO_MAX_DESTS) {
                        const uint32_t hop_field = w & dspf2d::ES_HOP_FIELD;
                        packed[d][n_dir[d]++] =
                            (page & dspf2d::FO_PAGE_MASK) | hop_field | (k << dspf2d::FO_SLOT_SHIFT);
                        const uint32_t hop = hop_field >> dspf2d::FO_HOP_SHIFT;
                        if (hop > far_dir[d]) {
                            far_dir[d] = hop;
                        }
                    }
                    continue;  // the cable carries these; only this chip's own go in a bucket
                }
            }
            const uint32_t at = c.bucket_fill[slot];
            // The bucket was sized from the offsets table, which the same routing produced. A table
            // that disagrees would otherwise write over the next bucket, and the ASSERT below that
            // reports the disagreement is compiled out on this hardware.
            if (at >= c.bucket_start[slot + 1u]) {
                continue;
            }
            c.bucket_fill[slot] = at + 1u;
            volatile tt_l1_ptr uint32_t* ent = c.entries + at * ENTRY_WORDS;
            ent[0] = t;
            ent[1] = page;
            ent[2] = k;
        }
        if constexpr (ct.fanout) {
            for (uint32_t d = 0; d < 2u; d++) {
                if (n_dir[d] == 0) {
                    continue;
                }
                volatile tt_l1_ptr uint32_t* ent = c.mc_entries + (d * ct.seq_len + mc_n[d]++) * mc_stride;
                ent[0] = t;
                ent[1] = n_dir[d];
                ent[2] = far_dir[d];
                for (uint32_t i = 0; i < n_dir[d]; i++) {
                    ent[3 + i] = packed[d][i];
                }
            }
        }
    }
    if constexpr (ct.fanout) {
        c.mc_count[0] = mc_n[0];
        c.mc_count[1] = mc_n[1];
    }
    // Every bucket filled to exactly the length the offsets table sized it at -- the replay agreeing
    // with the production allocator, which is the reason to build the index before moving a byte. A
    // divergence would otherwise surface as wrong pages or, once chunk lengths are computed from these
    // same numbers, as a deadlock.
    //
    // The tail of a bucket the replay left short is neutralised rather than merely asserted about.
    // The phases take their run length from the bucket, not from the fill, so an entry the pass never
    // wrote would be read out of a control region nothing zeroes -- and its `page` word becomes a
    // fabric write to an arbitrary DRAM address on another chip. This costs one load per bucket when
    // the tables agree, which they do, and turns that into a duplicate write of token 0 to page 0.
    for (uint32_t b = 0; b < ct.extent * ct.experts_per_chip; b++) {
        ASSERT(c.bucket_fill[b] == c.bucket_start[b + 1u]);
        for (uint32_t at = c.bucket_fill[b]; at < c.bucket_start[b + 1u]; at++) {
            volatile tt_l1_ptr uint32_t* ent = c.entries + at * ENTRY_WORDS;
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

// A share of a run, by fraction rather than count: token counts are data-dependent and unknown to the
// host, and integer arithmetic makes consecutive slices meet exactly whatever the count turns out to be.
uint32_t slice_begin(uint32_t n, uint32_t idx, uint32_t count) { return (n * idx) / count; }

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

    // Hand back a slot claimed as scratch. Only valid for the most recent claim while nothing has been
    // published for it, which is what keeps the sender from ever seeing it.
    void release_slot() {
        ASSERT(claimed == ready + 1);
        claimed--;
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

// The reach index a region chunk is sized at. Whoever holds a page whose farthest destination is the
// chip across its cable writes that destination's output pages itself, the same fabric write the
// unicast path makes, so a page enters a forwarding region only while something BEYOND that region's
// chip is still outstanding. A chunk landing h hops out therefore carries the hop-(h + 1) population,
// the chunk at h = m is always empty, and a token whose only destination is the neighbour never
// touches a region at all.
//
// Delivering ahead only at the TERMINAL hop is what keeps that cheap. A relay that also delivered a
// hop-(j + 1) destination while forwarding the page would put the same payload on that cable twice;
// here the delivery replaces the forward rather than joining it. The origin is the one exception --
// see mc_own_phase.
//
// Both sides of a region derive this the same way, so the dense layout is unaffected: a region still
// holds exactly m chunk lengths drawn from the one reach row, so both ends compute identical offsets.
uint32_t mc_region_hop(uint32_t hop) { return hop + 1u; }

// Whether the chip `j` hops from the origin puts this page back on the cable, and whether it is the
// last holder and so delivers the page's remaining destinations itself. Both go through
// mc_region_hop, because a forward the downstream did not size its chunk for is a deadlock: these are
// the same number as `fwd_len` and it must stay that way through any edit to the rule above.
bool mc_forwards(uint32_t far, uint32_t j) { return far >= mc_region_hop(j + 1u); }
bool mc_delivers_ahead(uint32_t far, uint32_t j) { return far + 1u == mc_region_hop(j + 1u); }

uint32_t mc_chunk_len(const Control& c, uint32_t origin_row, uint32_t dir_idx, uint32_t hop, uint32_t link) {
    const uint32_t m = ct.extent / 2u;
    uint32_t len = 0;
    for (uint32_t far = hop; far <= m; far++) {
        const uint32_t n = mc_class_size(c, origin_row, dir_idx, far);
        len += slice_begin(n, link + 1, ct.num_links) - slice_begin(n, link, ct.num_links);
    }
    return len;
}

// Farthest hop of a page in flight, which is the class it is split by. A staged entry carries the
// same quantity in a field, because the pass that emitted it already knew it; a wire tail does not,
// because the 64 bytes are full -- it is zero-padded to FO_MAX_DESTS and hop 0 marks an unused word.
uint32_t mc_tail_far(volatile tt_l1_ptr dspf2d::FanoutMetadata* tail) {
    uint32_t far = 0;
    for (uint32_t d = 0; d < dspf2d::FO_MAX_DESTS; d++) {
        const uint32_t hop = (tail->dests[d] >> dspf2d::FO_HOP_SHIFT) & dspf2d::FO_HOP_MASK;
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
        at += mc_chunk_len(c, origin, dir_idx, mc_region_hop(i + 1u), link);
    }
    ASSERT(at <= ct.fwd_pages_per_stream);
    return at;
}

uint32_t slot_addr(uint32_t slot) { return ct.ring_addr + slot * ct.slot_stride(); }

volatile tt_l1_ptr dspf2d::FwdMetadata* slot_tail(uint32_t slot) {
    return reinterpret_cast<volatile tt_l1_ptr dspf2d::FwdMetadata*>(slot_addr(slot) + ct.token_size_bytes);
}

// Stage one delivery for the sender to issue out of `slot`: the two page addresses and the metadata
// words it sends alongside the token. Staged here because only the reader holds the output accessors;
// issued by the sender because its NoC port has the headroom this one lacks.
//
// One function for deliveries landing here and for deliveries landing on the chip across the cable.
// Every output buffer is interleaved DRAM with a base uniform across the mesh, so a page index names
// the same place on the neighbour as it does here -- which is how the unicast tail's final addresses
// have always travelled. Which kind a record is comes from its position in the list, not its content.
template <typename OutAcc, typename MetaAcc>
void stage_delivery(
    uint32_t slot,
    uint32_t at,
    uint32_t packed,
    uint32_t src_chip,
    uint32_t token,
    const OutAcc& out_acc,
    const MetaAcc& meta_acc) {
    const uint32_t page = packed & dspf2d::FO_PAGE_MASK;
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        ct.mc_meta_addr + (slot * dspf2d::FO_MAX_DESTS + at) * dspf2d::MC_META_SLOT_BYTES);
    meta[0] = src_chip;
    meta[1] = token;
    meta[2] = packed >> dspf2d::FO_SLOT_SHIFT;
    meta[3] = 0;
    volatile tt_l1_ptr dspf2d::FanoutDelivery* d =
        reinterpret_cast<volatile tt_l1_ptr dspf2d::FanoutDelivery*>(
            ct.mc_delivery_addr + slot * dspf2d::FO_MAX_DESTS * sizeof(dspf2d::FanoutDelivery)) +
        at;
    d->payload_addr = out_acc.get_noc_addr(page);
    d->meta_addr = meta_acc.get_noc_addr(page);
    d->meta_src = (uint32_t)meta;
    d->pad = 0;
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
            const uint32_t bucket = c.bucket_start[b];
            const uint32_t n = c.bucket_start[b + 1u] - bucket;
            const uint32_t from = slice_begin(n, split_idx, split_count);
            const uint32_t to = slice_begin(n, split_idx + 1, split_count);
            const uint32_t out_base = direct ? 0 : c.out_start[a * ct.experts_per_chip + j];
            for (uint32_t i = from; i < to; i++) {
                const uint32_t at = (bucket + i) * ENTRY_WORDS;
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
            const uint32_t bucket = c.bucket_start[b];
            const uint32_t n = c.bucket_start[b + 1u] - bucket;
            const uint32_t from = slice_begin(n, ct.stream, 2 * ct.num_links);
            const uint32_t to = slice_begin(n, ct.stream + 1, 2 * ct.num_links);
            for (uint32_t i = from; i < to; i++) {
                const uint32_t at = (bucket + i) * ENTRY_WORDS;
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
// The chip one hop short of the farthest destination does not pass it on: it writes those last pages
// into the next chip's output pages itself, so the page is never landed in a region only to be read
// straight back out. At a relay that substitution is exclusive -- a page either ends next door or
// travels on -- so only an origin slot ever puts deliveries AND a forward on the same cable.

// This chip's own tokens, one entry per (token, direction), narrowed to this link's share. ONE slot
// and one DRAM read per entry: the neighbour's copies and the copy that travels further are the same
// bytes, and the sender can issue all of them out of the slot it is already holding.
//
// This is the exception to the rule mc_region_hop states. A relay delivers a hop-(j + 1) destination
// only when the page ends there, because otherwise the payload would cross that cable twice. The
// origin delivers its hop-1 destinations unconditionally and forwards as well, paying exactly that
// second crossing -- because it is holding the token already, and because it is what the unicast path
// does, so the two modes put the same bytes on link 0 -> 1. Sending them as a count in the forwarded
// tail instead would make the neighbour read the page back out to write them, which is the cost this
// whole mode exists to avoid.
//
// The share is by farthest-hop class, which is what makes a link's count at every later hop derivable
// from the same reach table -- see mc_link_of.
template <typename InAcc, typename OutAcc, typename MetaAcc, typename FwdAcc>
void mc_own_phase(
    const Control& c,
    Ring& ring,
    const InAcc& in_acc,
    const OutAcc& out_acc,
    const MetaAcc& meta_acc,
    const FwdAcc& fwd_acc,
    uint32_t my_region,
    uint32_t dir_idx,
    uint32_t link) {
    const uint32_t m = ct.extent / 2u;
    const uint32_t stride = dspf2d::fo_entry_words(ct.topk);
    const uint32_t len = mc_chunk_len(c, ct.my_row, dir_idx, mc_region_hop(1u), link);
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
        volatile tt_l1_ptr uint32_t* ent = c.mc_entries + (dir_idx * ct.seq_len + i) * stride;
        const uint32_t far = ent[2];
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
        // flight: a chip j hops along takes the destinations with hop == j, delivers those at hop
        // j + 1 if the page ends there, and otherwise forwards it untouched. The hop-1 destinations
        // are dropped from the tail rather than carried -- this slot delivers them, and leaving them
        // in would make the neighbour write them a second time.
        volatile tt_l1_ptr dspf2d::FanoutMetadata* tail = slot_mc_tail(slot);
        tail->src_chip = ct.linearized_coord;
        tail->token = token;
        uint32_t carried = 0;  // destinations left in the tail for chips further on
        uint32_t remote = 0;   // destinations this slot writes into the neighbour itself
        for (uint32_t d = 0; d < n_dests; d++) {
            const uint32_t packed = ent[3 + d];
            if (((packed >> dspf2d::FO_HOP_SHIFT) & dspf2d::FO_HOP_MASK) >= mc_region_hop(1u)) {
                tail->dests[carried++] = packed;
                continue;
            }
            // The neighbour's pages go straight to their final addresses, exactly as the unicast path
            // sends them: one fabric write for the token and one for its metadata, costing the
            // receiving chip nothing.
            stage_delivery(slot, remote++, packed, ct.linearized_coord, token, out_acc, meta_acc);
        }
        for (uint32_t d = carried; d < dspf2d::FO_MAX_DESTS; d++) {
            tail->dests[d] = 0u;
        }
        // Nothing lands on this chip -- the local phase placed those -- so the remote records start at
        // zero, which is the base the sender re-derives from this count.
        tail->local_count = 0;
        tail->remote_count = remote;
        if (mc_forwards(far, 0u)) {
            // The last page of a chunk forces the downstream bump, which is the boundary that reader
            // switches on: leave it uncounted and the whole axis waits.
            tail->cmd = (q + 1 == len) ? dspf2d::CMD_FORWARD_END : dspf2d::CMD_FORWARD;
            tail->this_addr = fwd_acc.get_noc_addr(my_region + c.out_start[0] + q);
            q++;
        } else {
            // Every destination this token had was the neighbour's, so nothing enters a region. The
            // slot still travels the ring in order, carrying a command that sends only the deliveries.
            tail->cmd = dspf2d::CMD_NO_FORWARD;
        }
        ring.mark_ready();
    }
    ASSERT(q == len);
}

// Arrivals. Chunk j came from the origin j hops upstream, so its destinations at hop j are THIS chip:
// they are written locally, out of the slot the page was read into. A page whose farthest destination
// is hop j + 1 ends across the cable, so this stream writes those pages into the neighbour's output
// pages itself rather than handing it a page to land and read straight back out. Only then -- doing it
// whenever the page happens to have a hop-(j + 1) destination would put the payload on that cable a
// second time whenever the page also travels on. Anything farther is forwarded and the neighbour
// consumes its own.
//
// The read count and the forward count are different numbers -- reach[j + 1] against reach[j + 2] --
// and that is the point of the mode rather than a rounding artefact. Both sides derive their own,
// order is preserved, and a page carries its own destinations, so a dense region still lines up.
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
        const uint32_t len = mc_chunk_len(c, origin, dir_idx, mc_region_hop(j), link);
        // A page is forwarded only if something is left beyond the NEXT chip, so the count is the read
        // count one hop further on -- which is exactly what that chip sizes its own chunk at. Nothing
        // travels past half the ring, so this telescopes to zero at the last two chunks.
        const uint32_t fwd_len = mc_chunk_len(c, origin, dir_idx, mc_region_hop(j + 1u), link);
        const uint32_t in_base = c.in_start[j - 1u];
        const uint32_t out_base = (j < m) ? c.out_start[j] : 0u;
        uint32_t q = 0;
        // Batched exactly as the unicast relay is: one read in flight per stream is a DRAM round trip
        // per page. The consume writes stage their metadata per (page-in-batch, destination), so a
        // batch's writes can overlap instead of each page flushing on its own.
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
            noc_async_read_barrier();  // the destination lists decide the next hop, so they must be here
            // The reads landed behind the data cache, and these slots carried different pages a few
            // iterations ago: without this, a tail can still be that page's.
            invalidate_l1_cache();

            for (uint32_t i = 0; i < n; i++) {
                volatile tt_l1_ptr dspf2d::FanoutMetadata* tail = slot_mc_tail(slots[i]);
                const uint32_t src_chip = tail->src_chip;
                const uint32_t token = tail->token;
                const uint32_t far = mc_tail_far(tail);
                // Every page in chunk j has a destination past this chip, or the chunk both sides size
                // from the reach table would not have counted it.
                ASSERT(far > j);
                uint32_t local = 0;
                for (uint32_t d = 0; d < dspf2d::FO_MAX_DESTS; d++) {
                    const uint32_t packed = tail->dests[d];
                    const uint32_t hop = (packed >> dspf2d::FO_HOP_SHIFT) & dspf2d::FO_HOP_MASK;
                    // Hop 0 is an unused slot, and a hop below j was consumed by a chip behind us: the
                    // page is never rewritten in flight, so both are still here and neither is ours.
                    if (hop != j) {
                        continue;
                    }
                    // Staged for the sender rather than written here: every destination needs its own
                    // record and metadata words because several are in flight from one slot at once.
                    stage_delivery(slots[i], local, packed, src_chip, token, out_acc, meta_acc);
                    local++;
                }
                uint32_t remote = 0;
                if (mc_delivers_ahead(far, j)) {
                    // The page ends across the cable, so its last destinations are written into the
                    // neighbour's output pages from here. They are staged AFTER the local ones, which
                    // is the only thing telling the sender a fabric send from a NoC write.
                    for (uint32_t d = 0; d < dspf2d::FO_MAX_DESTS; d++) {
                        const uint32_t packed = tail->dests[d];
                        const uint32_t hop = (packed >> dspf2d::FO_HOP_SHIFT) & dspf2d::FO_HOP_MASK;
                        if (hop != j + 1u) {
                            continue;
                        }
                        stage_delivery(slots[i], local + remote, packed, src_chip, token, out_acc, meta_acc);
                        remote++;
                    }
                }
                tail->local_count = local;
                tail->remote_count = remote;
                if (mc_forwards(far, j)) {
                    tail->cmd = (q + 1 == fwd_len) ? dspf2d::CMD_FORWARD_END : dspf2d::CMD_FORWARD;
                    tail->this_addr = fwd_acc.get_noc_addr(my_region + out_base + q);
                    q++;
                } else {
                    // Nothing left past the neighbour. The slot still travels the ring in order, so it
                    // carries a command that puts only its deliveries on the cable.
                    tail->cmd = dspf2d::CMD_NO_FORWARD;
                }
            }
            // No writes leave these slots from this RISC any more; the sender issues the deliveries and
            // its own batch flush covers them before it frees a slot.
            for (uint32_t i = 0; i < n; i++) {
                ring.mark_ready();
            }
            p += n;
        }
        ASSERT(q == fwd_len);
    }
}

}  // namespace

void kernel_main() {
    const Control c = carve_control();
    ASSERT(c.end <= ct.filled_addr);  // the control region must not run into the semaphores

    {
        DeviceZoneScopedN("dspf2d_prologue");
        read_control_tables(c);
        read_indices(c);
        build_expert_slots(c);
        size_buckets(c);
        build_routing_index(c);
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
    if (ct.fanout) {
        // Clockwise is direction 0 on both sides -- mc_dir_of resolves a tie at exactly half the ring
        // the same way the reach table was built. Differ in one place and the lengths silently
        // disagree.
        const uint32_t dir_idx = ct.stream % 2u;
        const uint32_t link = ct.stream / 2u;
        const int32_t travel = (dir_idx == 0u) ? 1 : -1;
        {
            DeviceZoneScopedN("dspf2d_mc_build");
            mc_chunk_starts(c, dir_idx, link, travel, /*outgoing=*/false, c.in_start);
            mc_chunk_starts(c, dir_idx, link, travel, /*outgoing=*/true, c.out_start);
        }
        wait_for_untilize();  // the own phase is the first thing that reads a token here
        {
            DeviceZoneScopedN("dspf2d_mc_own");
            mc_own_phase(c, ring, in_acc, out_acc, meta_acc, fwd_acc, my_region, dir_idx, link);
            ring.flush_publish();
        }
        {
            DeviceZoneScopedN("dspf2d_mc_relay");
            mc_relay_phase(c, ring, out_acc, meta_acc, fwd_acc, my_region, dir_idx, link, travel);
            ring.flush_publish();
        }
        // Last, as on the unicast path: these pages never leave the chip, so anything ahead of them in
        // this order is a chip downstream waiting. The multicast entries have their own control block,
        // so the bucket index this reads is still the one the prologue built.
        {
            DeviceZoneScopedN("dspf2d_mc_local");
            local_phase(c, ring, in_acc, out_acc, meta_acc);
        }
    } else {
        {
            DeviceZoneScopedN("dspf2d_starts");
            chunk_starts(c, ct.in_chunks_base, c.in_start);
            chunk_starts(c, ct.out_chunks_base, c.out_start);
        }
        wait_for_untilize();  // the own phase is the first thing that reads a token here
        {
            DeviceZoneScopedN("dspf2d_own");
            own_phase(c, ring, in_acc, out_acc, meta_acc, fwd_acc, my_region);
            ring.flush_publish();
        }
        {
            DeviceZoneScopedN("dspf2d_relay");
            relay_phase(c, ring, fwd_acc, my_region, nbr_row);
            ring.flush_publish();
        }
        {
            DeviceZoneScopedN("dspf2d_local");
            local_phase(c, ring, in_acc, out_acc, meta_acc);
        }
    }

    {
        DeviceZoneScopedN("dspf2d_end");
        const uint32_t end_slot = ring.claim_slot();
        slot_tail(end_slot)->cmd = dspf2d::CMD_END;
        ring.mark_ready();
        ring.flush_publish();

        noc_async_atomic_barrier();
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.fwd_sem_addr), 0);
        if constexpr (ct.untilize_stripes > 0) {
            // Safe here and only here: the wait above this launch's first token read proved all
            // untilize_stripes increments had arrived, so no writer is still bumping this counter.
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.untilize_sem_addr), 0);
        }
    }
}
