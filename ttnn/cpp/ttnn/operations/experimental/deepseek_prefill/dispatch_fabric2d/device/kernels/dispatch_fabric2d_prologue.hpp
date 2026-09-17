// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The prologue of a stream core: the routing index, built by its four RISCs at once. Included by the
// reader (a dataflow kernel) and by the compute kernel that runs on the three TRISCs, so nothing here
// touches the NoC; everything is L1 loads and stores and program semaphores.
//
// Why it splits. Production `dispatch` assigns a token's output page by advancing one counter per
// expert as it walks the picks in token order, dropping a pick whose counter has passed the expert's
// capacity while still advancing the counter. Replaying that walk exactly is what keeps the output
// byte-identical to production, and every later page depends on every earlier one -- so the walk
// cannot be shortcut, but it can be composed: a lane holding a contiguous slice of the tokens knows
// each pick's position within its bucket up to a per-bucket offset, and that offset is the count of
// picks the earlier slices routed to the same bucket. One count pass, one exchange of bucket_slots()
// counts per lane through L1, one fill pass. Every lane writes disjoint, token-ordered runs, and
// together they are exactly what one sequential walk over all tokens would write, so the phases that
// consume the index need not know it was built in slices.
//
// Under fan-out the per-token entries are slice-contiguous rather than packed: a token yields at most
// one entry per direction, so lane w owns the entry positions of its own tokens and the consumer
// walks the four runs in order (mc_run). Packing them would need a second exchange, because whether a
// token has any surviving remote pick is only known once its pages are.
//
// Why here and not upstream. The index is a pure function of (indices, dispatch table, offsets,
// capacity), the inputs the routing-setup ops already hold, and an op there could emit it once per
// chip for the stream cores to DMA. That costs a launch under unicast, a per-chip DRAM output, and
// a second op that has to agree byte for byte with production's allocator. Replaying it here costs
// nothing outside this op and stays inside the same trace; the four-lane split brings it under the
// untilize pool, which is the point at which it stops being the exposed part of the launch.

#include <cstdint>
#include "api/debug/assert.h"
#include "api/debug/waypoint.h"
#include "core_config.h"
#include "hostdev/dev_msgs.h"
#include "noc/noc_parameters.h"
#include "dataflow/dispatch_fabric2d_reader_ct_args.hpp"

namespace dspf2d::prologue {

// The reader's compile-time arguments, which the compute kernel is built from verbatim so that every
// lane carves the same control region from the same constants.
inline constexpr dspf2d::ReaderCtArgs ct{};

constexpr uint32_t LANES = PROLOGUE_LANES;

// The stream core's L1 working set, carved out of the control region in one fixed order from the
// compile-time arguments alone: every chip lays it out identically, and so do the four RISCs of one
// core, each of which carves it for itself. `indices` comes first because it is the only part read
// straight from DRAM per token, and its records must stay 64-byte aligned.
struct Control {
    volatile tt_l1_ptr uint16_t* indices;       // seq_len records, each padded to indices_pad_stride
    volatile tt_l1_ptr uint32_t* offsets;       // extent x num_routed_experts: every source chip's row
    volatile tt_l1_ptr uint32_t* counts;        // num_routed_experts, summed over source chips
    volatile tt_l1_ptr uint32_t* region;        // num_routed_experts
    volatile tt_l1_ptr int32_t* table;          // num_routed_experts (+1 sentinel), expert -> chip in group
    volatile tt_l1_ptr uint32_t* expert_slot;   // the same domain, packed as ES_* for the routing pass
    volatile tt_l1_ptr uint32_t* first_page;    // extent x experts_per_chip: each bucket's first output page
    volatile tt_l1_ptr uint32_t* chip_experts;  // extent x experts_per_chip, ascending global expert id
    volatile tt_l1_ptr uint32_t* row_fill;      // extent, while the chip -> experts inverse is built
    volatile tt_l1_ptr uint32_t* bucket_start;  // extent x experts_per_chip + 1, exclusive prefix sums with a total
    volatile tt_l1_ptr uint32_t* entries;       // 3 words per surviving (token, top-k slot)
    volatile tt_l1_ptr uint32_t* mc_entries;    // fanout: one entry per (token, direction), slice-contiguous
    volatile tt_l1_ptr uint32_t* mc_count;      // fanout: entries per direction over all lanes
    // fanout: one reach row per (origin, direction), each padded to 64 bytes. An address rather than a
    // pointer because the pad makes the stride wider than the row.
    uint32_t reach;
    volatile tt_l1_ptr uint32_t* padding;    // [real_token_count, pad_side], when one was supplied
    volatile tt_l1_ptr uint32_t* in_start;   // page offset of each chunk this stream reads
    volatile tt_l1_ptr uint32_t* out_start;  // page offset of each chunk it writes downstream
    volatile tt_l1_ptr uint32_t* lane;       // PROLOGUE_LANES x prologue_lane_words
    uint32_t end;
};

// The geometry the control region is sized from. The host builds the same struct and reserves
// control_region_bytes of it; carve_control below walks the same block list in the same order, so the
// two cannot drift apart.
inline dspf2d::ControlGeometry control_geometry() {
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

inline Control carve_control() {
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
    c.first_page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbAlloc));
    c.chip_experts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbChipExperts));
    c.row_fill = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbRowFill));
    c.bucket_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbBucketStart));
    c.entries = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbEntries));
    c.mc_entries = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbMcEntries));
    c.mc_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbMcCount));
    c.reach = take(dspf2d::kCbReach);
    c.padding = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbPadding));
    c.in_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbInStart));
    c.out_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbOutStart));
    c.lane = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbLane));
    c.end = a;
    // The host reserved exactly this, from the same list. A carve that outgrew the reservation would
    // run into the global semaphores, so say so here rather than corrupting them.
    ASSERT(c.end - ct.control_addr == dspf2d::control_region_bytes(g));
    return c;
}

// Tokens the routing pass walks. A padding_config shortens it to the real ones, exactly as the
// production op shortens its batch loop, and for the same reason: with right padding the real tokens
// hold the low indices, so the allocator reaches all of them before the first padded one.
//
// This cannot desynchronise the ring. Every chunk length comes from the offsets table, never from how
// far this loop ran, and supplying the config asserts that padded tokens are sentinel-marked -- their
// picks resolve to ES_NOT_HERE and contribute no page. Skipping them is skipping no-ops.
inline uint32_t routed_token_count(const Control& c) {
    if constexpr (ct.has_padding_config) {
        const uint32_t real = c.padding[0];
        const uint32_t pad_side = c.padding[1];
        if (pad_side == 0u && real < ct.seq_len) {
            return real;
        }
    }
    return ct.seq_len;
}

// A share of a run, by fraction rather than count: token counts are data-dependent and unknown to the
// host, and integer arithmetic makes consecutive slices meet exactly whatever the count turns out to be.
constexpr uint32_t slice_begin(uint32_t n, uint32_t idx, uint32_t count) { return (n * idx) / count; }

// Lane w's slice of the tokens: contiguous, in token order, the slices tiling [0, tokens).
constexpr uint32_t slice_lo(uint32_t tokens, uint32_t lane) { return slice_begin(tokens, lane, LANES); }
constexpr uint32_t slice_hi(uint32_t tokens, uint32_t lane) { return slice_begin(tokens, lane + 1u, LANES); }

constexpr uint32_t bucket_slots() { return ct.extent * ct.experts_per_chip; }

// How many of `routed` picks at a bucket whose first page is `first_page` survive capacity: pages are
// handed out in order and dropped once past it, so the survivors are the first `room` picks. The one
// rule that makes the replay byte-identical to production, spelled once.
constexpr uint32_t survivors_of(uint32_t first_page, uint32_t routed) {
    const uint32_t room = ct.max_dispatch_buf_tokens > first_page ? ct.max_dispatch_buf_tokens - first_page : 0u;
    return routed < room ? routed : room;
}

// A block this RISC only reads, and that nothing writes after the handoff that made it visible: dropping
// volatile lets the compiler keep loads in registers instead of a round trip to L1 for every reread.
template <typename T>
inline const T* frozen(volatile tt_l1_ptr T* p) {
    return reinterpret_cast<const T*>(reinterpret_cast<uint32_t>(p));
}

// One lane's scratch in kCbLane. Field order and prologue_lane_words are one layout; the assert
// below is what ties them.
struct Lane {
    volatile tt_l1_ptr uint32_t* cnt;         // routed picks per bucket in my slice, survivors or not
    volatile tt_l1_ptr uint32_t* next_page;   // running page counter for the fill pass
    volatile tt_l1_ptr uint32_t* next_entry;  // entry cursor per bucket for the fill pass
    volatile tt_l1_ptr uint32_t* mc_n;        // fanout: entries I wrote, per direction
};
static_assert(dspf2d::prologue_lane_words(1u) == 3u + 2u, "Lane has three per-bucket arrays and two direction words");

inline Lane lane_view(const Control& c, uint32_t lane) {
    const uint32_t n = bucket_slots();
    volatile tt_l1_ptr uint32_t* base = c.lane + lane * dspf2d::prologue_lane_words(n);
    return Lane{base, base + n, base + 2u * n, base + 3u * n};
}

// The fan-out entry run one lane wrote for one direction: where it starts and how many it holds. The
// consumer walks lanes 0..LANES-1 in order, which is the single token-ordered list.
inline void mc_run(const Control& c, uint32_t dir, uint32_t lane, uint32_t tokens, uint32_t* base, uint32_t* n) {
    *base = dir * ct.seq_len + slice_lo(tokens, lane);
    *n = lane_view(c, lane).mc_n[dir];
}

// Program semaphores: the runtime writes their initial value on every launch, so no lane can take a
// stale word for a signal. They sit in the launch's kernel-config region, whose base the dataflow
// firmware resolves into sem_l1_base. The TRISC firmware on this architecture does not carry that
// symbol, so a compute lane reads the same launch message the firmware did; BRISC advances the read
// pointer only after every RISC of the core has finished, so it names this launch for the whole run.
inline uint32_t semaphore_base() {
#if defined(COMPILE_FOR_TRISC)
    const uint32_t rd = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    const volatile tt_l1_ptr kernel_config_msg_t* cfg = &GET_MAILBOX_ADDRESS_DEV(launch[rd])->kernel_config;
    return cfg->kernel_config_base[ProgrammableCoreType::TENSIX] + cfg->sem_offset[ProgrammableCoreType::TENSIX];
#else
    return reinterpret_cast<uint32_t>(sem_l1_base[static_cast<int>(ProgrammableCoreType::TENSIX)]);
#endif
}

// Same word on every RISC of the core; the L1 alignment is the stride the runtime lays them out at.
inline volatile tt_l1_ptr uint32_t* semaphore(uint32_t id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_base() + id * L1_ALIGNMENT);
}

// A fence before the store orders this RISC's data stores ahead of the signal; the one after keeps the
// signal from being reordered behind what follows.
inline void fence() { asm volatile("fence" ::: "memory"); }

inline void signal(uint32_t id, uint32_t value) {
    fence();
    *semaphore(id) = value;
    fence();
}

// Invalidating before each poll keeps a data cache, if the runtime turned one on, from pinning a stale
// line. The waypoints are the only hang-diagnosis channel this op has: kernel asserts are compiled out.
inline void wait_at_least(uint32_t id, uint32_t value) {
    WAYPOINT("PLW");
    volatile tt_l1_ptr uint32_t* sem = semaphore(id);
    while (true) {
        invalidate_l1_cache();
        if (*sem >= value) {
            break;
        }
    }
    fence();
    WAYPOINT("PLD");
}

inline void wait_all_lanes(uint32_t value) {
    for (uint32_t lane = 0; lane < LANES; lane++) {
        wait_at_least(dspf2d::prologue_lane_sem(lane), value);
    }
}

// Pass 1: how many picks my slice routes to each bucket, counting the ones capacity will drop as well,
// because the allocator counter they advance is what positions everything after them. Under fan-out
// only the local picks reach a bucket, but the counter is per expert regardless of who consumes it.
inline void count_pass(const Control& c, const Lane& me, uint32_t t0, uint32_t t1) {
    const uint32_t n = bucket_slots();
    for (uint32_t b = 0; b < n; b++) {
        me.cnt[b] = 0u;
    }
    const uint32_t* es = frozen(c.expert_slot);
    uint32_t idx_addr = reinterpret_cast<uint32_t>(c.indices) + t0 * ct.indices_pad_stride;
    for (uint32_t t = t0; t < t1; t++, idx_addr += ct.indices_pad_stride) {
        const uint16_t* idx = reinterpret_cast<const uint16_t*>(idx_addr);
        static_assert(ct.topk <= 8, "the unroll count is the top-k bound");
#pragma GCC unroll 8
        for (uint32_t k = 0; k < ct.topk; k++) {
            const uint32_t w = es[idx[k]];
            if (w == dspf2d::ES_NOT_HERE) {
                continue;
            }
            const uint32_t slot = w & dspf2d::ES_SLOT_MASK;
            // A word past the table (an index the host never validated) could name any slot; a
            // counter outside this lane's block is somebody else's state.
            if (slot >= n) {
                continue;
            }
            me.cnt[slot] = me.cnt[slot] + 1u;
        }
    }
}

// Between the passes: where my slice's pages and entries start in every bucket, from the counts of the
// slices before mine.
inline void place_slice(const Control& c, const Lane& me, uint32_t lane) {
    const uint32_t n = bucket_slots();
    for (uint32_t b = 0; b < n; b++) {
        uint32_t before = 0;
        for (uint32_t v = 0; v < lane; v++) {
            before += lane_view(c, v).cnt[b];
        }
        const uint32_t first_page = c.first_page[b];
        me.next_page[b] = first_page + before;
        me.next_entry[b] = c.bucket_start[b] + survivors_of(first_page, before);
    }
    me.mc_n[0] = 0u;
    me.mc_n[1] = 0u;
}

// Pass 2: the walk over my slice, from the positions place_slice gave me. The same per-pick rule as
// production, with the cursors per lane.
inline void fill_pass(const Control& c, const Lane& me, uint32_t t0, uint32_t t1) {
    const uint32_t cap = ct.max_dispatch_buf_tokens;
    [[maybe_unused]] const uint32_t mc_stride = dspf2d::fo_entry_words(ct.topk);
    const uint32_t* es = frozen(c.expert_slot);
    uint32_t idx_addr = reinterpret_cast<uint32_t>(c.indices) + t0 * ct.indices_pad_stride;
    for (uint32_t t = t0; t < t1; t++, idx_addr += ct.indices_pad_stride) {
        const uint16_t* idx = reinterpret_cast<const uint16_t*>(idx_addr);
        [[maybe_unused]] uint32_t n_dir[2] = {0, 0};
        [[maybe_unused]] uint32_t far_dir[2] = {0, 0};
        [[maybe_unused]] uint32_t packed[2][dspf2d::FO_MAX_DESTS];
#pragma GCC unroll 8
        for (uint32_t k = 0; k < ct.topk; k++) {
            const uint32_t w = es[idx[k]];
            if (w == dspf2d::ES_NOT_HERE) {
                continue;  // the expert is not in this dispatch group
            }
            const uint32_t slot = w & dspf2d::ES_SLOT_MASK;
            if (slot >= bucket_slots()) {
                continue;  // as in count_pass: never index another lane's block
            }
            const uint32_t page = me.next_page[slot];
            me.next_page[slot] = page + 1u;
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
            const uint32_t at = me.next_entry[slot];
            // The bucket was sized from the offsets table, which the same routing produced. A table
            // that disagrees would otherwise write over the next bucket, and the ASSERT that reports
            // the disagreement is compiled out on this hardware.
            if (at >= c.bucket_start[slot + 1u]) {
                continue;
            }
            me.next_entry[slot] = at + 1u;
            volatile tt_l1_ptr uint32_t* ent = c.entries + at * dspf2d::entry_words();
            ent[0] = t;
            ent[1] = page;
            ent[2] = k;
        }
        if constexpr (ct.fanout) {
            for (uint32_t d = 0; d < 2u; d++) {
                if (n_dir[d] == 0) {
                    continue;
                }
                // Slice-contiguous: token t's entry can only sit at or after position t0 of its
                // direction's run, and the run is walked by the count this lane reports.
                volatile tt_l1_ptr uint32_t* ent = c.mc_entries + (d * ct.seq_len + t0 + me.mc_n[d]) * mc_stride;
                me.mc_n[d] = me.mc_n[d] + 1u;
                ent[0] = t;
                ent[1] = n_dir[d];
                ent[2] = far_dir[d];
                for (uint32_t i = 0; i < n_dir[d]; i++) {
                    ent[3 + i] = packed[d][i];
                }
            }
        }
    }
}

// The whole of one lane's share: wait for the tables, count, exchange, fill, report.
inline void run_lane(const Control& c, uint32_t lane) {
    wait_at_least(dspf2d::kSemTablesReady, 1u);
    const uint32_t tokens = routed_token_count(c);
    const Lane me = lane_view(c, lane);
    count_pass(c, me, slice_lo(tokens, lane), slice_hi(tokens, lane));
    signal(dspf2d::prologue_lane_sem(lane), dspf2d::kLaneCounted);
    wait_all_lanes(dspf2d::kLaneCounted);
    place_slice(c, me, lane);
    fill_pass(c, me, slice_lo(tokens, lane), slice_hi(tokens, lane));
    signal(dspf2d::prologue_lane_sem(lane), dspf2d::kLaneFilled);
}

// The reader's side, in the one order that is correct: the tables the lanes read with plain loads
// are complete before the signal that releases them, and this RISC runs its own lane in between.
// The caller waits for the other lanes' fills (wait_all_lanes(kLaneFilled)) before it reads anything
// they wrote.
template <typename BuildTables>
inline void reader_prologue(const Control& c, BuildTables&& build_tables) {
    build_tables();
    signal(dspf2d::kSemTablesReady, 1u);
    run_lane(c, dspf2d::kLaneReader);
}

}  // namespace dspf2d::prologue
