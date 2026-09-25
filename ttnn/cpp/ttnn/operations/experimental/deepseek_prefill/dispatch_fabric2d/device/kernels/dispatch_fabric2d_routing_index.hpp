// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The routing index of a stream core: the routing index, built by its four RISCs at once. Included by the
// reader (a dataflow kernel) and by the compute kernel that runs on the three TRISCs, so nothing here
// touches the NoC; everything is L1 loads and stores and program semaphores.
//
// Why it splits. Production `dispatch` assigns a token's output page by advancing one counter per
// expert as it walks the picks in token order, dropping a pick whose counter has passed the expert's
// capacity while still advancing the counter. Replaying that walk exactly is what keeps the output
// byte-identical to production, and every later page depends on every earlier one -- so the walk
// cannot be shortcut, but it can be composed: a RISC holding a contiguous slice of the tokens knows
// each pick's position within its bucket up to a per-bucket offset, and that offset is the count of
// picks the earlier slices routed to the same bucket. One count pass, one exchange of num_buckets()
// counts per RISC through L1, one fill pass. Every RISC writes disjoint, token-ordered runs, and
// together they are exactly what one sequential walk over all tokens would write, so the phases that
// consume the index need not know it was built in slices.
//
// Why here and not upstream. The index is a pure function of (indices, dispatch table, offsets,
// capacity), the inputs the routing-setup ops already hold, so an op there could emit it once per
// chip for the stream cores to DMA. That costs a launch, a per-chip DRAM output, and a second op that
// has to agree byte for byte with production's allocator. Replaying it here costs nothing outside
// this op and stays inside the same trace; the four-RISC split brings it under the untilize pool,
// which is the point at which it stops being the exposed part of the launch.

#include <cstdint>
#include "api/debug/assert.h"
#include "api/debug/waypoint.h"
#include "core_config.h"
#include "hostdev/dev_msgs.h"
#include "noc/noc_parameters.h"
#include "dataflow/dispatch_fabric2d_reader_ct_args.hpp"

namespace dspf2d::routing_index {

// The reader's compile-time arguments, which the compute kernel is built from verbatim so that every
// RISC lays out the same scratch from the same constants.
inline constexpr dspf2d::ReaderCtArgs ct{};

constexpr uint32_t RISCS = INDEX_RISCS;

// The stream core's L1 working set, laid out in the scratch in one fixed order from the
// compile-time arguments alone: every chip lays it out identically, and so do the four RISCs of one
// core, each of which lays it out for itself. `indices` comes first because it is the only part read
// straight from DRAM per token, and its records must stay 64-byte aligned.
struct Control {
    volatile tt_l1_ptr uint16_t* indices;         // seq_len records, each padded to indices_pad_stride
    volatile tt_l1_ptr uint32_t* offsets;         // extent x num_routed_experts: every source chip's row
    volatile tt_l1_ptr uint32_t* counts;          // num_routed_experts, summed over source chips
    volatile tt_l1_ptr uint32_t* region_offsets;  // num_routed_experts
    volatile tt_l1_ptr int32_t* table;            // num_routed_experts (+1 sentinel), expert -> chip in group
    volatile tt_l1_ptr uint32_t* expert_bucket;   // the same domain, as a bucket or BUCKET_NOT_HERE
    volatile tt_l1_ptr uint32_t* first_page;      // extent x experts_per_chip: each bucket's first output page
    volatile tt_l1_ptr uint32_t* chip_experts;    // extent x experts_per_chip, ascending global expert id
    volatile tt_l1_ptr uint32_t* row_fill;        // extent, while the chip -> experts inverse is built
    volatile tt_l1_ptr uint32_t* bucket_start;    // extent x experts_per_chip + 1, exclusive prefix sums with a total
    volatile tt_l1_ptr uint32_t* entries;         // 3 words per surviving (token, top-k slot)
    volatile tt_l1_ptr uint32_t* padding;         // [real_token_count, pad_side], when one was supplied
    volatile tt_l1_ptr uint32_t* in_start;        // page offset of each chunk this stream reads
    volatile tt_l1_ptr uint32_t* out_start;       // page offset of each chunk it writes downstream
    volatile tt_l1_ptr uint32_t* risc;            // INDEX_RISCS x index_risc_words
    uint32_t end;
};

// The geometry the scratch is sized from. The host builds the same struct and reserves
// scratch_bytes of it; layout_scratch below walks the same block list in the same order, so the
// two cannot drift apart.
inline dspf2d::ControlGeometry control_geometry() {
    dspf2d::ControlGeometry g;
    g.seq_len = ct.seq_len;
    g.indices_pad_stride = ct.indices_pad_stride;
    g.extent = ct.extent;
    g.num_routed_experts = ct.num_routed_experts;
    g.experts_per_chip = ct.experts_per_chip;
    g.topk = ct.topk;
    g.num_forward = ct.num_forward;
    return g;
}

inline Control layout_scratch() {
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
    c.region_offsets = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbRegionOffsets));
    c.table = reinterpret_cast<volatile tt_l1_ptr int32_t*>(take(dspf2d::kCbTable));
    c.expert_bucket = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbExpertBucket));
    c.first_page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbFirstPage));
    c.chip_experts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbChipExperts));
    c.row_fill = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbRowFill));
    c.bucket_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbBucketStart));
    c.entries = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbEntries));
    c.padding = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbPadding));
    c.in_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbInStart));
    c.out_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbOutStart));
    c.risc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kCbRisc));
    c.end = a;
    // The host reserved exactly this, from the same list. A layout that outgrew the reservation would
    // run into the global semaphores, so say so here rather than corrupting them.
    ASSERT(c.end - ct.control_addr == dspf2d::scratch_bytes(g));
    return c;
}

// Tokens the routing pass walks. A padding_config shortens it to the real ones, exactly as the
// production op shortens its batch loop, and for the same reason: with right padding the real tokens
// hold the low indices, so the allocator reaches all of them before the first padded one.
//
// This cannot desynchronise the ring. Every chunk length comes from the offsets table, never from how
// far this loop ran, and supplying the config asserts that padded tokens are sentinel-marked -- their
// picks resolve to BUCKET_NOT_HERE and contribute no page. Skipping them is skipping no-ops.
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

// RISC w's slice of the tokens: contiguous, in token order, the slices tiling [0, tokens).
constexpr uint32_t slice_lo(uint32_t tokens, uint32_t risc) { return slice_begin(tokens, risc, RISCS); }
constexpr uint32_t slice_hi(uint32_t tokens, uint32_t risc) { return slice_begin(tokens, risc + 1u, RISCS); }

constexpr uint32_t num_buckets() { return ct.extent * ct.experts_per_chip; }

// How many of `routed` picks at a bucket whose first page is `first_page` survive capacity: pages are
// handed out in order and dropped once past it, so the kept are the first `room` picks. The one
// rule that makes the replay byte-identical to production, spelled once.
constexpr uint32_t kept_count(uint32_t first_page, uint32_t routed) {
    const uint32_t room = ct.max_dispatch_buf_tokens > first_page ? ct.max_dispatch_buf_tokens - first_page : 0u;
    return routed < room ? routed : room;
}

// A block this RISC only reads, and that nothing writes after the handoff that made it visible: dropping
// volatile lets the compiler keep loads in registers instead of a round trip to L1 for every reread.
template <typename T>
inline const T* frozen(volatile tt_l1_ptr T* p) {
    return reinterpret_cast<const T*>(reinterpret_cast<uint32_t>(p));
}

// One RISC's scratch in kCbRisc. Field order and index_risc_words are one layout; the assert
// below is what ties them.
struct Risc {
    volatile tt_l1_ptr uint32_t* cnt;         // routed picks per bucket in my slice, kept or not
    volatile tt_l1_ptr uint32_t* next_page;   // running page counter for the fill pass
    volatile tt_l1_ptr uint32_t* next_entry;  // entry cursor per bucket for the fill pass
};
static_assert(dspf2d::index_risc_words(1u) == 3u, "Risc has three per-bucket arrays");

inline Risc risc_view(const Control& c, uint32_t risc) {
    const uint32_t n = num_buckets();
    volatile tt_l1_ptr uint32_t* base = c.risc + risc * dspf2d::index_risc_words(n);
    return Risc{base, base + n, base + 2u * n};
}

// Program semaphores: the runtime writes their initial value on every launch, so no RISC can take a
// stale word for a signal. They sit in the launch's kernel-config region, whose base the dataflow
// firmware resolves into sem_l1_base. The TRISC firmware on this architecture does not carry that
// symbol, so a compute RISC reads the same launch message the firmware did; BRISC advances the read
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

inline void wait_all_riscs(uint32_t value) {
    for (uint32_t risc = 0; risc < RISCS; risc++) {
        wait_at_least(dspf2d::index_risc_sem(risc), value);
    }
}

// Pass 1: how many picks my slice routes to each bucket, counting the ones capacity will drop as well,
// because the allocator counter they advance is what positions everything after them.
inline void count_pass(const Control& c, const Risc& me, uint32_t t0, uint32_t t1) {
    const uint32_t n = num_buckets();
    for (uint32_t b = 0; b < n; b++) {
        me.cnt[b] = 0u;
    }
    const uint32_t* es = frozen(c.expert_bucket);
    uint32_t idx_addr = reinterpret_cast<uint32_t>(c.indices) + t0 * ct.indices_pad_stride;
    for (uint32_t t = t0; t < t1; t++, idx_addr += ct.indices_pad_stride) {
        const uint16_t* idx = reinterpret_cast<const uint16_t*>(idx_addr);
        static_assert(ct.topk <= 8, "the unroll count is the top-k bound");
#pragma GCC unroll 8
        for (uint32_t k = 0; k < ct.topk; k++) {
            const uint32_t bucket = es[idx[k]];
            // A word past the table (an index the host never validated) could name any bucket; a
            // counter outside this RISC's block is somebody else's state.
            if (bucket >= n) {
                continue;  // BUCKET_NOT_HERE, or an expert id the table does not resolve
            }
            me.cnt[bucket] = me.cnt[bucket] + 1u;
        }
    }
}

// Between the passes: where my slice's pages and entries start in every bucket, from the counts of the
// slices before mine.
inline void place_slice(const Control& c, const Risc& me, uint32_t risc) {
    const uint32_t n = num_buckets();
    for (uint32_t b = 0; b < n; b++) {
        uint32_t before = 0;
        for (uint32_t v = 0; v < risc; v++) {
            before += risc_view(c, v).cnt[b];
        }
        const uint32_t first_page = c.first_page[b];
        me.next_page[b] = first_page + before;
        me.next_entry[b] = c.bucket_start[b] + kept_count(first_page, before);
    }
}

// Pass 2: the walk over my slice, from the positions place_slice gave me. The same per-pick rule as
// production, with the cursors per RISC.
inline void fill_pass(const Control& c, const Risc& me, uint32_t t0, uint32_t t1) {
    const uint32_t cap = ct.max_dispatch_buf_tokens;
    const uint32_t n = num_buckets();
    const uint32_t* es = frozen(c.expert_bucket);
    uint32_t idx_addr = reinterpret_cast<uint32_t>(c.indices) + t0 * ct.indices_pad_stride;
    for (uint32_t t = t0; t < t1; t++, idx_addr += ct.indices_pad_stride) {
        const uint16_t* idx = reinterpret_cast<const uint16_t*>(idx_addr);
#pragma GCC unroll 8
        for (uint32_t k = 0; k < ct.topk; k++) {
            const uint32_t bucket = es[idx[k]];
            if (bucket >= n) {
                continue;  // as in count_pass: never index another RISC's block
            }
            const uint32_t page = me.next_page[bucket];
            me.next_page[bucket] = page + 1u;
            if (page >= cap) {
                continue;  // dropped for want of capacity, with the counter already advanced
            }
            const uint32_t at = me.next_entry[bucket];
            // The bucket was sized from the offsets table, which the same routing produced. A table
            // that disagrees would otherwise write over the next bucket, and the ASSERT that reports
            // the disagreement is compiled out on this hardware.
            if (at >= c.bucket_start[bucket + 1u]) {
                continue;
            }
            me.next_entry[bucket] = at + 1u;
            volatile tt_l1_ptr uint32_t* ent = c.entries + at * dspf2d::entry_words();
            ent[0] = t;
            ent[1] = page;
            ent[2] = k;
        }
    }
}

// The whole of one RISC's share: wait for the tables, count, exchange, fill, report.
inline void run_risc(const Control& c, uint32_t risc) {
    wait_at_least(dspf2d::kSemTablesReady, 1u);
    const uint32_t tokens = routed_token_count(c);
    const Risc me = risc_view(c, risc);
    count_pass(c, me, slice_lo(tokens, risc), slice_hi(tokens, risc));
    signal(dspf2d::index_risc_sem(risc), dspf2d::kRiscCounted);
    wait_all_riscs(dspf2d::kRiscCounted);
    place_slice(c, me, risc);
    fill_pass(c, me, slice_lo(tokens, risc), slice_hi(tokens, risc));
    signal(dspf2d::index_risc_sem(risc), dspf2d::kRiscFilled);
}

// The reader's side, in the one order that is correct: the tables the RISCs read with plain loads
// are complete before the signal that releases them, and this RISC runs its own RISC in between.
// The caller waits for the other RISCs' fills (wait_all_riscs(kRiscFilled)) before it reads anything
// they wrote.
template <typename BuildTables>
inline void reader_routing_index(const Control& c, BuildTables&& build_tables) {
    build_tables();
    signal(dspf2d::kSemTablesReady, 1u);
    run_risc(c, dspf2d::kRiscReader);
}

}  // namespace dspf2d::routing_index
