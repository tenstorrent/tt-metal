// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// A stream core's routing index: a counting sort of this chip's (token, topk index) picks into buckets, one
// bucket per expert of each chip on the axis, giving each kept pick its output page. Included by the reader
// and by the compute kernel on the three TRISCs, so it uses only L1 loads, stores and program semaphores.
//
// The four RISCs build it at once, each over a contiguous slice of the tokens. Each RISC counts its slice's
// picks per bucket, the counts are exchanged through L1, and each RISC fills its records starting from the
// sum of the earlier slices' counts. The result equals one sequential walk over the tokens in order.

#include <cstdint>
#include "api/debug/assert.h"
#include "api/debug/waypoint.h"
#include "core_config.h"
#include "hostdev/dev_msgs.h"
#include "noc/noc_parameters.h"
#include "dataflow/dispatch_fabric2d_reader_ct_args.hpp"

namespace dspf2d::routing_index {

// The reader's compile-time arguments. The compute kernel is built from the same ones, so every RISC lays out
// the same scratch.
inline constexpr dspf2d::ReaderCtArgs ct{};

constexpr uint32_t RISCS = INDEX_RISCS;

// The stream core's L1 working set, laid out in scratch in a fixed order from compile-time arguments only, so
// every chip and every RISC computes the same layout.
struct Scratch {
    volatile tt_l1_ptr uint16_t* indices;         // seq_len records, each padded to indices_pad_stride
    volatile tt_l1_ptr uint32_t* offsets;         // extent x num_routed_experts: every source chip's row
    volatile tt_l1_ptr uint32_t* counts;          // num_routed_experts, summed over source chips
    volatile tt_l1_ptr uint32_t* region_offsets;  // num_routed_experts
    volatile tt_l1_ptr int32_t* table;            // num_routed_experts (+1 sentinel), expert -> chip in group
    volatile tt_l1_ptr uint32_t* expert_bucket;   // the same domain, as a bucket or BUCKET_NOT_HERE
    volatile tt_l1_ptr uint32_t* first_page;      // extent x experts_per_chip: each bucket's first output page
    volatile tt_l1_ptr uint32_t* chip_experts;    // extent x experts_per_chip, ascending global expert id
    volatile tt_l1_ptr uint32_t* pos_fill;        // extent, while the chip -> experts inverse is built
    volatile tt_l1_ptr uint32_t* bucket_start;    // extent x experts_per_chip + 1, exclusive prefix sums with a total
    volatile tt_l1_ptr uint32_t* records;         // 3 words per kept (token, topk index)
    volatile tt_l1_ptr uint32_t* padding;         // [real_token_count, pad_side], when one was supplied
    volatile tt_l1_ptr uint32_t* in_start;        // page offset of each chunk this stream reads
    volatile tt_l1_ptr uint32_t* out_start;       // page offset of each chunk it writes downstream
    volatile tt_l1_ptr uint32_t* risc;            // INDEX_RISCS x index_risc_words
    uint32_t end;
};

// The geometry the scratch is sized from. The host builds the same struct to reserve scratch_bytes, and
// layout_scratch walks the same block list in the same order.
inline dspf2d::ScratchGeometry scratch_geometry() {
    dspf2d::ScratchGeometry g;
    g.seq_len = ct.seq_len;
    g.indices_pad_stride = ct.indices_pad_stride;
    g.extent = ct.extent;
    g.num_routed_experts = ct.num_routed_experts;
    g.experts_per_chip = ct.experts_per_chip;
    g.topk = ct.topk;
    g.num_forward = ct.num_forward;
    return g;
}

inline Scratch layout_scratch() {
    const dspf2d::ScratchGeometry g = scratch_geometry();
    uint32_t a = ct.scratch_addr;
    const auto take = [&](uint32_t block) {
        const uint32_t at = a;
        a += dspf2d::scratch_block_bytes(g, block);
        return at;
    };

    Scratch c;
    c.indices = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(take(dspf2d::kBlkIndices));
    c.offsets = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kBlkOffsets));
    c.counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kBlkCounts));
    c.region_offsets = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kBlkRegionOffsets));
    c.table = reinterpret_cast<volatile tt_l1_ptr int32_t*>(take(dspf2d::kBlkTable));
    c.expert_bucket = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kBlkExpertBucket));
    c.first_page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kBlkFirstPage));
    c.chip_experts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kBlkChipExperts));
    c.pos_fill = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kBlkPosFill));
    c.bucket_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kBlkBucketStart));
    c.records = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kBlkRecords));
    c.padding = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kBlkPadding));
    c.in_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kBlkInStart));
    c.out_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kBlkOutStart));
    c.risc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(take(dspf2d::kBlkRisc));
    c.end = a;
    // The host reserved exactly this; anything more would run into the global semaphores.
    ASSERT(c.end - ct.scratch_addr == dspf2d::scratch_bytes(g));
    return c;
}

// Tokens the routing index walks. With right padding the real tokens come first, so the walk stops after
// them. Chunk lengths come from the offsets table, not from this count, and padded tokens are sentinel-marked
// so their picks resolve to BUCKET_NOT_HERE; stopping early changes no page.
inline uint32_t routed_token_count(const Scratch& c) {
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

// A RISC's slice of the tokens: contiguous and in token order; the slices tile [0, tokens).
constexpr uint32_t slice_lo(uint32_t tokens, uint32_t risc) { return slice_begin(tokens, risc, RISCS); }
constexpr uint32_t slice_hi(uint32_t tokens, uint32_t risc) { return slice_begin(tokens, risc + 1u, RISCS); }

constexpr uint32_t num_buckets() { return ct.extent * ct.experts_per_chip; }

// How many of `routed` picks to a bucket starting at `first_page` are kept. Pages are handed out in order and
// picks past max_dispatch_buffer_token_size are dropped, so the kept are the first `room`.
constexpr uint32_t kept_count(uint32_t first_page, uint32_t routed) {
    const uint32_t room =
        ct.max_dispatch_buffer_token_size > first_page ? ct.max_dispatch_buffer_token_size - first_page : 0u;
    return routed < room ? routed : room;
}

// For a block nothing writes after the signal that made it visible. Dropping volatile lets the compiler keep
// its values in registers.
template <typename T>
inline const T* frozen(volatile tt_l1_ptr T* p) {
    return reinterpret_cast<const T*>(reinterpret_cast<uint32_t>(p));
}

// One RISC's scratch in kBlkRisc. Field order and index_risc_words are one layout; the assert
// below is what ties them.
struct Risc {
    volatile tt_l1_ptr uint32_t* cnt;          // routed picks per bucket in my slice, kept or not
    volatile tt_l1_ptr uint32_t* next_page;    // running page counter for the fill pass
    volatile tt_l1_ptr uint32_t* next_record;  // record cursor per bucket for the fill pass
};
static_assert(dspf2d::index_risc_words(1u) == 3u, "Risc has three per-bucket arrays");

inline Risc risc_view(const Scratch& c, uint32_t risc) {
    const uint32_t n = num_buckets();
    volatile tt_l1_ptr uint32_t* base = c.risc + risc * dspf2d::index_risc_words(n);
    return Risc{base, base + n, base + 2u * n};
}

// Program semaphores: the runtime resets them on every launch, so a stale value is never read as a signal.
// Dataflow firmware exposes their base as sem_l1_base; TRISC firmware does not, so a compute RISC reads it
// from the launch message. BRISC advances launch_msg_rd_ptr only after every RISC of the core finishes, so
// the pointer names this launch throughout.
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

// The fences keep this RISC's earlier stores ahead of the signal and later accesses behind it.
inline void fence() { asm volatile("fence" ::: "memory"); }

inline void signal(uint32_t id, uint32_t value) {
    fence();
    *semaphore(id) = value;
    fence();
}

// Invalidate before each poll in case the data cache is enabled. With the watcher enabled, a RISC stuck in
// this wait shows waypoint RIW (waiting); RID means it got past.
inline void wait_at_least(uint32_t id, uint32_t value) {
    WAYPOINT("RIW");
    volatile tt_l1_ptr uint32_t* sem = semaphore(id);
    while (true) {
        invalidate_l1_cache();
        if (*sem >= value) {
            break;
        }
    }
    fence();
    WAYPOINT("RID");
}

inline void wait_all_riscs(uint32_t value) {
    for (uint32_t risc = 0; risc < RISCS; risc++) {
        wait_at_least(dspf2d::index_risc_sem(risc), value);
    }
}

// A pick's bucket, or BUCKET_NOT_HERE. The host does not check expert ids, and an id past the sentinel
// column would load a word from the next scratch block that can look like a valid bucket.
inline uint32_t bucket_of(const uint32_t* es, uint32_t expert) {
    return expert <= ct.num_routed_experts ? es[expert] : dspf2d::BUCKET_NOT_HERE;
}

// Pass 1: count my slice's picks per bucket, including those capacity will drop, since they still advance
// the page counter.
inline void count_pass(const Scratch& c, const Risc& me, uint32_t t0, uint32_t t1) {
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
            const uint32_t bucket = bucket_of(es, idx[k]);
            if (bucket >= n) {
                continue;  // BUCKET_NOT_HERE, or a corrupt table entry; never index past this RISC's counters
            }
            me.cnt[bucket] = me.cnt[bucket] + 1u;
        }
    }
}

// Between the passes: where my slice's pages and records start in every bucket, from the counts of the
// slices before mine.
inline void place_slice(const Scratch& c, const Risc& me, uint32_t risc) {
    const uint32_t n = num_buckets();
    for (uint32_t b = 0; b < n; b++) {
        uint32_t before = 0;
        for (uint32_t v = 0; v < risc; v++) {
            before += risc_view(c, v).cnt[b];
        }
        const uint32_t first_page = c.first_page[b];
        me.next_page[b] = first_page + before;
        me.next_record[b] = c.bucket_start[b] + kept_count(first_page, before);
    }
}

// Pass 2: walk my slice from the positions place_slice computed, writing one record per kept pick.
inline void fill_pass(const Scratch& c, const Risc& me, uint32_t t0, uint32_t t1) {
    const uint32_t cap = ct.max_dispatch_buffer_token_size;
    const uint32_t n = num_buckets();
    const uint32_t* es = frozen(c.expert_bucket);
    uint32_t idx_addr = reinterpret_cast<uint32_t>(c.indices) + t0 * ct.indices_pad_stride;
    for (uint32_t t = t0; t < t1; t++, idx_addr += ct.indices_pad_stride) {
        const uint16_t* idx = reinterpret_cast<const uint16_t*>(idx_addr);
#pragma GCC unroll 8
        for (uint32_t k = 0; k < ct.topk; k++) {
            const uint32_t bucket = bucket_of(es, idx[k]);
            if (bucket >= n) {
                continue;  // as in count_pass
            }
            const uint32_t page = me.next_page[bucket];
            me.next_page[bucket] = page + 1u;
            if (page >= cap) {
                continue;  // dropped for want of capacity, with the counter already advanced
            }
            const uint32_t at = me.next_record[bucket];
            // Keeps an offsets table that disagrees with the indices from writing into the next bucket.
            // The ASSERT in merge_routing_index reports it, but only when the watcher is enabled.
            if (at >= c.bucket_start[bucket + 1u]) {
                continue;
            }
            me.next_record[bucket] = at + 1u;
            volatile tt_l1_ptr uint32_t* rec = c.records + at * dspf2d::record_words();
            rec[0] = t;
            rec[1] = page;
            rec[2] = k;
        }
    }
}

// The whole of one RISC's share: wait for the tables, count, exchange, fill, report.
inline void run_risc(const Scratch& c, uint32_t risc) {
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

// The reader's side. The other RISCs read the tables with plain loads, so the tables must be complete before
// the signal that releases them. The caller must wait_all_riscs(kRiscFilled) before reading their records.
template <typename BuildTables>
inline void run_on_reader(const Scratch& c, BuildTables&& build_tables) {
    build_tables();
    signal(dspf2d::kSemTablesReady, 1u);
    run_risc(c, dspf2d::kRiscReader);
}

}  // namespace dspf2d::routing_index
