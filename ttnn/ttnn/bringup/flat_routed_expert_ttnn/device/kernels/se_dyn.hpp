// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Dynamic token counts for the flat expert (data movement side). The routing produces, on device, a token count and a
// region row offset per global expert (uint32 [1, N] row-major tensors in DRAM). A program serves its chip's NUM_E
// local experts whose count lies in its band [LO, HI] (a hybrid runs two programs with disjoint bands); every other
// expert is skipped outright, weights included. Every kernel derives the same active-expert list from the same
// tensors, so the roles stay in step without any host round trip.
//
// Runtime args at DYN0 (identical on every core): counts address, regions address, row bytes (4 N), LO, HI, then the
// NUM_E local experts' global ids.
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "se_meta.hpp"

// bytes of scratch per table (counts + the local experts' global ids, regions); a RISC owns 2 * SE_DYN_HALF of CB 7
// (BRISC the low, NCRISC the high half)
#ifndef SE_DYN_HALF
#define SE_DYN_HALF 1024
#endif
// runtime args of the dynamic schedule (from DYN0): counts row address, regions row address, row bytes (4 x global
// experts), count band lo / hi, global expert id table address (this device's local experts' ids, uint32), its bytes
constexpr uint32_t SE_DYN_NARGS = 7;
// gate/up weight ring regions (one expert's slice each; host: ring / blocks per expert)
#ifndef SE_GU_NREG
#define SE_GU_NREG 2
#endif

// The schedule is a list of entries, each a run of sub-blocks of one expert (its rows off.. off + cnt). Without
// pinning an entry is a whole active expert. With SE_PIN_MIN (sub-blocks), the expert with the most sub-blocks (B)
// is cut into K chunks of >= SE_PIN_MIN sub-blocks interleaved with the small (<= SE_PIN_SMALL sub-blocks) experts,
// B1 s1 B2 s2 .. BK sK sK+1 .. then the larger experts,
// and its gate/up weights stay in ring region 0 until BK (one load, several entries) while the others cycle
// through the other region: each small expert's weights stream in during a compute-bound chunk of B instead of the
// small expert running weight-bound on its own. Everything but the gate/up weights (x, h, down weights, y) just
// sees more entries (B's down weights are re-read per chunk).
//
// Gate/up loads land in the ring in load order, one region each; the receivers grant slots by count (the compute
// pops a block after its last use), so load l must go to the region of the (l - NREG)-th load to retire, and that
// retirement must come before load l's first use (else the schedule would deadlock: then pinning is dropped).
struct SeDyn {
    uint32_t n_act = 0;       // schedule entries
    uint32_t num_v = 0;       // sub-blocks over all entries
    uint32_t off[SE_MAX_V];   // entry a's first row in the dispatch buffer (TILE_HEIGHT-aligned)
    uint32_t cnt[SE_MAX_V];   // its tokens
    uint16_t subs[SE_MAX_V];  // its sub-blocks (<= 65535: a count below 2M rows)
    uint8_t eid[SE_MAX_V];    // local expert index of entry a
    uint8_t ld[SE_MAX_V];     // its gate/up weight load
    uint8_t last[SE_MAX_V];   // 1: the load's last use
    uint32_t n_load = 0;      // gate/up weight loads (the distinct active experts), in stream order
    uint8_t load_eid[SE_MAX_E];
    uint8_t region[SE_MAX_E];  // ring region of load l
    uint32_t max_cnt = 0;      // the largest active count
    uint32_t rps = 0;          // rows per sub-block
    bool small = false;        // SE_SMALL_T: every active expert has at most SE_SMALL_T tokens (small-M role split)
};

// Ring regions from the retirement order; false if some load would wait on a retirement that comes after its use.
inline bool se_dyn_regions(SeDyn& d) {
    // (8-bit: entries < SE_MAX_V <= 128; the 8 KB RISC local memory holds the stack, SeDyn and the pin copy)
    uint8_t ret_pos[SE_MAX_E], ret_ld[SE_MAX_E], first[SE_MAX_E];
    uint32_t n_ret = 0;
    for (uint32_t l = 0; l < d.n_load; ++l) {
        first[l] = 0xFF;
    }
    for (uint32_t a = 0; a < d.n_act; ++a) {
        if (first[d.ld[a]] == 0xFF) {
            first[d.ld[a]] = a;
        }
        if (d.last[a]) {
            ret_pos[n_ret] = a;
            ret_ld[n_ret++] = d.ld[a];
        }
    }
    for (uint32_t l = 0; l < d.n_load; ++l) {
        if (l < SE_GU_NREG) {
            d.region[l] = l;
            continue;
        }
        if (ret_pos[l - SE_GU_NREG] >= first[l]) {
            return false;
        }
        d.region[l] = d.region[ret_ld[l - SE_GU_NREG]];
    }
    return true;
}

#ifdef SE_PIN_MIN
#ifndef SE_PIN_SMALL
#define SE_PIN_SMALL 2  // sub-blocks: an expert this small runs weight-bound, so it is worth hiding under B
#endif
inline void se_dyn_pin(SeDyn& d, uint32_t rps) {
    if (d.n_act < 2 || d.n_act > SE_MAX_E) {
        return;
    }
    uint32_t b = 0;
    for (uint32_t a = 1; a < d.n_act; ++a) {
        b = d.subs[a] > d.subs[b] ? a : b;
    }
    const uint32_t sb = d.subs[b];
    // the others in schedule order: the small (weight-bound) ones first, they are interleaved with B's chunks; the
    // larger ones (compute-bound themselves) after B
    uint8_t order[SE_MAX_E];
    uint32_t n_small = 0, n_o = 0;
    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t a = 0; a < d.n_act; ++a) {
            if (a != b && (d.subs[a] <= SE_PIN_SMALL) == (pass == 0)) {
                order[n_o++] = a;
                n_small += pass == 0;
            }
        }
    }
    uint32_t k = sb / SE_PIN_MIN;
    k = k < n_small + 1 ? k : n_small + 1;  // one chunk per small expert (+1)
    if (k < 2) {
        return;
    }
    // the original entries (at most SE_MAX_E: a full SeDyn copy overflows the RISC stack at SE_MAX_E 64)
    struct {
        uint32_t n_act, off[SE_MAX_E], cnt[SE_MAX_E];
        uint16_t subs[SE_MAX_E];
        uint8_t eid[SE_MAX_E];
    } o;
    o.n_act = d.n_act;
    for (uint32_t a = 0; a < d.n_act; ++a) {
        o.off[a] = d.off[a];
        o.cnt[a] = d.cnt[a];
        o.subs[a] = d.subs[a];
        o.eid[a] = d.eid[a];
    }
    uint32_t v = 0, s0 = 0, j = 0;
    auto put = [&](uint32_t src, uint32_t off, uint32_t cnt, uint32_t subs, uint32_t ld, bool last) {
        d.eid[v] = o.eid[src];
        d.off[v] = off;
        d.cnt[v] = cnt;
        d.subs[v] = subs;
        d.ld[v] = ld;
        d.last[v] = last;
        ++v;
    };
    auto other = [&](uint32_t jj) { return static_cast<uint32_t>(order[jj]); };
    for (uint32_t c = 0; c < k; ++c) {
        const uint32_t ns = sb / k + (c < sb % k ? 1 : 0);
        const uint32_t cnt = o.cnt[b] - s0 * rps < ns * rps ? o.cnt[b] - s0 * rps : ns * rps;
        put(b, o.off[b] + s0 * rps, cnt, ns, 0, c + 1 == k);
        s0 += ns;
        if (j + 1 < o.n_act) {
            const uint32_t x = other(j);
            put(x, o.off[x], o.cnt[x], o.subs[x], 1 + j, true);
            ++j;
        }
    }
    for (; j + 1 < o.n_act; ++j) {
        const uint32_t x = other(j);
        put(x, o.off[x], o.cnt[x], o.subs[x], 1 + j, true);
    }
    d.n_act = v;
    d.load_eid[0] = o.eid[b];
    for (uint32_t jj = 0; jj + 1 < o.n_act; ++jj) {
        d.load_eid[1 + jj] = o.eid[other(jj)];
    }
    if (!se_dyn_regions(d)) {  // back to the plain order (se_dyn_load's)
        d.n_act = o.n_act;
        for (uint32_t a = 0; a < o.n_act; ++a) {
            d.off[a] = o.off[a];
            d.cnt[a] = o.cnt[a];
            d.subs[a] = o.subs[a];
            d.eid[a] = o.eid[a];
            d.ld[a] = a;
            d.last[a] = 1;
            d.load_eid[a] = o.eid[a];
        }
    }
}
#endif

#ifdef SE_SG
// Disjoint subgrids (SE_SG = number of subgrids): each subgrid is a complete copy of the pipeline on its own cores
// and serves its own experts. The active experts are assigned to subgrids deterministically (every core computes
// the same assignment from the same counts): largest first, each onto the currently least-loaded subgrid, cost
// = sub-blocks + SE_SG_WCOST (the weight streaming an expert costs whatever its rows). The subgrid id is the
// runtime arg after the NUM_E global ids. Keeps only this subgrid's experts (in their original order).
#ifndef SE_SG_WCOST
#define SE_SG_WCOST 2
#endif
inline void se_dyn_subgrid(SeDyn& d, uint32_t sg) {
    // An expert costing more than a subgrid's fair share (total / SE_SG) is cut by token range into SE_SG pieces
    // (whole sub-blocks), each streaming its own copy of the weights: its rows then run on every subgrid at once.
    {
        uint32_t total = 0;
        for (uint32_t a = 0; a < d.n_act; ++a) {
            total += d.subs[a] + SE_SG_WCOST;
        }
        const uint32_t n0 = d.n_act, rps = d.rps;
        for (uint32_t a = 0; a < n0; ++a) {
            if ((uint32_t(d.subs[a]) + SE_SG_WCOST) * SE_SG <= total || d.subs[a] < SE_SG ||
                d.n_act + SE_SG - 1 > SE_MAX_E) {
                continue;
            }
            const uint32_t sb = d.subs[a], cnt = d.cnt[a], off = d.off[a];
            uint32_t s0 = 0;
            for (uint32_t k = 0; k < SE_SG; ++k) {
                const uint32_t ns = sb / SE_SG + (k < sb % SE_SG ? 1 : 0);
                const uint32_t idx = k == 0 ? a : d.n_act++;
                d.eid[idx] = d.eid[a];
                d.off[idx] = off + s0 * rps;
                d.cnt[idx] = cnt - s0 * rps < ns * rps ? cnt - s0 * rps : ns * rps;
                d.subs[idx] = ns;
                s0 += ns;
            }
        }
    }
    uint32_t load[SE_SG], owner[SE_MAX_E], done = 0;
    for (uint32_t k = 0; k < SE_SG; ++k) {
        load[k] = 0;
    }
    bool taken[SE_MAX_E];
    for (uint32_t a = 0; a < d.n_act; ++a) {
        taken[a] = false;
    }
    for (; done < d.n_act; ++done) {
        uint32_t best = 0xFFFF;
        for (uint32_t a = 0; a < d.n_act; ++a) {  // the largest unassigned (ties: lowest index)
            if (!taken[a] && (best == 0xFFFF || d.subs[a] > d.subs[best])) {
                best = a;
            }
        }
        uint32_t k_min = 0;
        for (uint32_t k = 1; k < SE_SG; ++k) {
            k_min = load[k] < load[k_min] ? k : k_min;
        }
        taken[best] = true;
        owner[best] = k_min;
        load[k_min] += d.subs[best] + SE_SG_WCOST;
    }
    uint32_t v = 0;
    d.num_v = 0;
    d.max_cnt = 0;
    for (uint32_t a = 0; a < d.n_act; ++a) {
        if (owner[a] != sg) {
            continue;
        }
        d.eid[v] = d.eid[a];
        d.cnt[v] = d.cnt[a];
        d.off[v] = d.off[a];
        d.subs[v] = d.subs[a];
        d.ld[v] = v;
        d.last[v] = 1;
        d.load_eid[v] = d.eid[a];
        d.num_v += d.subs[a];
        d.max_cnt = d.cnt[a] > d.max_cnt ? d.cnt[a] : d.max_cnt;
        ++v;
    }
    d.n_act = v;
    d.n_load = v;
}
#endif

// Reads the counts / regions rows and the global expert id table into SCRATCH (two SE_DYN_HALF halves of L1 this RISC
// owns: counts then the ids, regions) and fills d. The sub-block size is SE_RPS rows when defined (every kernel must
// build the same schedule), else ROWS_PER_SUB. The subgrid id (SE_SG) is the runtime arg after the dynamic args.
template <uint32_t num_e>
inline void se_dyn_load(SeDyn& d, uint32_t dyn0, uint32_t scratch, uint32_t rows_per_sub) {
    static_assert(num_e <= SE_MAX_E);
#ifdef SE_RPS
    rows_per_sub = SE_RPS;
#endif
    const uint32_t counts_addr = get_arg_val<uint32_t>(dyn0), regions_addr = get_arg_val<uint32_t>(dyn0 + 1);
    const uint32_t row_bytes = get_arg_val<uint32_t>(dyn0 + 2);
    const uint32_t lo = get_arg_val<uint32_t>(dyn0 + 3), hi = get_arg_val<uint32_t>(dyn0 + 4);
    const uint32_t gidx_addr = get_arg_val<uint32_t>(dyn0 + 5), gidx_bytes = get_arg_val<uint32_t>(dyn0 + 6);
    const uint32_t rd = (row_bytes + 63) / 64 * 64;
    const uint32_t ids_l1 = scratch + rd;  // after the counts row (host: SE_DYN_HALF >= rd + the ids, rounded)
    const InterleavedAddrGen<true> cg = {.bank_base_address = counts_addr, .page_size = row_bytes};
    const InterleavedAddrGen<true> rg = {.bank_base_address = regions_addr, .page_size = row_bytes};
    const InterleavedAddrGen<true> ig = {.bank_base_address = gidx_addr, .page_size = gidx_bytes};
    noc_async_read(get_noc_addr(0, cg), scratch, rd);
    noc_async_read(get_noc_addr(0, rg), scratch + SE_DYN_HALF, rd);
    noc_async_read(get_noc_addr(0, ig), ids_l1, (gidx_bytes + 63) / 64 * 64);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    volatile tt_l1_ptr uint32_t* regions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + SE_DYN_HALF);
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_l1);
    d.n_act = 0;
    d.num_v = 0;
    d.rps = rows_per_sub;
    for (uint32_t e = 0; e < num_e; ++e) {
        const uint32_t g = ids[e];
        const uint32_t c = counts[g];
        if (c == 0 || c < lo || c > hi) {
            continue;
        }
        const uint32_t a = d.n_act++;
        d.eid[a] = e;
        d.cnt[a] = c;
        d.off[a] = regions[g];
        d.subs[a] = (c + rows_per_sub - 1) / rows_per_sub;
        d.ld[a] = a;
        d.last[a] = 1;
        d.load_eid[a] = e;
        d.num_v += d.subs[a];
        d.max_cnt = c > d.max_cnt ? c : d.max_cnt;
    }
    d.n_load = d.n_act;
#ifdef SE_SG
    se_dyn_subgrid(d, get_arg_val<uint32_t>(dyn0 + SE_DYN_NARGS));
#endif
#ifdef SE_SMALL_T
    d.small = d.max_cnt <= SE_SMALL_T;
#endif
#ifdef SE_PIN_MIN
    if (!d.small) {
        se_dyn_pin(d, rows_per_sub);
    }
#endif
    se_dyn_regions(d);  // plain order: always consistent (region l % NREG)
}

inline uint32_t se_dyn_rps(const SeDyn& d) { return d.rps; }

// Hands the schedule to this core's compute kernel: one page of CB META_CB in the se_meta.hpp layout (read there
// with read_tile_value). EMPTY publishes no entries (the core has no compute work this launch).
inline void se_dyn_publish(const SeDyn& d, uint32_t meta_cb, bool empty = false) {
#ifndef SE_META_BYTES
#define SE_META_BYTES 512
#endif
    static_assert(SE_META_WORDS * 4 <= SE_META_BYTES, "CB 6 page (host: SE_META_BYTES)");
    cb_reserve_back(meta_cb, 1);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(meta_cb));
    p[0] = empty ? 0 : d.n_act;
    p[1] = empty ? 0 : d.num_v;
    for (uint32_t a = 0; a < d.n_act; ++a) {
        p[SE_META_SUBS + a] = d.subs[a];
        p[SE_META_GU + a] = d.ld[a] | (d.region[d.ld[a]] << 8) | (uint32_t(d.last[a]) << 16);
        p[SE_META_LMT + a] = (d.cnt[a] - (d.subs[a] - 1u) * se_dyn_rps(d) + 31) / 32;
    }
    p[SE_META_SMALL] = d.small;
    cb_push_back(meta_cb, 1);
}
