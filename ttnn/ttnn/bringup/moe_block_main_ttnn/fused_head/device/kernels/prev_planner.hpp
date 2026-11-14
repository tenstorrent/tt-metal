// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The expert replica planner of MOE_REPLICAS=prev (DeepSeek-V4.1 P.56; terms, data layouts and the host mirror in
// models/demos/deepseek_v41_flash_d_p/tt/expert_prev.py): one MoE layer's plan for its NEXT call from the counts of the
// call that just ran - every chip runs the same integer arithmetic on every chip's counts and gets the same plan: the
// slot map into the layer's router bias, this chip's copies into the fill job pages of the coming MoE calls (the
// combine's REC_FX pages), the layer's and the global state.
//
// Two hosts (P.62b): the standalone planner program (the model's kernels/moe_prev_planner.cpp, ROLE 0) and the fused
// head's watcher tail (xg_watcher.cpp under W_PREV: MOE_PLANNER=folded, no launch of its own). Its compile-time
// constants are the model kernel's compile args from PP_CT0 on (expert_prev.Geometry.ct order: ROLE, NC, EPC, ...).
//
// Io.cnt != 0: this chip's own counts row is read from the fused head's counts (the standalone planner right after the
// call). Io.cnt == 0 (folded): the own row comes from the landing rows too - the combine's port wrote it there, its
// last word (SPW - 1) the call ordinal + 1 - and a plan runs only when that tag says the call G[0] is pending (else
// nothing is read past the state or written: plan() returns false).
#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

#ifndef PP_CT0
#error "prev_planner.hpp: define PP_CT0 (the index of the planner's first compile arg)"
#endif

namespace pp {

constexpr uint32_t NC = get_compile_time_arg_val(PP_CT0 + 1);        // chips
constexpr uint32_t EPC = get_compile_time_arg_val(PP_CT0 + 2);       // home slots a chip
constexpr uint32_t SPC = get_compile_time_arg_val(PP_CT0 + 3);       // slots a chip (EPC + the spare)
constexpr uint32_t SPW = get_compile_time_arg_val(PP_CT0 + 4);       // a landing row's words (SPC rounded up to 16)
constexpr uint32_t W = get_compile_time_arg_val(PP_CT0 + 5);         // job page ring (MoE calls)
constexpr uint32_t NCALL = get_compile_time_arg_val(PP_CT0 + 6);     // MoE calls a chunk
constexpr uint32_t PCS = get_compile_time_arg_val(PP_CT0 + 7);       // pieces a copy
constexpr uint32_t BUD = get_compile_time_arg_val(PP_CT0 + 8);       // pieces a source port a call
constexpr uint32_t CAP = get_compile_time_arg_val(PP_CT0 + 9);       // the pages' cap word
constexpr uint32_t MAXJ = get_compile_time_arg_val(PP_CT0 + 10);     // jobs a page
constexpr uint32_t FIX = get_compile_time_arg_val(PP_CT0 + 11);      // a replica's fixed cost (tokens)
constexpr uint32_t MINX = get_compile_time_arg_val(PP_CT0 + 12);     // the least tokens a replica takes
constexpr uint32_t SLACK = get_compile_time_arg_val(PP_CT0 + 13);    // a held replica is kept within this many tokens
constexpr uint32_t MAXNEW = get_compile_time_arg_val(PP_CT0 + 14);   // new copies a plan
constexpr uint32_t LNK = get_compile_time_arg_val(PP_CT0 + 15);      // links a direction
constexpr uint32_t CPAGE = get_compile_time_arg_val(PP_CT0 + 16);    // head.counts' page bytes (aligned)
constexpr uint32_t GW = get_compile_time_arg_val(PP_CT0 + 17);       // global state words
constexpr uint32_t SW = get_compile_time_arg_val(PP_CT0 + 18);       // a layer state's words
constexpr uint32_t MAPPAGE = get_compile_time_arg_val(PP_CT0 + 19);  // the router bias's slot map tile (page)
// > 0: ONE job tensor of W page sets (slot s = pages s x RINGPG ..); 0: W job tensors, one address arg each
constexpr uint32_t RINGPG = get_compile_time_arg_val(PP_CT0 + 20);
constexpr uint32_t E = NC * EPC;
constexpr uint32_t NO_EXPERT = 0xFFFFFFFFu;
constexpr uint32_t PG = 1024;                   // a job page's bytes
constexpr uint32_t GSTAMP = 16, GBUD = 16 + W;  // global state: stamps [W], budgets [W][NC][4] (pieces | jobs << 16)
constexpr uint32_t SHOLD = 16, SREADY = 16 + NC, SRECV = 16 + 2 * NC, SRX = 16 + 3 * NC;

inline uint32_t al64(uint32_t x) { return (x + 63) & ~63u; }

inline uint32_t f32_bits(uint32_t v) {  // an integer below 2^24 as fp32 bits (exact)
    if (v == 0) {
        return 0;
    }
    uint32_t e = 0, w = v;  // (v's top bit in five steps: the RISCs have no clz)
    if (w >> 16) {
        w >>= 16;
        e += 16;
    }
    if (w >> 8) {
        w >>= 8;
        e += 8;
    }
    if (w >> 4) {
        w >>= 4;
        e += 4;
    }
    if (w >> 2) {
        w >>= 2;
        e += 2;
    }
    if (w >> 1) {
        e += 1;
    }
    return ((127u + e) << 23) | ((v << (23 - e)) & 0x7FFFFFu);
}

inline uint32_t tile_index(uint32_t i) {  // word i of a row-major 32 x 32 tile -> its place in the tile's faces
    const uint32_t r = i >> 5, c = i & 31;
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

inline void read_page(uint32_t base, uint32_t page_bytes, uint32_t page, uint32_t l1, uint32_t bytes) {
    const InterleavedAddrGen<true> g = {.bank_base_address = base, .page_size = page_bytes};
    noc_async_read(g.get_noc_addr(page), l1, bytes);
}

inline void write_page(uint32_t base, uint32_t page_bytes, uint32_t page, uint32_t off, uint32_t l1, uint32_t bytes) {
    const InterleavedAddrGen<true> g = {.bank_base_address = base, .page_size = page_bytes};
    noc_async_write(l1, g.get_noc_addr(page, off), bytes);
}

// The planner's inputs: the counts (0: folded, see above), the landing rows, the global state, the layer's state,
// router bias, gate/up and down buffers, the directions + empty map page, this chip's position, the arrival counter (an
// L1 word on this core), the job ring (RINGPG: its one address; else the runtime arg ring_arg0 + slot holds each
// slot's), and the L1 the planner may use from base (expert_prev.Geometry.l1_bytes).
struct Io {
    uint32_t cnt, land, gst, sta, rb, gu, dw, cst, me, sem, ring, ring_arg0, base;
};

inline uint32_t ring_at(const Io& io, uint32_t s) {
    if constexpr (RINGPG != 0) {
        return io.ring;
    } else {
        return get_arg_val<uint32_t>(io.ring_arg0 + s);
    }
}
inline uint32_t ring_page(uint32_t s, uint32_t pg) { return RINGPG ? s * RINGPG + pg : pg; }

// true: planned (false: folded and no plan pending - nothing written)
inline bool plan(const Io& io) {
    const uint32_t cnt = io.cnt, land = io.land, gst = io.gst, sta = io.sta, rb = io.rb, gu = io.gu, dw = io.dw;
    const uint32_t cst = io.cst, me = io.me, sem = io.sem, base = io.base;
    // L1: counts page, landing rows, global state, layer state, directions, map tile, header / job staging, work
    const uint32_t l_cnt = base, l_land = al64(l_cnt + CPAGE), l_g = al64(l_land + NC * SPW * 4);
    const uint32_t l_st = al64(l_g + GW * 4), l_dir = al64(l_st + SW * 4), l_map = al64(l_dir + NC * NC * 4);
    const uint32_t l_hdr = l_map + 4096, l_job = l_hdr + 64, l_stg = l_job + PG, l_pgs = l_stg + al64(NCALL * 16);
    const uint32_t l_load = l_pgs + NCALL * PG;
    const uint32_t l_share = l_load + E * 4, l_T = l_share + E * 4, l_nr = l_T + NC * 4, l_used = l_nr + E * 4;
    const uint32_t l_fail = l_used + NC * 4;
    volatile tt_l1_ptr uint32_t* cw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_cnt);
    volatile tt_l1_ptr uint32_t* lw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_land);
    volatile tt_l1_ptr uint32_t* G = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_g);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_st);
    volatile tt_l1_ptr uint32_t* dir = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_dir);
    volatile tt_l1_ptr uint32_t* map = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_map);
    volatile tt_l1_ptr uint32_t* hw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_hdr);
    volatile tt_l1_ptr uint32_t* stg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_stg);
    uint32_t* load = reinterpret_cast<uint32_t*>(l_load);    // (this core's own work array)
    uint32_t* share = reinterpret_cast<uint32_t*>(l_share);  // (this core's own work array)
    uint32_t* T = reinterpret_cast<uint32_t*>(l_T);          // (this core's own work array)
    uint32_t* nr = reinterpret_cast<uint32_t*>(l_nr);        // (this core's own work array)
    uint32_t* used = reinterpret_cast<uint32_t*>(l_used);    // (this core's own work array)
    uint32_t* fail = reinterpret_cast<uint32_t*>(l_fail);    // (this core's own work array)
    const uint32_t t0 = get_timestamp_32b();
    read_page(gst, GW * 4, 0, l_g, GW * 4);
    read_page(sta, SW * 4, 0, l_st, SW * 4);
    read_page(cst, NC * NC * 4 + 4096, 0, l_dir, NC * NC * 4);
    {  // (the empty map's tile, fp32 bits in tile order: the directions' page holds it after them)
        const InterleavedAddrGen<true> cg = {.bank_base_address = cst, .page_size = NC * NC * 4 + 4096};
        noc_async_read(cg.get_noc_addr(0, NC * NC * 4), l_map, 4096);
    }
    if (cnt) {
        read_page(cnt, CPAGE, 0, l_cnt, CPAGE);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t g = G[0], par = g & 1, epoch = G[1];
    if (!cnt) {  // (folded: my own row, written by my chip's combine port; its tag says whether call g is pending)
        read_page(land, SPW * 4, par * NC + me, l_land + me * SPW * 4, SPW * 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
        static_assert(SPW > SPC, "prev_planner: the folded landing row needs a tag word past the slots");
        if (lw[me * SPW + SPW - 1] != g + 1) {
            return false;
        }
    }
    const uint32_t t1 = get_timestamp_32b();
    // the other chips' counts: n - 1 arrivals, then taken back
    if (NC > 1) {
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
        if (cnt) {
            noc_semaphore_wait_min(arrived, NC - 1);
        } else {  // (folded: bounded - a missing packet must not hang the head; then nothing is written, the host's
                  // mirror sees the difference)
            while (true) {
                invalidate_l1_cache();
                if (*arrived >= NC - 1) {
                    break;
                }
                if (get_timestamp_32b() - t0 > 0x7FFFFFFFu) {
                    return false;
                }
            }
        }
        noc_semaphore_inc(get_noc_addr(sem), 0u - (NC - 1));
        noc_async_atomic_barrier();
    }
    for (uint32_t s = 0; s < NC; ++s) {
        if (s != me) {
            read_page(land, SPW * 4, par * NC + s, l_land + s * SPW * 4, SPW * 4);
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t t2 = get_timestamp_32b();
    if (cnt) {
        for (uint32_t j = 0; j < SPW; ++j) {
            lw[me * SPW + j] = j < SPC ? cw[me * SPC + j] : 0u;
        }
    }
    // the job pages call g just read: their headers zeroed (that ring slot is next used at g + W)
    for (uint32_t i = 0; i < 16; ++i) {
        hw[i] = 0;
    }
    for (uint32_t d = 0; d < 4; ++d) {
        write_page(ring_at(io, g % W), PG, ring_page(g % W, d * LNK), 0, l_hdr, 64);
    }
    noc_async_write_barrier();
    // the layer's state (another epoch: empty)
    if (st[0] != epoch) {
        st[0] = epoch;
        for (uint32_t k = 0; k < NC; ++k) {
            st[SHOLD + k] = NO_EXPERT;
            st[SREADY + k] = 0;
        }
    }
    for (uint32_t k = 0; k < NC; ++k) {
        st[SRECV + k] = NO_EXPERT;
        st[SRX + k] = 0;
        used[k] = 0;
        fail[k] = 0;
    }
    // each expert's tokens in the call that just ran (its home slot + the spare slots that held it)
    for (uint32_t b = 0; b < E; ++b) {
        load[b] = lw[(b / EPC) * SPW + b % EPC];
        nr[b] = 0;
    }
    for (uint32_t k = 0; k < NC; ++k) {
        if (st[SHOLD + k] != NO_EXPERT) {
            load[st[SHOLD + k]] += lw[k * SPW + EPC];
        }
    }
    uint32_t tsum = 0, tmax0 = 0;
    for (uint32_t c = 0; c < NC; ++c) {
        uint32_t t = 0;
        for (uint32_t j = 0; j < EPC; ++j) {
            t += load[c * EPC + j];
        }
        T[c] = t;
        tsum += t;
        tmax0 = t > tmax0 ? t : tmax0;
    }
    for (uint32_t b = 0; b < E; ++b) {
        share[b] = load[b];
    }
    uint32_t nrec = 0, nreuse = 0, nnew = 0, ndrop = 0, pieces = 0;
    // fill: copy (h -> k) of home slot `tab` in the coming calls g + 1 .. g + NCALL - 2 (the last one before the
    // layer's next call is a fence: its start barrier on the same routes lands the pieces); on success the last call
    // that carries a piece, else 0 (nothing changed)
    auto alloc = [&](uint32_t h, uint32_t k, uint32_t tab) -> uint32_t {
        const uint32_t p = dir[h * NC + k];
        uint32_t room = 0;
        for (uint32_t o = g + 1; o + 2 <= g + NCALL; ++o) {
            const uint32_t s = o % W;
            const uint32_t v = G[GSTAMP + s] == o ? G[GBUD + (s * NC + h) * 4 + p] : 0u;
            if ((v >> 16) < MAXJ && (v & 0xFFFF) < BUD) {
                room += BUD - (v & 0xFFFF);
            }
        }
        if (room < PCS) {
            return 0;
        }
        uint32_t rem = PCS, p0 = 0, last = 0, ns = 0;
        for (uint32_t o = g + 1; o + 2 <= g + NCALL && rem; ++o) {
            const uint32_t s = o % W;
            if (G[GSTAMP + s] != o) {
                G[GSTAMP + s] = o;
                for (uint32_t i = 0; i < NC * 4; ++i) {
                    G[GBUD + s * NC * 4 + i] = 0;
                }
            }
            const uint32_t v = G[GBUD + (s * NC + h) * 4 + p];
            const uint32_t u = v & 0xFFFF, j = v >> 16;
            if (j >= MAXJ || u >= BUD) {
                continue;
            }
            const uint32_t take = BUD - u < rem ? BUD - u : rem;
            G[GBUD + (s * NC + h) * 4 + p] = (u + take) | (j + 1) << 16;
            if (h == me) {  // (my job: page dir x LNK of ring slot s, job j - staged, written below)
                stg[4 * ns] = s;
                stg[4 * ns + 1] = j;
                stg[4 * ns + 2] = p0;
                stg[4 * ns + 3] = take;
                ++ns;
            }
            p0 += take;
            rem -= take;
            last = o;
        }
        if (ns) {  // (my jobs' pages read at once, changed, written at once: whole pages, DRAM stays 64 B aligned)
            for (uint32_t i = 0; i < ns; ++i) {
                read_page(ring_at(io, stg[4 * i]), PG, ring_page(stg[4 * i], p * LNK), l_pgs + i * PG, PG);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t i = 0; i < ns; ++i) {
                volatile tt_l1_ptr uint32_t* pg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_pgs + i * PG);
                const uint32_t j = stg[4 * i + 1];
                pg[0] = j + 1;
                pg[1] = CAP;
                volatile tt_l1_ptr uint32_t* e = pg + 16 + 8 * j;
                e[0] = tab | k << 8;
                e[1] = stg[4 * i + 2];
                e[2] = stg[4 * i + 2] + stg[4 * i + 3];
                e[3] = 0;
                e[4] = 0;
                e[5] = 0;
                e[6] = gu;
                e[7] = dw;
                write_page(ring_at(io, stg[4 * i]), PG, ring_page(stg[4 * i], p * LNK), 0, l_pgs + i * PG, PG);
            }
            noc_async_write_barrier();
        }
        pieces += PCS;
        return last;
    };
    for (uint32_t it = 0; it < 2 * NC; ++it) {
        uint32_t h = 0;
        for (uint32_t c = 1; c < NC; ++c) {
            h = T[c] > T[h] ? c : h;
        }
        uint32_t b = h * EPC;
        for (uint32_t j = 1; j < EPC; ++j) {
            b = share[h * EPC + j] > share[b] ? h * EPC + j : b;
        }
        if (share[b] == 0) {
            break;
        }
        uint32_t kr = NO_EXPERT, kn = NO_EXPERT;
        for (uint32_t k = 0; k < NC; ++k) {
            if (k == h || used[k]) {
                continue;
            }
            if (st[SHOLD + k] == b && st[SREADY + k] <= g + NCALL && (kr == NO_EXPERT || T[k] < T[kr])) {
                kr = k;
            }
            if (!fail[k] && st[SREADY + k] <= g && nnew < MAXNEW && (kn == NO_EXPERT || T[k] < T[kn])) {
                kn = k;
            }
        }
        uint32_t k = NO_EXPERT;
        bool fresh = false;
        if (kr != NO_EXPERT && (kn == NO_EXPERT || T[kr] <= T[kn] + SLACK)) {
            k = kr;
        } else if (kn != NO_EXPERT) {
            k = kn;
            fresh = st[SHOLD + kn] != b;
        } else {
            break;
        }
        const int32_t gap = static_cast<int32_t>(T[h]) - static_cast<int32_t>(T[k]) - static_cast<int32_t>(FIX);
        uint32_t x = gap > 0 ? static_cast<uint32_t>(gap) >> 1 : 0u;
        x = x < share[b] ? x : share[b];
        if (x < MINX) {
            break;
        }
        if (fresh) {
            const uint32_t last = alloc(h, k, b % EPC);
            if (last == 0) {
                fail[k] = 1;
                ++ndrop;
                continue;
            }
            st[SHOLD + k] = b;
            st[SREADY + k] = last + 2;
            ++nnew;
        } else {
            ++nreuse;
        }
        used[k] = 1;
        st[SRECV + k] = b;
        st[SRX + k] = x;
        ++nr[b];
        ++nrec;
        share[b] -= x;
        T[h] -= x;
        T[k] += x + FIX;
    }
    const uint32_t t3 = get_timestamp_32b();
    uint32_t tmax1 = 0;
    for (uint32_t c = 0; c < NC; ++c) {
        tmax1 = T[c] > tmax1 ? T[c] : tmax1;
    }
    // the slot map (router_rows.hpp slot_of, the uneven split): own slots, copy words, copies (receivers by chip)
    uint32_t first = 0;  // (l_map holds the empty map: own slots, no copies)
    for (uint32_t b = 0; b < E; ++b) {
        if (nr[b] == 0) {
            continue;
        }
        map[tile_index(E + b)] = f32_bits(nr[b] | first << 6);
        const uint32_t tot = load[b];
        uint32_t acc = share[b], i = 0;
        for (uint32_t k = 0; k < NC; ++k) {
            if (st[SRECV + k] != b) {
                continue;
            }
            uint32_t lb = (acc * 32768u + tot / 2) / tot;
            lb = lb < 1 ? 1 : (lb > 32767 ? 32767 : lb);
            map[tile_index(2 * E + first + i)] = f32_bits((k * SPC + EPC) | lb << 9);
            acc += st[SRX + k];
            ++i;
        }
        first += nr[b];
    }
    write_page(rb, 4096, MAPPAGE, 0, l_map, 4096);
    st[1] = g;
    st[2] = nrec;
    st[3] = nreuse;
    st[4] = nnew;
    st[5] = ndrop;
    st[6] = tmax0;
    st[7] = tmax1;
    st[8] = tsum / NC;
    st[9] = 1;
    st[10] = t1 - t0;  // (diagnostic cycle stamps: state reads, the counts' wait + rows, the plan, the map)
    st[11] = t2 - t1;
    st[12] = t3 - t2;
    st[13] = get_timestamp_32b() - t3;
    G[0] = g + 1;
    G[2] = G[2] + nnew;
    G[3] = G[3] + pieces;
    G[4] = G[4] + ndrop;
    write_page(sta, SW * 4, 0, 0, l_st, SW * 4);
    write_page(gst, GW * 4, 0, 0, l_g, GW * 4);
    noc_async_write_barrier();
    return true;
}

}  // namespace pp
