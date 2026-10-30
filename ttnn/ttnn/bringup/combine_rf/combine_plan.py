# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""combine_plan — the flat expert's and the router-forwarded combine's per-call tables, built on the device.

From the routing (every token's top-k global expert ids and gate weights, replicated on every chip: [1, N * S * K]
uint16 ids and bf16 weights, token-major), each chip's planner (one core) builds, for its local experts and its home
tokens:
  counts / regions [1, NG] uint32       the flat expert's per-global-expert token counts and first rows (padded to 32)
  token_index [1, rows_cap] uint32      per flat row, the gathered-token row it reads (home * S + t)
  hand-off table [rows_cap / 32, 64]    per flat row (destination | gate weight bf16 << 16, header word): the y writers'
                                        L1 hand-off (se_yrm.hpp SE_Y_L1); header = home | t << 8 | local expert << 24
  plan [1, 1024] uint32                 the combine's per-call counts: n_inj[8], n_ret[8] per port id, n_local[16] per
                                        accumulator, want[16][32] per accumulator and source chip (N <= 32)
Rows of a local expert are in (token, home, k) scan order with the multi-core planner (token_major, the default; the
deterministic combine and the one-core planners: (home, token, k)): a hot expert's consecutive rows go to every home,
over every first-hop port and accumulator ring, not to one home at a time (in home order a hot expert's burst to one
home has only the ports toward it and that home's rings: the combine blocks on credits, FLAT_COMBINE_OVERLAP_WORKLOG
"GLM-5.3-Flash"). A row of home token t goes to accumulator a = t % A over
link (a + source + home) % L - fixed per (accumulator, source, home), so each accumulator ring has one producer port
per source chip - through this chip's first-hop port toward home (Y first, then X; rings the shorter way, ties Y- /
X+ as the fabric's tables; port_lut[direction * 2 + link] = port id); a local row (home is this chip) goes to
accumulator t % A (destination id P + t % A). The same rules as sparse_combine_rf_prepare's host
tables (plan_reference below is the host model)."""

import os

import torch

import ttnn

PLAN_WORDS = 1024
P_NINJ, P_NRET, P_NLOC, P_WANT = 0, 8, 16, 32  # plan word offsets
P_ROWS, P_OVER = 600, 601

_PLANNER = r"""
#include <cstdint>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
__DETINC1__
constexpr uint32_t ME = get_compile_time_arg_val(0), R = get_compile_time_arg_val(1), C = get_compile_time_arg_val(2);
constexpr uint32_t S = get_compile_time_arg_val(3), K = get_compile_time_arg_val(4), EPC = get_compile_time_arg_val(5);
constexpr uint32_t NG = get_compile_time_arg_val(6), A = get_compile_time_arg_val(7), L = get_compile_time_arg_val(8);
constexpr uint32_t P = get_compile_time_arg_val(9), ROWS_CAP = get_compile_time_arg_val(10);
// (column / pass options, defaults = the whole-mesh plan: EOFF 0, SF = S, T0 0, SKIPF 0) EOFF: subtracted from every id
// (a column's first global expert); SF: tokens per home in ids / x (S of them from T0 on are planned: a pass);
// SKIPF: ids outside [0, N EPC) after EOFF (another column's experts) and tokens past SF are skipped
constexpr bool SKIPF = __SKIPF__;
#if __COLOPT__
constexpr uint32_t EOFF = get_compile_time_arg_val(11), SF = get_compile_time_arg_val(12);
constexpr uint32_t T0 = get_compile_time_arg_val(13);
#else
constexpr uint32_t EOFF = 0, SF = S, T0 = 0;
#endif
constexpr uint32_t IDW = __IDW__;  // bytes per id in the ids page: 2 (uint16, default) or 4 (uint32)
using id_t = std::conditional_t<IDW == 4, uint32_t, uint16_t>;
constexpr uint32_t N = R * C, T = N * SF * K;
constexpr uint32_t cb_in = 0, cb_out = 1;
// direction index (sparse_all_gather_2d DIRS): 0 X+, 1 X-, 2 Y+, 3 Y-; Y first, then X (lines)
constexpr uint32_t WX = __WX__, WY = __WY__;  // torus axes (rings: the shorter way, ties Y- / X+ as the fabric)
constexpr uint32_t YT = __YT__;  // (SPC_YTIE: sparse_combine_l1.route(ytie=))
FORCE_INLINE uint32_t first_hop(uint32_t src, uint32_t dst) {
    const uint32_t sr = src / C, sc = src % C, dr = dst / C, dc = dst % C;
    if (sr != dr) {
        if constexpr (WY) {
            const uint32_t f2 = 2 * ((dr + R - sr) % R);  // (YT: the opposite row splits by column parity)
            return f2 < R || (YT && f2 == R && ((sc + dc) & 1)) ? 2 : 3;
        }
        return dr > sr ? 2 : 3;
    }
    if constexpr (WX) return 2 * ((dc + C - sc) % C) <= C ? 0 : 1;
    return dc > sc ? 0 : 1;
}
void kernel_main() {
    size_t a = 0;
    const uint32_t ids_addr = get_arg_val<uint32_t>(a++), w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t cnt_addr = get_arg_val<uint32_t>(a++), reg_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tidx_addr = get_arg_val<uint32_t>(a++), tab_addr = get_arg_val<uint32_t>(a++);
    const uint32_t plan_addr = get_arg_val<uint32_t>(a++);
    uint32_t lut[8];
    for (uint32_t i = 0; i < 8; ++i) lut[i] = get_arg_val<uint32_t>(a++);
    // L1: ids, weights | counts, regions, token index, table, plan
    cb_reserve_back(cb_in, 1);  // (64-byte aligned starts: DRAM transfers)
    const uint32_t ids_l1 = (get_write_ptr(cb_in) + 63) & ~63u, w_l1 = ids_l1 + ((T * IDW + 63) / 64) * 64;
    cb_reserve_back(cb_out, 1);
    const uint32_t cnt_l1 = (get_write_ptr(cb_out) + 63) & ~63u;
    const uint32_t reg_l1 = cnt_l1 + ((NG * 4 + 63) / 64) * 64;
    const uint32_t tidx_l1 = reg_l1 + ((NG * 4 + 63) / 64) * 64;
    const uint32_t tab_l1 = tidx_l1 + ROWS_CAP * 4;
    const uint32_t plan_l1 = tab_l1 + ROWS_CAP * 8;
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt_l1);
    volatile tt_l1_ptr uint32_t* reg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reg_l1);
    volatile tt_l1_ptr uint32_t* tidx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tidx_l1);
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab_l1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(plan_l1);
    if constexpr (__TILEIN__) {
        // (tile_in) ids / weights as TILE [tokens, K] (K <= 16: a token's K values sit in face 0 or face 2 of its
        // row tile): tile by tile into a staging slot (the table area, zeroed below), each row's values packed
        constexpr uint32_t NTOK = T / K, NTT = (NTOK + 31) / 32;
        const InterleavedAddrGen<true> it = {.bank_base_address = ids_addr, .page_size = 1024 * IDW};
        const InterleavedAddrGen<true> wt = {.bank_base_address = w_addr, .page_size = 2048};
        volatile tt_l1_ptr id_t* io = reinterpret_cast<volatile tt_l1_ptr id_t*>(ids_l1);
        volatile tt_l1_ptr uint16_t* wo = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w_l1);
        // (every tile read at once into the staging CB, cb 2: one barrier)
        constexpr uint32_t TB = 1024 * IDW + 2048;
        cb_reserve_back(2, 1);
        const uint32_t stg = (get_write_ptr(2) + 63) & ~63u;
        // (SKIPF, a pass: only the tiles holding its tokens [T0, T0 + S) of every home)
        auto needed = [&](uint32_t i) -> bool {
            if constexpr (!SKIPF) {
                return true;
            }
            for (uint32_t o = 0; o < N; ++o) {
                const uint32_t lo = o * SF + T0, hi = o * SF + (T0 + S < SF ? T0 + S : SF);
                if (i * 32 < hi && i * 32 + 32 > lo) return true;
            }
            return false;
        };
        for (uint32_t i = 0; i < NTT; ++i) {
            if (!needed(i)) continue;
            noc_async_read(get_noc_addr(i, it), stg + i * TB, 1024 * IDW);
            noc_async_read(get_noc_addr(i, wt), stg + i * TB + 1024 * IDW, 2048);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < NTT; ++i) {
            if (!needed(i)) continue;
            const id_t* si = reinterpret_cast<const id_t*>(stg + i * TB);
            const uint16_t* sw = reinterpret_cast<const uint16_t*>(stg + i * TB + 1024 * IDW);
            for (uint32_t r = 0; r < 32 && i * 32 + r < NTOK; ++r) {
                const uint32_t off = (r < 16 ? 0 : 512) + (r % 16) * 16, o = (i * 32 + r) * K;
                for (uint32_t k = 0; k < K; ++k) {
                    io[o + k] = si[off + k];
                    wo[o + k] = sw[off + k];
                }
            }
        }
    } else {
        const InterleavedAddrGen<true> ids_g = {.bank_base_address = ids_addr, .page_size = T * IDW};
        const InterleavedAddrGen<true> w_g = {.bank_base_address = w_addr, .page_size = T * 2};
        noc_async_read(get_noc_addr(0, ids_g), ids_l1, T * IDW);
        noc_async_read(get_noc_addr(0, w_g), w_l1, T * 2);
    }
    for (uint32_t i = 0; i < NG; ++i) cnt[i] = reg[i] = 0;
    for (uint32_t i = 0; i < ROWS_CAP; ++i) tidx[i] = 0;
    for (uint32_t i = 0; i < 2 * ROWS_CAP; ++i) tab[i] = 0;
    for (uint32_t i = 0; i < 1024; ++i) plan[i] = 0;
    noc_async_read_barrier();
    const volatile tt_l1_ptr id_t* ids = reinterpret_cast<const volatile tt_l1_ptr id_t*>(ids_l1);
    const volatile tt_l1_ptr uint16_t* wts = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(w_l1);
    constexpr uint32_t E0 = ME * EPC;
    // pass 1: tokens per local expert
    uint32_t lc[EPC];
    for (uint32_t e = 0; e < EPC; ++e) lc[e] = 0;
    if constexpr (SKIPF) {  // (a pass: only its tokens, home by home)
        constexpr uint32_t T1 = T0 + S < SF ? T0 + S : SF;
        const id_t* idn = const_cast<const id_t*>(ids);
        for (uint32_t o = 0; o < N; ++o) {
            for (uint32_t i = (o * SF + T0) * K, i1 = (o * SF + T1) * K; i < i1; ++i) {
                const uint32_t e = (idn[i] - EOFF) - E0;
                if (e < EPC) ++lc[e];
            }
        }
    } else {
        for (uint32_t i = 0; i < T; ++i) {
            const uint32_t e = (ids[i] - EOFF) - E0;
            if (e < EPC) ++lc[e];
        }
    }
    uint32_t fill[EPC], off = 0;
    for (uint32_t e = 0; e < EPC; ++e) {
        cnt[E0 + e] = lc[e];
        reg[E0 + e] = off;
        fill[e] = off;
        off += (lc[e] + 31) / 32 * 32;
    }
    plan[600] = off;
    plan[601] = off > ROWS_CAP ? 1 : 0;
    // pass 2: place the local experts' rows; count the combine's rows (this chip as source and as home)
    for (uint32_t o = 0; o < N; ++o) {
        const uint32_t dir_home = o == ME ? 0 : first_hop(ME, o);
        for (uint32_t t = 0; t < S; ++t) {
            if (SKIPF && T0 + t >= SF) break;
            const uint32_t base = (o * SF + T0 + t) * K;
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t g = ids[base + k] - EOFF, src = g / EPC, l = (t % A + src + o) % L;
                if (SKIPF && src >= N) continue;  // (another column's expert)
                if (src == ME) {
                    const uint32_t e = g - E0, r = fill[e]++;
                    if (r < ROWS_CAP) {
                        tidx[r] = o * SF + T0 + t;
                        uint32_t dest;
                        if (o == ME) {
                            dest = P + t % A;
                        } else {
                            dest = lut[dir_home * 2 + l];
                            ++plan[0 + dest];  // n_inj
                        }
                        tab[2 * r] = dest | (static_cast<uint32_t>(wts[base + k]) << 16);
                        tab[2 * r + 1] = o | (t << 8) | (e << 24);
                    }
                }
                if (o == ME) {  // a row for my token
                    if (src == ME) {
                        ++plan[16 + t % A];  // n_local
                    } else {
                        ++plan[32 + (t % A) * 32 + src];  // want (per source chip: its link is fixed)
                        ++plan[8 + lut[first_hop(ME, src) * 2 + l]];  // n_ret: the port facing the source
                    }
                }
            }
        }
    }
#if __DET1__
    // (det, the one-core planner: the deterministic combine's order table, as the multi-core planner's det) per source
    // chip of the group, its flat schedule (se_dyn_from_counts on its experts' counts of this pass) and each entry's
    // predicted finish (an entry costs max(WR, its rows)); per home token of mine, its rows' sources ordered by
    // (predicted finish of the entry carrying the row, source): per source its production order (no deadlock)
    {
        constexpr uint32_t VM = __VM1__, WR = __WR1__, DPG = __DPG1__;
        const uint32_t dord_addr = get_arg_val<uint32_t>(15);
        cb_reserve_back(3, 1);
        const uint32_t B3 = (get_write_ptr(3) + 63) & ~63u;
        SeDyn& dy = *reinterpret_cast<SeDyn*>(B3);
        static_assert(sizeof(SeDyn) <= 4096, "the schedule scratch");
        uint16_t* gs = reinterpret_cast<uint16_t*>(B3 + 4096);  // [NG] rows per expert (this pass)
        uint16_t* hb = gs + NG;                                 // [NG] rows of homes before mine
        uint16_t* run = hb + NG;                                // [NG] my tokens' rows so far
        uint32_t* sch = reinterpret_cast<uint32_t*>(B3 + 4096 + ((6 * NG + 63) / 64) * 64);  // (off, cnt) per entry
        uint32_t* tf = sch + 2 * N * VM;                        // predicted finish per entry
        uint32_t* nact = tf + N * VM;
        uint8_t* seid = reinterpret_cast<uint8_t*>(nact + 32);
        uint8_t* vfirst = seid + ((N * VM + 63) / 64) * 64;     // [N EPC] an expert's first entry (0xFF: none)
        uint8_t* vnext = vfirst + ((N * EPC + 63) / 64) * 64;   // [N VM] the expert's next entry
        uint8_t* ord = vnext + ((N * VM + 63) / 64) * 64;       // [A pages][DPG]
        for (uint32_t i = 0; i < 3 * NG; ++i) gs[i] = 0;
        constexpr uint32_t T1 = T0 + S < SF ? T0 + S : SF;
        const id_t* idn = const_cast<const id_t*>(ids);
        for (uint32_t o = 0; o < N; ++o) {
            for (uint32_t i = (o * SF + T0) * K, i1 = (o * SF + T1) * K; i < i1; ++i) {
                const uint32_t g = idn[i] - EOFF;
                if (g < NG) {
                    ++gs[g];
                    if (o < ME) ++hb[g];
                }
            }
        }
        for (uint32_t i = 0; i < N * EPC; ++i) vfirst[i] = 0xFF;
        for (uint32_t s_ = 0; s_ < N; ++s_) {
            se_dyn_from_counts(dy, EPC, SE_RPS, [&](uint32_t e) { return static_cast<uint32_t>(gs[s_ * EPC + e]); });
            nact[s_] = dy.n_act;
            uint32_t fin = 0;
            for (uint32_t v = dy.n_act; v-- > 0;) {  // (lists in schedule order)
                vnext[s_ * VM + v] = vfirst[s_ * EPC + dy.eid[v]];
                vfirst[s_ * EPC + dy.eid[v]] = static_cast<uint8_t>(v);
            }
            for (uint32_t v = 0; v < dy.n_act; ++v) {
                seid[s_ * VM + v] = dy.eid[v];
                sch[2 * (s_ * VM + v)] = dy.off[v];
                sch[2 * (s_ * VM + v) + 1] = dy.cnt[v];
                fin += dy.cnt[v] > WR ? dy.cnt[v] : WR;
                tf[s_ * VM + v] = fin;
            }
        }
        for (uint32_t i = 0; i < A * DPG; ++i) ord[i] = 0xFF;
        for (uint32_t t = 0; t < S && T0 + t < SF; ++t) {
            uint8_t* oa = ord + (t % A) * DPG + (t / A) * K;
            const uint32_t base = (ME * SF + T0 + t) * K;
            uint32_t kt[K], nk = 0;
            uint8_t ks[K];
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t g = idn[base + k] - EOFF;
                if (g >= NG) continue;
                const uint32_t r = hb[g] + run[g]++;
                const uint32_t s_ = g / EPC;
                for (uint32_t v = vfirst[g]; v != 0xFF; v = vnext[s_ * VM + v]) {
                    const uint32_t o_ = sch[2 * (s_ * VM + v)];
                    if (r >= o_ && r < o_ + sch[2 * (s_ * VM + v) + 1]) {
                        const uint32_t key = tf[s_ * VM + v];
                        uint32_t i = nk++;
                        for (; i > 0 && (kt[i - 1] > key || (kt[i - 1] == key && ks[i - 1] > s_)); --i) {
                            kt[i] = kt[i - 1];
                            ks[i] = ks[i - 1];
                        }
                        kt[i] = key;
                        ks[i] = static_cast<uint8_t>(s_);
                        break;
                    }
                }
            }
            for (uint32_t i = 0; i < nk; ++i) oa[i] = ks[i];
        }
        const InterleavedAddrGen<true> og = {.bank_base_address = dord_addr, .page_size = DPG};
        for (uint32_t a_ = 0; a_ < A; ++a_) noc_async_write(reinterpret_cast<uint32_t>(ord + a_ * DPG), get_noc_addr(a_, og), DPG);
    }
#endif
    // out
    const InterleavedAddrGen<true> cg = {.bank_base_address = cnt_addr, .page_size = NG * 4};
    const InterleavedAddrGen<true> rg = {.bank_base_address = reg_addr, .page_size = NG * 4};
    const InterleavedAddrGen<true> ig = {.bank_base_address = tidx_addr, .page_size = ROWS_CAP * 4};
    const InterleavedAddrGen<true> tg = {.bank_base_address = tab_addr, .page_size = 256};
    const InterleavedAddrGen<true> pg = {.bank_base_address = plan_addr, .page_size = 4096};
    noc_async_write(cnt_l1, get_noc_addr(0, cg), NG * 4);
    noc_async_write(reg_l1, get_noc_addr(0, rg), NG * 4);
    noc_async_write(tidx_l1, get_noc_addr(0, ig), ROWS_CAP * 4);
    for (uint32_t p = 0; p < ROWS_CAP / 32; ++p) noc_async_write(tab_l1 + p * 256, get_noc_addr(p, tg), 256);
    noc_async_write(plan_l1, get_noc_addr(0, pg), 4096);
    noc_async_write_barrier();
}
"""


# Multi-core planner: M cores (core p): phase 1 each scans routing chunk p (CH entries) and buckets its local-expert
# entries by expert group (GS experts per group, group q placed by core q) into compact lists (4 bytes: chunk index |
# expert in group << 13 | gate weight bf16 << 16), counts per expert, and a 1/M slice of this chip's home tokens' rows
# (n_local / want / n_ret partials); lists + counts go to each group's core, partials to core 0. Phase 2 (once every
# chunk's list landed) core q totals the counts (regions), places its group's rows in scan order (chunk by chunk) and
# writes its row range (32-row aligned) of the token index and the table straight to DRAM; core 0 sums the partials and
# the n_inj counts and writes counts / regions / plan.
_PLANNER_MC = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t ME = get_compile_time_arg_val(0), R = get_compile_time_arg_val(1), C = get_compile_time_arg_val(2);
constexpr uint32_t S = get_compile_time_arg_val(3), K = get_compile_time_arg_val(4), EPC = get_compile_time_arg_val(5);
constexpr uint32_t NG = get_compile_time_arg_val(6), A = get_compile_time_arg_val(7), L = get_compile_time_arg_val(8);
constexpr uint32_t P = get_compile_time_arg_val(9), ROWS_CAP = get_compile_time_arg_val(10);
constexpr uint32_t M = get_compile_time_arg_val(11), CH = get_compile_time_arg_val(12), GS = get_compile_time_arg_val(13);
constexpr uint32_t HS = get_compile_time_arg_val(14);
constexpr uint32_t N = R * C, T = N * S * K, SK = S * K;
constexpr uint32_t al64(uint32_t v) { return (v + 63) / 64 * 64; }
// L1 layout (the same on every core)
constexpr uint32_t O_IDS = 0, O_WTS = O_IDS + al64(CH * 2), O_HOME = O_WTS + al64(CH * 2);
constexpr uint32_t O_OUT = O_HOME + al64(HS * 2);                 // M lists x CH entries
constexpr uint32_t HDR = al64((1 + EPC) * 4);                     // a list header: length, counts[EPC]
constexpr uint32_t O_HDR = O_OUT + M * CH * 4;                    // M outgoing headers
constexpr uint32_t PART = al64(544 * 4);                          // a home partial (plan words 0..544)
constexpr uint32_t O_PART = O_HDR + M * HDR;
constexpr uint32_t O_IN = O_PART + PART;                          // M incoming lists
constexpr uint32_t O_HIN = O_IN + M * CH * 4;                     // M incoming headers
constexpr uint32_t O_PIN = O_HIN + M * HDR;                       // core 0: M partials
constexpr uint32_t O_NIN = O_PIN + M * PART;                      // core 0: M n_inj (64 B each)
constexpr uint32_t O_NINJ = O_NIN + M * 64;                       // this core's n_inj
constexpr uint32_t O_TIDX = O_NINJ + 64, O_TAB = O_TIDX + al64(ROWS_CAP * 4);
constexpr uint32_t O_PLAN = O_TAB + al64(ROWS_CAP * 8), O_CNT = O_PLAN + 4096, O_REG = O_CNT + al64(NG * 4);
constexpr uint32_t O_END = O_REG + al64(NG * 4);
constexpr uint32_t WX = __WX__, WY = __WY__;  // torus axes (rings: the shorter way, ties Y- / X+ as the fabric)
constexpr uint32_t YT = __YT__;  // (SPC_YTIE: sparse_combine_l1.route(ytie=))
FORCE_INLINE uint32_t first_hop(uint32_t src, uint32_t dst) {
    const uint32_t sr = src / C, sc = src % C, dr = dst / C, dc = dst % C;
    if (sr != dr) {
        if constexpr (WY) {
            const uint32_t f2 = 2 * ((dr + R - sr) % R);  // (YT: the opposite row splits by column parity)
            return f2 < R || (YT && f2 == R && ((sc + dc) & 1)) ? 2 : 3;
        }
        return dr > sr ? 2 : 3;
    }
    if constexpr (WX) return 2 * ((dc + C - sc) % C) <= C ? 0 : 1;
    return dc > sc ? 0 : 1;
}
void kernel_main() {
    size_t a = 0;
    const uint32_t ids_addr = get_arg_val<uint32_t>(a++), w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t cnt_addr = get_arg_val<uint32_t>(a++), reg_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tidx_addr = get_arg_val<uint32_t>(a++), tab_addr = get_arg_val<uint32_t>(a++);
    const uint32_t plan_addr = get_arg_val<uint32_t>(a++);
    uint32_t lut[8];
    for (uint32_t i = 0; i < 8; ++i) lut[i] = get_arg_val<uint32_t>(a++);
    const uint32_t p = get_arg_val<uint32_t>(a++);
    uint32_t px[M], py[M];
    for (uint32_t i = 0; i < M; ++i) {
        px[i] = get_arg_val<uint32_t>(a++);
        py[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t tdbg = get_arg_val<uint32_t>(a++);
    uint32_t ts[16];
    ts[0] = get_timestamp_32b();
    cb_reserve_back(0, 1);
    const uint32_t B = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t sem_l = get_semaphore(0), sem_c = get_semaphore(1);
    // in: my routing chunk and my slice of my home tokens' rows
    const uint32_t c0 = p * CH, chl = c0 >= T ? 0 : (T - c0 < CH ? T - c0 : CH);
    const InterleavedAddrGen<true> ids_g = {.bank_base_address = ids_addr, .page_size = T * 2};
    const InterleavedAddrGen<true> w_g = {.bank_base_address = w_addr, .page_size = T * 2};
    if (chl) {
        noc_async_read(get_noc_addr(0, ids_g) + c0 * 2, B + O_IDS, al64(chl * 2));
        noc_async_read(get_noc_addr(0, w_g) + c0 * 2, B + O_WTS, al64(chl * 2));
    }
    const uint32_t h0 = ME * SK + p * HS;
    noc_async_read(get_noc_addr(0, ids_g) + h0 * 2, B + O_HOME, al64(HS * 2));
    noc_async_read_barrier();
    invalidate_l1_cache();
    ts[1] = get_timestamp_32b();
    const uint16_t* ids = reinterpret_cast<const uint16_t*>(B + O_IDS);
    const uint16_t* wts = reinterpret_cast<const uint16_t*>(B + O_WTS);
    const uint16_t* hid = reinterpret_cast<const uint16_t*>(B + O_HOME);
    uint32_t* out = reinterpret_cast<uint32_t*>(B + O_OUT);
    constexpr uint32_t E0 = ME * EPC;
    // phase 1: bucket my chunk's local-expert entries by group
    uint32_t len[M], cnt[EPC];
    for (uint32_t q = 0; q < M; ++q) len[q] = 0;
    for (uint32_t e = 0; e < EPC; ++e) cnt[e] = 0;
    for (uint32_t ci = 0; ci < chl; ++ci) {
        const uint32_t e = static_cast<uint32_t>(ids[ci]) - E0;
        if (e < EPC) {
            const uint32_t q = e / GS;
            out[q * CH + len[q]++] = ci | ((e - q * GS) << 13) | (static_cast<uint32_t>(wts[ci]) << 16);
            ++cnt[e];
        }
    }
    ts[2] = get_timestamp_32b();
    // my slice of my home tokens' rows: n_local / want / n_ret partials (plan words)
    uint32_t* part = reinterpret_cast<uint32_t*>(B + O_PART);
    for (uint32_t i = 0; i < 544; ++i) part[i] = 0;
    for (uint32_t hi = 0; hi < HS; ++hi) {
        const uint32_t idx = p * HS + hi, t = idx / K, src = hid[hi] / EPC, l = (t % A + src + ME) % L;
        if (src == ME) {
            ++part[16 + t % A];
        } else {
            ++part[32 + (t % A) * 32 + src];
            ++part[8 + lut[first_hop(ME, src) * 2 + l]];
        }
    }
    ts[3] = get_timestamp_32b();
    // send: each group's list + header to its core, the partial to core 0
    for (uint32_t q = 0; q < M; ++q) {
        uint32_t* h = reinterpret_cast<uint32_t*>(B + O_HDR + q * HDR);
        h[0] = len[q];
        for (uint32_t e = 0; e < EPC; ++e) h[1 + e] = cnt[e];
        if (len[q]) {
            noc_async_write(B + O_OUT + q * CH * 4, get_noc_addr(px[q], py[q], B + O_IN + p * CH * 4), (len[q] * 4 + 15) & ~15u);
        }
        noc_async_write(B + O_HDR + q * HDR, get_noc_addr(px[q], py[q], B + O_HIN + p * HDR), HDR);
    }
    noc_async_write(B + O_PART, get_noc_addr(px[0], py[0], B + O_PIN + p * PART), PART);
    noc_async_write_barrier();
    for (uint32_t q = 0; q < M; ++q) noc_semaphore_inc(get_noc_addr(px[q], py[q], sem_l), 1);
    noc_semaphore_inc(get_noc_addr(px[0], py[0], sem_c), 1);
    // phase 2: totals, regions; place my group's rows in scan order
    ts[4] = get_timestamp_32b();
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l), M);
    invalidate_l1_cache();
    ts[5] = get_timestamp_32b();
    uint32_t reg[EPC], tot[EPC], off = 0;
    for (uint32_t e = 0; e < EPC; ++e) {
        uint32_t s_ = 0;
        for (uint32_t j = 0; j < M; ++j) s_ += reinterpret_cast<const uint32_t*>(B + O_HIN + j * HDR)[1 + e];
        tot[e] = s_;
        reg[e] = off;
        off += (s_ + 31) / 32 * 32;
    }
    const uint32_t g0 = p * GS, g1 = (p + 1) * GS < EPC ? (p + 1) * GS : EPC;
    uint32_t* ninj = reinterpret_cast<uint32_t*>(B + O_NINJ);
    for (uint32_t i = 0; i < 16; ++i) ninj[i] = 0;
    if (g0 < g1) {
        const uint32_t r0 = reg[g0], r1 = reg[g1 - 1] + (tot[g1 - 1] + 31) / 32 * 32;
        uint32_t* tidx = reinterpret_cast<uint32_t*>(B + O_TIDX);
        uint32_t* tab = reinterpret_cast<uint32_t*>(B + O_TAB);
        for (uint32_t r = 0; r < r1 - r0; ++r) tidx[r] = 0;
        uint32_t fill[GS];
        for (uint32_t e = g0; e < g1; ++e) {
            fill[e - g0] = reg[e] - r0;
            for (uint32_t r = reg[e] + tot[e]; r < reg[e] + (tot[e] + 31) / 32 * 32; ++r) {  // (padding rows: zero)
                tab[2 * (r - r0)] = tab[2 * (r - r0) + 1] = 0;
            }
        }
        for (uint32_t j = 0; j < M; ++j) {
            const uint32_t n = reinterpret_cast<const uint32_t*>(B + O_HIN + j * HDR)[0];
            const uint32_t* lst = reinterpret_cast<const uint32_t*>(B + O_IN + j * CH * 4);
            for (uint32_t x = 0; x < n; ++x) {
                const uint32_t v = lst[x], i = j * CH + (v & 0x1FFF), eg = (v >> 13) & 7, w = v >> 16;
                const uint32_t o = i / SK, rem = i - o * SK, t = rem / K, k = rem - t * K;
                const uint32_t r = fill[eg]++;
                if (r0 + r >= ROWS_CAP) continue;
                tidx[r] = o * S + t;
                uint32_t dest;
                if (o == ME) {
                    dest = P + t % A;
                } else {
                    dest = lut[first_hop(ME, o) * 2 + (t % A + ME + o) % L];
                    ++ninj[dest];
                }
                tab[2 * r] = dest | (w << 16);
                tab[2 * r + 1] = o | (t << 8) | ((g0 + eg) << 24);
            }
        }
        ts[6] = get_timestamp_32b();
        const uint32_t w1 = r1 < ROWS_CAP ? r1 : ROWS_CAP;
        if (w1 > r0) {
            const InterleavedAddrGen<true> ig = {.bank_base_address = tidx_addr, .page_size = ROWS_CAP * 4};
            const InterleavedAddrGen<true> tg = {.bank_base_address = tab_addr, .page_size = 256};
            noc_async_write(B + O_TIDX, get_noc_addr(0, ig) + r0 * 4, (w1 - r0) * 4);
            for (uint32_t pg = r0 / 32; pg < w1 / 32; ++pg) {
                noc_async_write(B + O_TAB + (pg - r0 / 32) * 256, get_noc_addr(pg, tg), 256);
            }
        }
    }
    noc_async_write(B + O_NINJ, get_noc_addr(px[0], py[0], B + O_NIN + p * 64), 64);
    noc_async_write_barrier();
    ts[7] = get_timestamp_32b();
    noc_semaphore_inc(get_noc_addr(px[0], py[0], sem_c), 1);
    // phase 3 (core 0): counts, regions, plan
    if (p == 0) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_c), 2 * M);
        invalidate_l1_cache();
        uint32_t* plan = reinterpret_cast<uint32_t*>(B + O_PLAN);
        uint32_t* cg_ = reinterpret_cast<uint32_t*>(B + O_CNT);
        uint32_t* rg_ = reinterpret_cast<uint32_t*>(B + O_REG);
        for (uint32_t i = 0; i < 1024; ++i) plan[i] = 0;
        for (uint32_t j = 0; j < M; ++j) {
            const uint32_t* pa = reinterpret_cast<const uint32_t*>(B + O_PIN + j * PART);
            for (uint32_t i = 8; i < 544; ++i) plan[i] += pa[i];
            const uint32_t* ni = reinterpret_cast<const uint32_t*>(B + O_NIN + j * 64);
            for (uint32_t i = 0; i < 8; ++i) plan[i] += ni[i];
        }
        plan[600] = off;
        plan[601] = off > ROWS_CAP ? 1 : 0;
        for (uint32_t i = 0; i < NG; ++i) cg_[i] = rg_[i] = 0;
        for (uint32_t e = 0; e < EPC; ++e) {
            cg_[E0 + e] = tot[e];
            rg_[E0 + e] = reg[e];
        }
        const InterleavedAddrGen<true> cg = {.bank_base_address = cnt_addr, .page_size = NG * 4};
        const InterleavedAddrGen<true> rg = {.bank_base_address = reg_addr, .page_size = NG * 4};
        const InterleavedAddrGen<true> pg = {.bank_base_address = plan_addr, .page_size = 4096};
        noc_async_write(B + O_CNT, get_noc_addr(0, cg), NG * 4);
        noc_async_write(B + O_REG, get_noc_addr(0, rg), NG * 4);
        noc_async_write(B + O_PLAN, get_noc_addr(0, pg), 4096);
        noc_async_write_barrier();
    }
    ts[8] = get_timestamp_32b();
    if (tdbg) {  // (phase timestamps: start, read, scan, home, sent, lists in, placed, rows out, end)
        uint32_t* d = reinterpret_cast<uint32_t*>(B + O_NINJ);
        for (uint32_t i = 0; i < 9; ++i) d[i] = ts[i];
        const InterleavedAddrGen<true> dg = {.bank_base_address = tdbg, .page_size = 64};
        noc_async_write(B + O_NINJ, get_noc_addr(p, dg), 64);
        noc_async_write_barrier();
    }
    // (an atomic still unacknowledged when the kernel returns can land after the next program's launch resets its
    // NoC counters: that program's atomic barrier then never completes - seen as hangs under back-to-back traces)
    noc_async_atomic_barrier();
}
"""


def _mc_l1_bytes(M, CH, HS, EPC, NG, rows_cap):
    al = lambda v: (v + 63) // 64 * 64  # noqa: E731
    HDR, PART = al((1 + EPC) * 4), al(544 * 4)
    return (
        2 * al(CH * 2)
        + al(HS * 2)
        + M * CH * 4
        + M * HDR
        + PART
        + M * CH * 4
        + M * HDR
        + M * PART
        + M * 64
        + 64
        + al(rows_cap * 4)
        + al(rows_cap * 8)
        + 4096
        + 2 * al(NG * 4)
        + 64
    )


# Multi-core planner v2 (M cores, core p): (1) counting-sort my routing chunk's local-expert entries by expert into a
# bucket, each entry already its final table words (destination | gate weight << 16, header), plus n_inj partials;
# send my per-expert counts to every core. (2) every core totals the counts (regions) and knows every chunk's base row per
# expert; core p takes 1/M of the row tiles and pulls exactly the matching entry slices from each chunk's bucket (NoC
# reads from that core's L1: a hot expert is split over all cores), writes its rows of the token index / table to DRAM.
# (3) n_inj and this chip's home-token counts (n_local / want / n_ret, a 1/M slice per core) go to core 0's plan as
# atomic increments; core 0 writes counts / regions / plan.
_PLANNER_MC2 = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
__DETINC__
constexpr uint32_t ME = get_compile_time_arg_val(0), R = get_compile_time_arg_val(1), C = get_compile_time_arg_val(2);
constexpr uint32_t S = get_compile_time_arg_val(3), K = get_compile_time_arg_val(4), EPC = get_compile_time_arg_val(5);
constexpr uint32_t NG = get_compile_time_arg_val(6), A = get_compile_time_arg_val(7), L = get_compile_time_arg_val(8);
constexpr uint32_t P = get_compile_time_arg_val(9), ROWS_CAP = get_compile_time_arg_val(10);
constexpr uint32_t M = get_compile_time_arg_val(11), CH = get_compile_time_arg_val(12), HS = get_compile_time_arg_val(13);
constexpr uint32_t N = R * C, T = N * S * K, SK = S * K;
constexpr uint32_t al64(uint32_t v) { return (v + 63) / 64 * 64; }
constexpr uint32_t gcd_(uint32_t a, uint32_t b) { return b ? gcd_(b, a % b) : a; }
constexpr uint32_t CNT = al64(2 * EPC * 4);  // a chunk's counts per expert, then its bucket prefix per expert
// token-major rows (TM): core p plans tokens [p TPC, (p + 1) TPC) of every home (its ids / weights one run per home,
// 64 B of slack for a run's unaligned start), so an expert's rows come out in (token, home, k) order
constexpr uint32_t TM = __TM__, TPC = S / M, RUNB = al64(TPC * K * 2) + 64;
constexpr uint32_t IDB = TM ? N * RUNB : al64(CH * 2);
constexpr uint32_t O_IDS = 0, O_WTS = O_IDS + IDB, O_HOME = O_WTS + IDB;
constexpr uint32_t O_LST = O_HOME + al64(SK * 2);           // my local entries: chunk index << 8 | expert
constexpr uint32_t O_BUCK = O_LST + al64(CH * 4);           // CH entries x 8 bytes, sorted by expert
constexpr uint32_t O_MYC = O_BUCK + al64(__BUCKB__);        // my counts (sent to everyone) - the bucket region also
                                                            // stages the row pages (64 B a token: more than CH x 8 at K < 8)
constexpr uint32_t O_CIN = O_MYC + CNT;                     // M x counts
constexpr uint32_t STG = get_compile_time_arg_val(14);      // slice staging bytes (all of a core's slices at once)
constexpr uint32_t XP = al64((M + 2) * 4);                  // per expert: total, region, then each chunk's first row
constexpr uint32_t O_XP = O_CIN + M * CNT;                  // EPC x XP: filled by the experts' owner cores
constexpr uint32_t O_STG = O_XP + EPC * XP;
constexpr uint32_t O_PART = O_STG + al64(STG);              // home partial (plan words 0..544)
// (a core's rows [R0, R1): the plan's 32-row tiles split M ways - at most INSTR, the worst case's share; the plan
// tensors in DRAM hold ROWS_CAP, every row a chip can get)
constexpr uint32_t INSTR = (ROWS_CAP / 32 + M - 1) / M * 32;
constexpr uint32_t O_TIDX = O_PART + al64(544 * 4), O_TAB = O_TIDX + al64(INSTR * 4);
constexpr uint32_t O_PLAN = O_TAB + al64(INSTR * 8), O_CNT = O_PLAN + 4096, O_REG = O_CNT + al64(NG * 4);
constexpr uint32_t O_DBG = O_REG + al64(NG * 4);  // 64 bytes of timestamps, then 512 slice records
// deterministic combine (DET): per owned accumulator, per home token, the source chips of its rows in a fixed order -
// the order the accumulator adds them in, whatever the arrival timing (combine_plan docstring)
constexpr uint32_t DET = __DET__, VM = __VM__, OWN = (A + M - 1) / M, GPB = al64(NG * 4);
constexpr uint32_t DMAXT = (S + A - 1) / A, OPG = al64(DMAXT * K), DPG = __DPG__, WR = __WR__;
constexpr uint32_t O_GP = O_DBG + al64(64 + 12 * 512), O_GIN = O_GP + GPB, O_GS = O_GIN + M * GPB;
constexpr uint32_t O_RUN = O_GS + al64(NG * 8), O_SCH = O_RUN + al64(NG * 2), O_SEID = O_SCH + al64(N * VM * 8);
constexpr uint32_t O_NACT = O_SEID + al64(N * VM), O_ORD = O_NACT + al64(N * 4), O_DYN = O_ORD + OWN * OPG;
constexpr uint32_t O_TF = O_DYN + 4096;
constexpr uint32_t WX = __WX__, WY = __WY__;  // torus axes (rings: the shorter way, ties Y- / X+ as the fabric)
constexpr uint32_t YT = __YT__;  // (SPC_YTIE: sparse_combine_l1.route(ytie=))
FORCE_INLINE uint32_t first_hop(uint32_t src, uint32_t dst) {
    const uint32_t sr = src / C, sc = src % C, dr = dst / C, dc = dst % C;
    if (sr != dr) {
        if constexpr (WY) {
            const uint32_t f2 = 2 * ((dr + R - sr) % R);  // (YT: the opposite row splits by column parity)
            return f2 < R || (YT && f2 == R && ((sc + dc) & 1)) ? 2 : 3;
        }
        return dr > sr ? 2 : 3;
    }
    if constexpr (WX) return 2 * ((dc + C - sc) % C) <= C ? 0 : 1;
    return dc > sc ? 0 : 1;
}
void kernel_main() {
    size_t a = 0;
    const uint32_t ids_addr = get_arg_val<uint32_t>(a++), w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t cnt_addr = get_arg_val<uint32_t>(a++), reg_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tidx_addr = get_arg_val<uint32_t>(a++), tab_addr = get_arg_val<uint32_t>(a++);
    const uint32_t plan_addr = get_arg_val<uint32_t>(a++);
    uint32_t lut[8];
    for (uint32_t i = 0; i < 8; ++i) lut[i] = get_arg_val<uint32_t>(a++);
    const uint32_t p = get_arg_val<uint32_t>(a++);
    uint32_t px[M], py[M];
    for (uint32_t i = 0; i < M; ++i) {
        px[i] = get_arg_val<uint32_t>(a++);
        py[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t tdbg = get_arg_val<uint32_t>(a++);
    // (the planner cores' rectangle, NoC 0: unused - multicast exchanges from every core at once serialize, 3.6x slower)
    const uint32_t mx0 = get_arg_val<uint32_t>(a++), my0 = get_arg_val<uint32_t>(a++);
    const uint32_t mx1 = get_arg_val<uint32_t>(a++), my1 = get_arg_val<uint32_t>(a++);
    (void)mx0, (void)my0, (void)mx1, (void)my1;
    const uint32_t dord_addr = DET ? get_arg_val<uint32_t>(a++) : 0;  // (DET: the order table, a page per accumulator)
    uint32_t ts[16];
    ts[0] = get_timestamp_32b();
    cb_reserve_back(0, 1);
    const uint32_t B = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t sem_l = get_semaphore(0), sem_c = get_semaphore(1);
    uint32_t* plan = reinterpret_cast<uint32_t*>(B + O_PLAN);
    if (p == 0) {
        for (uint32_t i = 0; i < 1024; ++i) plan[i] = 0;  // (before my counts go out: the others' atomics follow them)
    }
    const uint32_t c0 = p * CH, chl = c0 >= T ? 0 : (T - c0 < CH ? T - c0 : CH);
    // the routing: one page of all T entries, or one page per token row ([N S, K] row-major, RP = 2 K bytes), or
    // (JW) one page per token of its K ids then its K weights ([N S, 2 K] uint16, RP = 4 K bytes; wts = ids)
    constexpr uint32_t RP = __RPAGE__, JW = __RJOINT__, RB = 2 * K;  // (RB: one row's ids, or weights, in bytes)
    constexpr uint32_t PT = __RPT__;  // tokens per page (joint rows: PT rows of RP bytes a page; 1: a page a row)
    // (staging stride: 64 B a row with one row a page; RP with multi-token pages - a run is one read that keeps its
    // source's 64 B alignment, so it starts SH rows into the staging when its first row is not 64 B aligned: rows come
    // back to a 64 B boundary every RA = lcm(64, RP) / RP rows - 2 at K 8, 8 at K 6 (DeepSeek V4's 24 B rows))
    constexpr uint32_t SR = PT == 1 ? 64 : (RP ? RP : 64);
    constexpr uint32_t RA = 64 / gcd_(64, SR);
    static_assert(PT == 1 || (JW && PT % RA == 0), "multi-token pages: joint rows only, whole 64 B runs a page");
    const InterleavedAddrGen<true> ids_g = {.bank_base_address = ids_addr, .page_size = RP ? RP * PT : T * 2};
    const InterleavedAddrGen<true> w_g = {.bank_base_address = w_addr, .page_size = RP ? RP * PT : T * 2};
    // (row pages land 64 bytes apart in local scratch, free until the sort: a DRAM read keeps its source's 64-byte
    // alignment in L1; then unpacked)
    constexpr uint32_t NPK = TM ? 2 * N + 1 : 3;
    uint32_t stg = B + O_BUCK, pk_src[NPK], pk_dst[NPK], pk_dw[NPK], pk_rows[NPK], npk = 0;
    // (returns where entry e0 lands: dst, or dst + e0's offset in its 64 B with one routing page)
    auto rd = [&](const InterleavedAddrGen<true>& g, uint32_t e0, uint32_t n, uint32_t dst, uint32_t dst_w,
                  bool home = false) -> uint32_t {
        if (home) stg = B + O_STG;  // (the home ids: in the slice staging, sized for them; the bucket region's tail
                                    // is the other cores' counts exchange, written any time)
        if constexpr (RP == 0) {
            const uint32_t d0 = TM ? (e0 * 2) & 63 : 0;  // (TM: a home's run may start mid 64 B)
            noc_async_read(get_noc_addr(0, g) + e0 * 2 - d0, dst, al64(n * 2 + d0));
            return dst + d0;
        } else {
            const uint32_t r0 = e0 / K, rows = n / K, sh = r0 % RA;
            if constexpr (PT == 1) {
                for (uint32_t r = 0; r < rows; ++r) noc_async_read(get_noc_addr(r0 + r, g), stg + r * 64, RP);
            } else {  // (whole runs of rows per page: one read each, rows SR apart)
                for (uint32_t r = r0; r < r0 + rows;) {
                    const uint32_t first = r % PT, cnt = PT - first < r0 + rows - r ? PT - first : r0 + rows - r;
                    noc_async_read(get_noc_addr(r / PT, g) + first * RP, stg + (r - r0 + sh) * SR, cnt * RP);
                    r += cnt;
                }
            }
            pk_src[npk] = stg + sh * SR, pk_dst[npk] = dst, pk_dw[npk] = dst_w, pk_rows[npk++] = rows;
            stg += (rows + sh) * SR;
            stg = (stg + 63) & ~63u;
            return dst;
        }
    };
    uint32_t runi[TM ? N : 1], runw[TM ? N : 1];  // (TM: each home's ids / weights for my tokens)
    if constexpr (TM) {
        for (uint32_t o = 0; o < N; ++o) {
            const uint32_t e0 = (o * S + p * TPC) * K;
            runi[o] = rd(ids_g, e0, TPC * K, B + O_IDS + o * RUNB, JW ? B + O_WTS + o * RUNB : 0);
            runw[o] = JW ? B + O_WTS + o * RUNB : rd(w_g, e0, TPC * K, B + O_WTS + o * RUNB, 0);
        }
    } else if (chl) {
        rd(ids_g, c0, chl, B + O_IDS, JW ? B + O_WTS : 0);
        if constexpr (!JW) rd(w_g, c0, chl, B + O_WTS, 0);
    }
    if (p < A) rd(ids_g, ME * SK, SK, B + O_HOME, 0, true);  // (owners only)
    noc_async_read_barrier();
    invalidate_l1_cache();
    if constexpr (RP != 0) {
        // (RP == 0 ||: a static_assert in a discarded if-constexpr branch of a non-template is still checked)
        static_assert(RP == 0 || (RP <= 64 && RB % 4 == 0 && RP == (JW ? 2 : 1) * RB), "row pages");
        // (unpacked by local NoC copies, one per row and array: scalar L1 copies cost ~50 cycles a word here; a row
        // under 16 B - K < 8, MiniMax M3's top-4 - by words: a NoC copy needs source and destination equally 16 B
        // aligned, and the packed destination rows alternate)
        for (uint32_t q = 0; q < npk; ++q) {
            for (uint32_t r = 0; r < pk_rows[q]; ++r) {
                if constexpr (RB % 16 == 0) {
                    noc_async_read(get_noc_addr(pk_src[q] + r * SR), pk_dst[q] + r * RB, RB);
                    if (pk_dw[q]) noc_async_read(get_noc_addr(pk_src[q] + r * SR + RB), pk_dw[q] + r * RB, RB);
                } else {
                    const uint32_t* sw = reinterpret_cast<const uint32_t*>(pk_src[q] + r * SR);
                    uint32_t* di = reinterpret_cast<uint32_t*>(pk_dst[q] + r * RB);
                    for (uint32_t i = 0; i < RB / 4; ++i) di[i] = sw[i];
                    if (pk_dw[q]) {
                        uint32_t* dw = reinterpret_cast<uint32_t*>(pk_dw[q] + r * RB);
                        for (uint32_t i = 0; i < RB / 4; ++i) dw[i] = sw[RB / 4 + i];
                    }
                }
            }
        }
        noc_async_read_barrier();
    }
    ts[1] = get_timestamp_32b();
    constexpr uint32_t E0 = ME * EPC;
    const uint32_t* ids2 = reinterpret_cast<const uint32_t*>(B + O_IDS);  // two ids per word
    const uint16_t* wts = reinterpret_cast<const uint16_t*>(B + O_WTS);
    // the first-hop port toward each chip, per link
    uint8_t hop[N * L];
    for (uint32_t o = 0; o < N; ++o) {
        for (uint32_t l = 0; l < L; ++l) hop[o * L + l] = o == ME ? 0xFF : lut[first_hop(ME, o) * 2 + l];
    }
    // (1) count my chunk's local entries (two ids per word) and list them; then place them (counting sort by expert),
    // each entry already the final table words
    uint32_t cnt[EPC], bo[EPC];
    for (uint32_t e = 0; e < EPC; ++e) cnt[e] = 0;
    uint32_t* lst = reinterpret_cast<uint32_t*>(B + O_LST);
    uint32_t nl = 0;
    auto take = [&](uint32_t v, uint32_t i) __attribute__((always_inline)) {
        const uint32_t e0 = (v & 0xFFFF) - E0, e1 = (v >> 16) - E0;
        if (e0 < EPC) {
            ++cnt[e0];
            lst[nl++] = (2 * i) << 8 | e0;
        }
        if (e1 < EPC) {
            ++cnt[e1];
            lst[nl++] = (2 * i + 1) << 8 | e1;
        }
    };
    uint32_t i4 = 0;
    if constexpr (TM) {  // (chunk index (token, home, k): what the placement decodes)
        for (uint32_t tt = 0; tt < TPC; ++tt) {
            for (uint32_t o = 0; o < N; ++o) {
                const uint32_t ci0 = (tt * N + o) * K;
                if constexpr (K % 2 == 0) {  // (two ids a word: a run starts 4-byte aligned with K even)
                    const uint32_t* r2 = reinterpret_cast<const uint32_t*>(runi[o] + tt * K * 2);
                    for (uint32_t kk = 0; kk < K; kk += 2) {
                        const uint32_t v = r2[kk / 2], e0 = (v & 0xFFFF) - E0, e1 = (v >> 16) - E0;
                        if (e0 < EPC) {
                            ++cnt[e0];
                            lst[nl++] = (ci0 + kk) << 8 | e0;
                        }
                        if (e1 < EPC) {
                            ++cnt[e1];
                            lst[nl++] = (ci0 + kk + 1) << 8 | e1;
                        }
                    }
                } else {
                    const uint16_t* r_ = reinterpret_cast<const uint16_t*>(runi[o]) + tt * K;
                    for (uint32_t kk = 0; kk < K; ++kk) {
                        const uint32_t e = r_[kk] - E0;
                        if (e < EPC) {
                            ++cnt[e];
                            lst[nl++] = (ci0 + kk) << 8 | e;
                        }
                    }
                }
            }
        }
        i4 = chl / 2;  // (the pairwise scan below: nothing left)
    }
    for (; i4 + 4 <= chl / 2; i4 += 4) {  // (four loads in flight)
        const uint32_t v0 = ids2[i4], v1 = ids2[i4 + 1], v2 = ids2[i4 + 2], v3 = ids2[i4 + 3];
        take(v0, i4);
        take(v1, i4 + 1);
        take(v2, i4 + 2);
        take(v3, i4 + 3);
    }
    for (; i4 < chl / 2; ++i4) take(ids2[i4], i4);
    if (!TM && (chl & 1)) {
        const uint32_t e0 = (ids2[chl / 2] & 0xFFFF) - E0;
        if (e0 < EPC) {
            ++cnt[e0];
            lst[nl++] = (chl - 1) << 8 | e0;
        }
    }
    if constexpr (DET) {  // my chunk's rows per global expert (all homes, then homes before mine) to the owners
        uint16_t* gp = reinterpret_cast<uint16_t*>(B + O_GP);
        for (uint32_t i = 0; i < 2 * NG; ++i) gp[i] = 0;
        const uint16_t* id16 = reinterpret_cast<const uint16_t*>(B + O_IDS);
        for (uint32_t i = 0; i < chl; ++i) {
            const uint32_t g = id16[i];
            ++gp[g];
            if (c0 + i < ME * SK) ++gp[NG + g];
        }
        constexpr uint32_t NOWN = A < M ? A : M;
        for (uint32_t q = 0; q < NOWN; ++q) {
            noc_async_write(B + O_GP, get_noc_addr(px[q], py[q], B + O_GIN + p * GPB), GPB);
        }
        noc_async_write_barrier();
        for (uint32_t q = 0; q < NOWN; ++q) noc_semaphore_inc(get_noc_addr(px[q], py[q], get_semaphore(3)), 1);
    }
    uint32_t* myc = reinterpret_cast<uint32_t*>(B + O_MYC);
    for (uint32_t e = 0, o_ = 0; e < EPC; ++e) {
        bo[e] = o_;
        myc[EPC + e] = o_;
        o_ += cnt[e];
        myc[e] = cnt[e];
    }
    uint32_t* buck = reinterpret_cast<uint32_t*>(B + O_BUCK);
    uint32_t ninj[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t x = 0; x < nl; ++x) {
        const uint32_t v = lst[x], ci = v >> 8, e = v & 0xFF;
        uint32_t o, t, wv;
        if constexpr (TM) {
            const uint32_t q = ci / K, kk = ci - q * K, tt = q / N;
            o = q - tt * N;
            t = p * TPC + tt;
            wv = reinterpret_cast<const uint16_t*>(runw[o])[tt * K + kk];
        } else {
            const uint32_t gi = c0 + ci;
            o = gi / SK;
            t = (gi - o * SK) / K;
            wv = wts[ci];
        }
        const uint32_t pos = bo[e]++;
        uint32_t dest;
        if (o == ME) {
            dest = P + t % A;
        } else {
            const uint32_t l = (t % A + ME + o) % L;  // (fixed per (accumulator, source, home): one ring each)
            dest = hop[o * L + l];
            ++ninj[dest];
        }
        buck[2 * pos] = dest | (wv << 16);
        buck[2 * pos + 1] = o | (t << 8) | (e << 24);
    }
    ts[2] = get_timestamp_32b();
    for (uint32_t q = 0; q < M; ++q) {  // (unicast: 32 cores multicasting into one rectangle at once serialize)
        noc_async_write(B + O_MYC, get_noc_addr(px[q], py[q], B + O_CIN + p * CNT), CNT);
    }
    noc_async_write_barrier();
    for (uint32_t q = 0; q < M; ++q) noc_semaphore_inc(get_noc_addr(px[q], py[q], sem_l), 1);
    ts[3] = get_timestamp_32b();
    uint32_t* cg_ = reinterpret_cast<uint32_t*>(B + O_CNT);
    uint32_t* rg_ = reinterpret_cast<uint32_t*>(B + O_REG);
    if (p == 0) {  // (off the critical path: the others are still sorting)
        for (uint32_t i = 0; i < NG; ++i) cg_[i] = rg_[i] = 0;
    }
    // (2) all counts in; the owner of expert e (core e % M) totals it and its chunks' first rows, sends that to all
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l), M);
    invalidate_l1_cache();
    ts[4] = get_timestamp_32b();
    const uint32_t* cin = reinterpret_cast<const uint32_t*>(B + O_CIN);
    for (uint32_t e = p; e < EPC; e += M) {
        uint32_t* xp = reinterpret_cast<uint32_t*>(B + O_XP + e * XP);
        uint32_t s_ = 0;
        for (uint32_t j = 0; j < M; ++j) {
            xp[2 + j] = s_;  // (relative to the region start)
            s_ += cin[j * (CNT / 4) + e];
        }
        xp[0] = s_;
        for (uint32_t q = 0; q < M; ++q) {
            if (q != p) noc_async_write(B + O_XP + e * XP, get_noc_addr(px[q], py[q], B + O_XP + e * XP), XP);
        }
    }
    noc_async_write_barrier();
    for (uint32_t q = 0; q < M; ++q) noc_semaphore_inc(get_noc_addr(px[q], py[q], get_semaphore(2)), 1);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2)), M);
    invalidate_l1_cache();
    uint32_t tot[EPC], reg[EPC], off = 0;
    for (uint32_t e = 0; e < EPC; ++e) {
        const uint32_t s_ = reinterpret_cast<const uint32_t*>(B + O_XP + e * XP)[0];
        tot[e] = s_;
        reg[e] = off;
        off += (s_ + 31) / 32 * 32;
    }
    const uint32_t NT = (off < ROWS_CAP ? off : ROWS_CAP) / 32;  // (past the worst case: flagged, plan word 601)
    const uint32_t R0 = (p * NT / M) * 32, R1 = ((p + 1) * NT / M) * 32;
    uint32_t* tidx = reinterpret_cast<uint32_t*>(B + O_TIDX);
    uint32_t* tab = reinterpret_cast<uint32_t*>(B + O_TAB);
    // (slices: issue every read, one barrier, then copy)
    struct Sl {
        uint32_t lo, n, stg;
    };
    constexpr uint32_t MAXSL = 512;
    Sl* sl = reinterpret_cast<Sl*>(B + O_DBG + 64);
    uint32_t nsl = 0, sp = 0;
    for (uint32_t e = 0; e < EPC && R1 > R0; ++e) {
        const uint32_t e_lo = reg[e], e_end = reg[e] + tot[e], e_pad = reg[e] + (tot[e] + 31) / 32 * 32;
        if (e_pad <= R0 || e_lo >= R1) continue;
        for (uint32_t r = (e_end > R0 ? e_end : R0); r < (e_pad < R1 ? e_pad : R1); ++r) {  // padding rows
            tidx[r - R0] = 0;
            tab[2 * (r - R0)] = tab[2 * (r - R0) + 1] = 0;
        }
        const uint32_t* xp = reinterpret_cast<const uint32_t*>(B + O_XP + e * XP);
        for (uint32_t j = 0; j < M; ++j) {
            const uint32_t* cj = cin + j * (CNT / 4);
            const uint32_t base = e_lo + xp[2 + j], n = (j + 1 < M ? xp[3 + j] : tot[e]) - xp[2 + j];
            const uint32_t lo = base > R0 ? base : R0, hi = base + n < R1 ? base + n : R1;
            if (lo < hi) {
                const uint32_t bj = cj[EPC + e];  // e's first entry in chunk j's bucket
                const uint32_t src = B + O_BUCK + (bj + (lo - base)) * 8, cnt_ = hi - lo;
                const uint32_t bytes = ((src & 15) + cnt_ * 8 + 15) & ~15u;
                if (nsl == MAXSL || sp + bytes > STG) {  // (staging full: drain)
                    noc_async_read_barrier();
                    invalidate_l1_cache();
                    for (uint32_t q = 0; q < nsl; ++q) {
                        const uint32_t* st = reinterpret_cast<const uint32_t*>(sl[q].stg);
                        for (uint32_t x = 0; x < sl[q].n; ++x) {
                            const uint32_t r = sl[q].lo + x - R0, w0 = st[2 * x], w1 = st[2 * x + 1];
                            tab[2 * r] = w0;
                            tab[2 * r + 1] = w1;
                            tidx[r] = (w1 & 0xFF) * S + ((w1 >> 8) & 0xFFFF);
                        }
                    }
                    nsl = sp = 0;
                }
                noc_async_read(get_noc_addr(px[j], py[j], src & ~15u), B + O_STG + sp, bytes);
                sl[nsl++] = Sl{lo, cnt_, B + O_STG + sp + (src & 15)};
                sp += bytes;
            }
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t q = 0; q < nsl; ++q) {
        const uint32_t* st = reinterpret_cast<const uint32_t*>(sl[q].stg);
        for (uint32_t x = 0; x < sl[q].n; ++x) {
            const uint32_t r = sl[q].lo + x - R0, w0 = st[2 * x], w1 = st[2 * x + 1];
            tab[2 * r] = w0;
            tab[2 * r + 1] = w1;
            tidx[r] = (w1 & 0xFF) * S + ((w1 >> 8) & 0xFFFF);
        }
    }
    ts[5] = get_timestamp_32b();
    const uint32_t W1 = R1 < ROWS_CAP ? R1 : ROWS_CAP;
    if (W1 > R0) {
        const InterleavedAddrGen<true> ig = {.bank_base_address = tidx_addr, .page_size = ROWS_CAP * 4};
        const InterleavedAddrGen<true> tg = {.bank_base_address = tab_addr, .page_size = 256};
        noc_async_write(B + O_TIDX, get_noc_addr(0, ig) + R0 * 4, (W1 - R0) * 4);
        for (uint32_t pg = R0 / 32; pg < W1 / 32; ++pg) {
            noc_async_write(B + O_TAB + (pg - R0 / 32) * 256, get_noc_addr(pg, tg), 256);
        }
    }
    // my home tokens' rows: core q owns accumulators a = q, q + M, ... (their n_local / want in full); n_ret partial
    uint32_t nret[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t* part = reinterpret_cast<uint32_t*>(B + O_PART);  // per owned accumulator: [n_local, pad x 15, want x 32]
    uint32_t n_own = 0;
    {
        const uint16_t* hid = reinterpret_cast<const uint16_t*>(B + O_HOME);
        for (uint32_t aa = p; aa < A; aa += M, ++n_own) {
            uint32_t* pw = part + n_own * 48;
            uint32_t wl[32], nlo = 0;
            for (uint32_t i = 0; i < 32; ++i) wl[i] = 0;
            const uint32_t* hw = reinterpret_cast<const uint32_t*>(hid);
            const uint32_t la = aa + ME;  // (a row's link: (accumulator + source + home) % L)
            auto one = [&](uint32_t g) __attribute__((always_inline)) {
                const uint32_t src = g / EPC;
                if (src == ME) {
                    ++nlo;
                } else {
                    ++wl[src];
                    ++nret[hop[src * L + (la + src) % L]];
                }
            };
            for (uint32_t t = aa; t < S; t += A) {
                static_assert(L == 2 && N <= 32, "two links, want[32]");
                if constexpr (K % 4 == 0) {
                    for (uint32_t kk = 0; kk < K; kk += 4) {  // (two words, four ids, in flight)
                        const uint32_t w0 = hw[(t * K + kk) / 2], w1 = hw[(t * K + kk) / 2 + 1];
                        one(w0 & 0xFFFF);
                        one(w0 >> 16);
                        one(w1 & 0xFFFF);
                        one(w1 >> 16);
                    }
                } else {  // (top-k not a multiple of 4, e.g. DeepSeek V4.1's 6: one id at a time)
                    for (uint32_t kk = 0; kk < K; ++kk) {
                        one(hid[t * K + kk]);
                    }
                }
            }
            pw[0] = nlo;
            for (uint32_t i = 0; i < 32; ++i) pw[16 + i] = wl[i];
        }
    }
    ts[6] = get_timestamp_32b();
#if __DET__  // (preprocessor: SeDyn / SE_RPS exist only with det's include)
    {
        if (p < A) {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3)), M);
            invalidate_l1_cache();
            uint32_t* gs = reinterpret_cast<uint32_t*>(B + O_GS);  // [NG] rows per expert, [NG] of homes before mine
            for (uint32_t i = 0; i < 2 * NG; ++i) gs[i] = 0;
            for (uint32_t j = 0; j < M; ++j) {
                const uint16_t* gj = reinterpret_cast<const uint16_t*>(B + O_GIN + j * GPB);
                for (uint32_t i = 0; i < 2 * NG; ++i) gs[i] += gj[i];
            }
            // every source chip's schedule: the flat expert's own (se_dyn_from_counts on that chip's expert counts)
            uint32_t* sch = reinterpret_cast<uint32_t*>(B + O_SCH);  // per (source, entry): first row, rows
            uint8_t* seid = reinterpret_cast<uint8_t*>(B + O_SEID);
            uint32_t* nact = reinterpret_cast<uint32_t*>(B + O_NACT);
            static_assert(sizeof(SeDyn) <= 4096, "the schedule scratch");
            SeDyn& dy = *reinterpret_cast<SeDyn*>(B + O_DYN);
            // and each entry's predicted finish on its chip, in rows: an entry costs max(WR, its rows) (WR: an
            // expert's weight read in rows of compute), summed in schedule order
            uint32_t* tf = reinterpret_cast<uint32_t*>(B + O_TF);
            for (uint32_t s = 0; s < N; ++s) {
                se_dyn_from_counts(dy, EPC, SE_RPS, [&](uint32_t e) { return gs[s * EPC + e]; });
                nact[s] = dy.n_act;
                uint32_t fin = 0;
                for (uint32_t v = 0; v < dy.n_act; ++v) {
                    seid[s * VM + v] = dy.eid[v];
                    sch[2 * (s * VM + v)] = dy.off[v];
                    sch[2 * (s * VM + v) + 1] = dy.cnt[v];
                    fin += dy.cnt[v] > WR ? dy.cnt[v] : WR;
                    tf[s * VM + v] = fin;
                }
            }
            // my home tokens in order: a row's place in its expert's row list (rows are in (home, token) order: the
            // homes before mine, then my earlier tokens) picks the entry that carries it; a token's rows are ordered
            // by (predicted finish, source) - per source that is its production order (no deadlock: combine docstring)
            uint8_t* ord = reinterpret_cast<uint8_t*>(B + O_ORD);
            for (uint32_t i = 0; i < OWN * OPG; ++i) ord[i] = 0xFF;
            uint16_t* run = reinterpret_cast<uint16_t*>(B + O_RUN);
            for (uint32_t i = 0; i < NG; ++i) run[i] = 0;
            const uint16_t* hid2 = reinterpret_cast<const uint16_t*>(B + O_HOME);
            for (uint32_t t = 0; t < S; ++t) {
                const uint32_t aa = t % A;
                const bool mine = aa % M == p;
                uint8_t* oa = ord + (aa / M) * OPG + (t / A) * K;
                uint32_t kt[K], nk = 0;
                uint8_t ks[K];
                for (uint32_t kk = 0; kk < K; ++kk) {
                    const uint32_t g = hid2[t * K + kk];
                    const uint32_t r = gs[NG + g] + run[g]++;
                    if (!mine) continue;
                    const uint32_t s = g / EPC, e = g - s * EPC;
                    for (uint32_t v = 0; v < nact[s]; ++v) {
                        const uint32_t o_ = sch[2 * (s * VM + v)];
                        if (seid[s * VM + v] == e && r >= o_ && r < o_ + sch[2 * (s * VM + v) + 1]) {
                            const uint32_t key = tf[s * VM + v];
                            uint32_t i = nk++;
                            for (; i > 0 && (kt[i - 1] > key || (kt[i - 1] == key && ks[i - 1] > s)); --i) {
                                kt[i] = kt[i - 1];
                                ks[i] = ks[i - 1];
                            }
                            kt[i] = key;
                            ks[i] = static_cast<uint8_t>(s);
                            break;
                        }
                    }
                }
                for (uint32_t i = 0; i < nk; ++i) oa[i] = ks[i];
            }
            static_assert(OPG <= DPG, "the order table page");
            const InterleavedAddrGen<true> og = {.bank_base_address = dord_addr, .page_size = DPG};
            for (uint32_t aa = p; aa < A; aa += M) {
                noc_async_write(B + O_ORD + (aa / M) * OPG, get_noc_addr(aa, og), OPG);
            }
            noc_async_write_barrier();
        }
    }
#endif
    // (3) my accumulators' want blocks and n_local onto core 0's plan; n_inj / n_ret as atomic increments (core 0
    // zeroed its plan before its counts went out)
    const uint32_t pl0 = B + O_PLAN;
    {
        uint32_t own = 0;
        for (uint32_t aa = p; aa < A; aa += M, ++own) {
            const uint32_t pw = B + O_PART + own * 48 * 4;
            noc_async_write(pw + 64, get_noc_addr(px[0], py[0], pl0 + (32 + aa * 32) * 4), 128);
            noc_inline_dw_write(get_noc_addr(px[0], py[0], pl0 + (16 + aa) * 4), reinterpret_cast<uint32_t*>(pw)[0]);
        }
    }
    for (uint32_t d = 0; d < 8; ++d) {
        if (ninj[d]) noc_semaphore_inc(get_noc_addr(px[0], py[0], pl0 + 4 * d), ninj[d]);
        if (nret[d]) noc_semaphore_inc(get_noc_addr(px[0], py[0], pl0 + 4 * (8 + d)), nret[d]);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    noc_semaphore_inc(get_noc_addr(px[0], py[0], sem_c), 1);
    if (p == 0) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_c), M);
        invalidate_l1_cache();
        plan[600] = off;
        plan[601] = off > ROWS_CAP ? 1 : 0;
        for (uint32_t e = 0; e < EPC; ++e) {
            cg_[E0 + e] = tot[e];
            rg_[E0 + e] = reg[e];
        }
        const InterleavedAddrGen<true> cg = {.bank_base_address = cnt_addr, .page_size = NG * 4};
        const InterleavedAddrGen<true> rg = {.bank_base_address = reg_addr, .page_size = NG * 4};
        const InterleavedAddrGen<true> pg = {.bank_base_address = plan_addr, .page_size = 4096};
        noc_async_write(B + O_CNT, get_noc_addr(0, cg), NG * 4);
        noc_async_write(B + O_REG, get_noc_addr(0, rg), NG * 4);
        noc_async_write(B + O_PLAN, get_noc_addr(0, pg), 4096);
        noc_async_write_barrier();
    }
    ts[7] = get_timestamp_32b();
    ts[8] = ts[7];
    if (tdbg) {  // (phase timestamps: start, read, sort, home, counts in, placed, out + atomics, end)
        uint32_t* d = reinterpret_cast<uint32_t*>(B + O_DBG);
        for (uint32_t i = 0; i < 9; ++i) d[i] = ts[i];
        const InterleavedAddrGen<true> dg = {.bank_base_address = tdbg, .page_size = 64};
        noc_async_write(B + O_DBG, get_noc_addr(p, dg), 64);
        noc_async_write_barrier();
    }
    // (an atomic still unacknowledged when the kernel returns can land after the next program's launch resets its
    // NoC counters: that program's atomic barrier then never completes - seen as hangs under back-to-back traces)
    noc_async_atomic_barrier();
}
"""


def _mc2_stg(M, EPC, rows_cap, home_rows=0):
    """Slice staging bytes; with row pages also the home ids' landing (64 bytes a token row, before the slices)."""
    return max(((rows_cap // M + 64) * 8 + 16 * min(EPC * M, 512) + 63) // 64 * 64, 64 * home_rows)


def det_page_bytes(S, K):
    """Bytes of an accumulator's page of the deterministic combine's order table (any accumulator count)."""
    return (S * K + 63) // 64 * 64


def _mc2_det_bytes(M, A, N, NG, VM, S, K):
    """L1 of the deterministic-combine tables (combine_planner_prepare det)."""
    al = lambda v: (v + 63) // 64 * 64  # noqa: E731
    own = (A + M - 1) // M
    return (
        (M + 1) * al(NG * 4)
        + al(NG * 8)
        + al(NG * 2)
        + al(N * VM * 8)
        + al(N * VM)
        + al(N * 4)
        + own * al((S + A - 1) // A * K)
        + 4096
        + al(N * VM * 4)
    )


def _bucket_bytes(CH, K, row_pages=False, joint=False):
    """The multi-core planner's bucket region: CH entries x 8 B, and the row pages staged there first (64 B a token,
    ids and weights once each or once joint: more than the entries at K < 8 - MiniMax M3's top-4)."""
    return max(CH * 8, 64 * (1 if joint else 2) * CH // K if row_pages else 0)


def _mc2_inst_rows(rows_cap, M):
    """A multi-core planner core's rows at most: the plan's 32-row tiles (rows_cap / 32 at most) split M ways."""
    return (rows_cap // 32 + M - 1) // M * 32


def _mc2_l1_bytes(M, CH, HS, EPC, NG, rows_cap, home_rows=0, idb=None, K=8, row_pages=False, joint=False):
    al = lambda v: (v + 63) // 64 * 64  # noqa: E731
    CNT = al(2 * EPC * 4)
    XP = al((M + 2) * 4)
    return (
        2 * (al(CH * 2) if idb is None else idb)
        + al(HS * M * 2)
        + al(CH * 4)
        + al(_bucket_bytes(CH, K, row_pages, joint))
        + CNT
        + M * CNT
        + EPC * XP
        + _mc2_stg(M, EPC, rows_cap, home_rows)
        + al(544 * 4)
        + al(_mc2_inst_rows(rows_cap, M) * 4)
        + al(_mc2_inst_rows(rows_cap, M) * 8)
        + 4096
        + 2 * al(NG * 4)
        + 64
        + 12 * 512
        + 64
    )


# Column multi-core planner (deepseek_v41_flash_d_p P.11): the V2 multi-core planner's structure (counting sort into
# per-chunk buckets, count exchange, per-expert chunk bases, row-tile slices pulled by every core, atomics into core 0's
# plan) for the one-core planner's column / pass options (axis=0: chip r of its column, id_offset, tokens / token_base,
# ids_uint32, tile_in, det) - every output bit-identical to the one-core planner's (tests/unit/test_column_planner_mc.py).
# M = N HCH cores, core p plans chunk p: home p / HCH, tokens [(p % HCH) CT, + CT) of the pass (chunks never straddle a
# home, so chunk order = the one-core planner's (home, token, k) scan order and an expert's rows land in that order).
# Per chunk: its tiles' faces 0 / 2 read (ids, weights; tile_in), the column's ids (id - EOFF, others' skipped) counted
# per local expert and listed; the home-token counts (n_local / want / n_ret) of this chip's chunks as atomics onto core
# 0's plan; det: every chunk's rows per column expert travel with the counts, so a chunk of this chip's home tokens knows
# each row's place in its expert's list (rows of the chunks before it) and builds its tokens' order lists itself; the
# accumulators' owners (core a % M) collect them into the order pages. The token index / table tails past the planned
# rows are written as zeros (the one-core planner writes its whole page).
_PLANNER_MCC = r"""
#include <cstdint>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
__DETINC__
constexpr uint32_t ME = get_compile_time_arg_val(0), N = get_compile_time_arg_val(1), S = get_compile_time_arg_val(2);
constexpr uint32_t K = get_compile_time_arg_val(3), EPC = get_compile_time_arg_val(4), NG = get_compile_time_arg_val(5);
constexpr uint32_t A = get_compile_time_arg_val(6), L = get_compile_time_arg_val(7), P = get_compile_time_arg_val(8);
constexpr uint32_t ROWS_CAP = get_compile_time_arg_val(9), M = get_compile_time_arg_val(10);
constexpr uint32_t CT = get_compile_time_arg_val(11), EOFF = get_compile_time_arg_val(12);
constexpr uint32_t SF = get_compile_time_arg_val(13), T0 = get_compile_time_arg_val(14);
constexpr uint32_t STG = get_compile_time_arg_val(15), CBB = get_compile_time_arg_val(16);
constexpr uint32_t IDW = __IDW__;  // bytes per id: 2 (uint16 tiles) or 4 (uint32)
using id_t = std::conditional_t<IDW == 4, uint32_t, uint16_t>;
constexpr uint32_t R = N, C = 1;  // (the planned group: the column)
constexpr uint32_t SV = (T0 + S < SF ? T0 + S : SF) - T0;  // tokens of a home this pass plans (the last pass: fewer)
constexpr uint32_t HCH = (S + CT - 1) / CT, CE = CT * K;
static_assert(M == N * HCH && CT % 32 == 0, "chunks: whole homes, 32-token multiples");
constexpr uint32_t al64(uint32_t v) { return (v + 63) / 64 * 64; }
// a staged tile: ids faces 0 and 2 (rows 0-15 / 16-31, columns 0-15: a token's K <= 16 values), weights faces 0 and 2
constexpr uint32_t FI = 256 * IDW, FW = 512, TBC = 2 * FI + 2 * FW, NTC = CT / 32 + 2;
constexpr uint32_t DET = __DET__, VM = __VM__, DPG = __DPG__, WR = __WR__;
constexpr uint32_t NGP = (NG + 1) / 2 * 2;
constexpr uint32_t MSG = al64(2 * EPC * 4 + (DET ? NGP * 2 : 0));  // counts, bucket starts per expert; det: rows per g
constexpr uint32_t XP = al64((M + 2) * 4);
constexpr uint32_t RC32 = ROWS_CAP / 32, RPC = (RC32 + M - 1) / M * 32 + 32;  // rows one core places at most
constexpr uint32_t OWN = (A + M - 1) / M, NOWN = A < M ? A : M, ZB = 2048, MAXSL = 512;
constexpr uint32_t O_STC = 0, O_LST = O_STC + NTC * TBC, O_BUCK = O_LST + al64(CE * 8), O_MSG = O_BUCK + al64(CE * 8);
constexpr uint32_t O_CIN = O_MSG + MSG, O_XP = O_CIN + M * MSG, O_STG = O_XP + EPC * XP;
constexpr uint32_t O_SL = O_STG + al64(STG), O_PART = O_SL + al64(MAXSL * 12);
constexpr uint32_t O_TIDX = O_PART + al64(544 * 4), O_TAB = O_TIDX + al64(RPC * 4), O_ZERO = O_TAB + al64(RPC * 8);
constexpr uint32_t O_PLAN = O_ZERO + ZB, O_CNT = O_PLAN + 4096, O_REG = O_CNT + al64(NG * 4);
constexpr uint32_t O_DBG = O_REG + al64(NG * 4);
constexpr uint32_t O_GS = O_DBG + 64, O_SCH = O_GS + al64(2 * NG * 4), O_SEID = O_SCH + al64(N * VM * 8);
constexpr uint32_t O_NACT = O_SEID + al64(N * VM), O_TF = O_NACT + al64(N * 4), O_DYN = O_TF + al64(N * VM * 4);
constexpr uint32_t O_ORDL = O_DYN + 4096, O_ORDT = O_ORDL + al64(CE), O_PG = O_ORDT + al64(HCH * CE);
constexpr uint32_t O_FF = O_PG + OWN * DPG;  // (det owners: ZB bytes of 0xFF, a page's tail)
constexpr uint32_t O_END = DET ? O_FF + ZB : O_GS;
static_assert(O_END + 64 <= CBB, "the planner's L1 (host: _mcc_l1_bytes)");
constexpr uint32_t WX = __WX__, WY = __WY__;  // torus axes (rings: the shorter way, ties Y- / X+ as the fabric)
FORCE_INLINE uint32_t first_hop(uint32_t src, uint32_t dst) {
    const uint32_t sr = src / C, sc = src % C, dr = dst / C, dc = dst % C;
    if (sr != dr) {
        if constexpr (WY) return 2 * ((dr + R - sr) % R) < R ? 2 : 3;
        return dr > sr ? 2 : 3;
    }
    if constexpr (WX) return 2 * ((dc + C - sc) % C) <= C ? 0 : 1;
    return dc > sc ? 0 : 1;
}
void kernel_main() {
    size_t a = 0;
    const uint32_t ids_addr = get_arg_val<uint32_t>(a++), w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t cnt_addr = get_arg_val<uint32_t>(a++), reg_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tidx_addr = get_arg_val<uint32_t>(a++), tab_addr = get_arg_val<uint32_t>(a++);
    const uint32_t plan_addr = get_arg_val<uint32_t>(a++);
    uint32_t lut[8];
    for (uint32_t i = 0; i < 8; ++i) lut[i] = get_arg_val<uint32_t>(a++);
    const uint32_t p = get_arg_val<uint32_t>(a++);
    uint32_t px[M], py[M];
    for (uint32_t i = 0; i < M; ++i) {
        px[i] = get_arg_val<uint32_t>(a++);
        py[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t tdbg = get_arg_val<uint32_t>(a++);
    const uint32_t dord_addr = DET ? get_arg_val<uint32_t>(a++) : 0;
    uint32_t ts[16];
    for (uint32_t i = 0; i < 16; ++i) ts[i] = 0;
    ts[0] = get_timestamp_32b();
    cb_reserve_back(0, 1);
    const uint32_t B = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t sem_l = get_semaphore(0), sem_c = get_semaphore(1), sem_x = get_semaphore(2);
    uint32_t* plan = reinterpret_cast<uint32_t*>(B + O_PLAN);
    // my chunk: home mo, its pass-local tokens [ta, ta + nt), gathered rows from row0
    const uint32_t mo = p / HCH, ta = (p % HCH) * CT;
    const uint32_t nt = ta < SV ? (SV - ta < CT ? SV - ta : CT) : 0;
    const bool home_me = mo == ME;
    const uint32_t row0 = mo * SF + T0 + ta, tile0 = row0 / 32;
    const uint32_t ntl = nt ? (row0 + nt - 1) / 32 - tile0 + 1 : 0;
    {
        const InterleavedAddrGen<true> ig = {.bank_base_address = ids_addr, .page_size = 1024 * IDW};
        const InterleavedAddrGen<true> wg = {.bank_base_address = w_addr, .page_size = 2048};
        for (uint32_t i = 0; i < ntl; ++i) {
            const uint32_t slot = B + O_STC + i * TBC;
            const uint64_t si = get_noc_addr(tile0 + i, ig), sw = get_noc_addr(tile0 + i, wg);
            noc_async_read(si, slot, FI);
            noc_async_read(si + 2 * FI, slot + FI, FI);
            noc_async_read(sw, slot + 2 * FI, FW);
            noc_async_read(sw + 2 * FW, slot + 2 * FI + FW, FW);
        }
    }
    uint32_t* zero = reinterpret_cast<uint32_t*>(B + O_ZERO);
    for (uint32_t i = 0; i < ZB / 4; ++i) zero[i] = 0;
    // (counters in local memory: an L1 read-modify-write costs ~10x)
    static_assert(NG <= 256 && N <= 8 && A <= 16 && N * S < 65536, "local counters");
    uint32_t nret[8], nloc[16], want[A * N];  // my home tokens' plan words (n_ret, n_local, want)
    for (uint32_t i = 0; i < 8; ++i) nret[i] = 0;
    for (uint32_t i = 0; i < 16; ++i) nloc[i] = 0;
    for (uint32_t i = 0; i < A * N; ++i) want[i] = 0;
    uint32_t* myc = reinterpret_cast<uint32_t*>(B + O_MSG);
    uint16_t gall[DET ? NGP : 2];  // (det) my chunk's rows per column expert
    for (uint32_t i = 0; i < (DET ? NGP : 2); ++i) gall[i] = 0;
    static_assert(!DET || (NG < 255 && CE <= 2048), "(det) a row's column expert in a byte, local");
    uint8_t grow[DET ? CE : 4];  // (det, my home's chunks) each row's column expert (0xFF: another column's)
    uint8_t hop[N * L];
    for (uint32_t o = 0; o < N; ++o) {
        for (uint32_t l = 0; l < L; ++l) hop[o * L + l] = o == ME ? 0xFF : lut[first_hop(ME, o) * 2 + l];
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    ts[1] = get_timestamp_32b();
    // a token's ids / weights in the staged tiles
    auto tok = [&](uint32_t lt, const id_t*& ip, const uint16_t*& wp) __attribute__((always_inline)) {
        const uint32_t row = row0 + lt, rr = row % 32, slot = B + O_STC + (row / 32 - tile0) * TBC;
        ip = reinterpret_cast<const id_t*>(slot + (rr >= 16 ? FI : 0) + (rr % 16) * 16 * IDW);
        wp = reinterpret_cast<const uint16_t*>(slot + 2 * FI + (rr >= 16 ? FW : 0) + (rr % 16) * 32);
    };
    // (1) my chunk's column rows: local experts listed (header, weight), counts per expert; my home tokens' plan words
    constexpr uint32_t E0 = ME * EPC;
    uint32_t cnt[EPC], bo[EPC];
    for (uint32_t e = 0; e < EPC; ++e) cnt[e] = 0;
    uint32_t* lst = reinterpret_cast<uint32_t*>(B + O_LST);
    uint32_t nl = 0;
    for (uint32_t lt = 0; lt < nt; ++lt) {
        const id_t* ip;
        const uint16_t* wp;
        tok(lt, ip, wp);
        const uint32_t t = ta + lt, ac = t % A;
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t g = static_cast<uint32_t>(ip[k]) - EOFF;
            if constexpr (DET) grow[lt * K + k] = g < NG ? static_cast<uint8_t>(g) : 0xFF;
            if (g >= NG) continue;  // (another column's expert)
            if constexpr (DET) ++gall[g];
            const uint32_t e = g - E0;
            if (e < EPC) {
                ++cnt[e];
                lst[2 * nl] = mo | (t << 8) | (e << 24);
                lst[2 * nl + 1] = wp[k];
                ++nl;
            }
            if (home_me) {  // (a row for my token)
                const uint32_t src = g / EPC;
                if (src == ME) {
                    ++nloc[ac];
                } else {
                    ++want[ac * N + src];
                    ++nret[hop[src * L + (ac + src + ME) % L]];
                }
            }
        }
    }
    for (uint32_t e = 0, o_ = 0; e < EPC; ++e) {
        bo[e] = o_;
        myc[EPC + e] = o_;
        o_ += cnt[e];
        myc[e] = cnt[e];
    }
    if constexpr (DET) {  // (two counts a word into the message)
        uint32_t* mg = reinterpret_cast<uint32_t*>(B + O_MSG + 2 * EPC * 4);
        for (uint32_t i = 0; i < NGP / 2; ++i) mg[i] = gall[2 * i] | (static_cast<uint32_t>(gall[2 * i + 1]) << 16);
    }
    uint32_t* buck = reinterpret_cast<uint32_t*>(B + O_BUCK);
    uint32_t ninj[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t x = 0; x < nl; ++x) {
        const uint32_t h = lst[2 * x], e = h >> 24, t = (h >> 8) & 0xFFFF, pos = bo[e]++;
        uint32_t dest;
        if (home_me) {
            dest = P + t % A;
        } else {
            dest = hop[mo * L + (t % A + ME + mo) % L];  // (fixed per (accumulator, source, home): one ring each)
            ++ninj[dest];
        }
        buck[2 * pos] = dest | (lst[2 * x + 1] << 16);
        buck[2 * pos + 1] = h;
    }
    ts[2] = get_timestamp_32b();
    for (uint32_t q = 0; q < M; ++q) {
        noc_async_write(B + O_MSG, get_noc_addr(px[q], py[q], B + O_CIN + p * MSG), MSG);
    }
    noc_async_write_barrier();
    for (uint32_t q = 0; q < M; ++q) noc_semaphore_inc(get_noc_addr(px[q], py[q], sem_l), 1);
    ts[3] = get_timestamp_32b();
    uint32_t* cg_ = reinterpret_cast<uint32_t*>(B + O_CNT);
    uint32_t* rg_ = reinterpret_cast<uint32_t*>(B + O_REG);
    if (p == 0) {  // (while the others sort: their atomics onto the plan follow the xp barrier, after this; local NoC
                   // copies of the zero buffer: a store loop over the 4 KB page costs ~4K cycles)
        static_assert(O_CNT == O_PLAN + 4096 && O_REG == O_CNT + al64(NG * 4), "plan, counts, regions in a row");
        const uint32_t zb = 4096 + 2 * al64(NG * 4);
        for (uint32_t b = 0; b < zb; b += ZB) {
            noc_async_write(B + O_ZERO, get_noc_addr(B + O_PLAN + b), zb - b < ZB ? zb - b : ZB);
        }
        noc_async_write_barrier();
    }
    // (2) all counts in; expert e's owner (core e % M) totals it and each chunk's first row, sends that to all
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l), M);
    invalidate_l1_cache();
    ts[4] = get_timestamp_32b();
    const uint32_t* cin = reinterpret_cast<const uint32_t*>(B + O_CIN);
    for (uint32_t e = p; e < EPC; e += M) {
        uint32_t* xp = reinterpret_cast<uint32_t*>(B + O_XP + e * XP);
        uint32_t s_ = 0;
        for (uint32_t j = 0; j < M; ++j) {
            xp[2 + j] = s_;
            s_ += cin[j * (MSG / 4) + e];
        }
        xp[0] = s_;
        for (uint32_t q = 0; q < M; ++q) {
            if (q != p) noc_async_write(B + O_XP + e * XP, get_noc_addr(px[q], py[q], B + O_XP + e * XP), XP);
        }
    }
    noc_async_write_barrier();
    for (uint32_t q = 0; q < M; ++q) noc_semaphore_inc(get_noc_addr(px[q], py[q], sem_x), 1);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_x), M);
    invalidate_l1_cache();
    uint32_t tot[EPC], reg[EPC], off = 0;
    for (uint32_t e = 0; e < EPC; ++e) {
        const uint32_t s_ = reinterpret_cast<const uint32_t*>(B + O_XP + e * XP)[0];
        tot[e] = s_;
        reg[e] = off;
        off += (s_ + 31) / 32 * 32;
    }
    // (rows past ROWS_CAP are dropped, as the one-core planner does: NTC tiles at most)
    const uint32_t NT = off / 32, NTC_ = NT < RC32 ? NT : RC32;
    const uint32_t R0 = (p * NTC_ / M) * 32, R1 = ((p + 1) * NTC_ / M) * 32;
    uint32_t* tidx = reinterpret_cast<uint32_t*>(B + O_TIDX);
    uint32_t* tab = reinterpret_cast<uint32_t*>(B + O_TAB);
    struct Sl {
        uint32_t lo, n, stg;
    };
    Sl* sl = reinterpret_cast<Sl*>(B + O_SL);
    uint32_t nsl = 0, sp = 0;
    auto drain = [&]() {
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t q = 0; q < nsl; ++q) {
            const uint32_t* st = reinterpret_cast<const uint32_t*>(sl[q].stg);
            for (uint32_t x = 0; x < sl[q].n; ++x) {
                const uint32_t r = sl[q].lo + x - R0, w0 = st[2 * x], w1 = st[2 * x + 1];
                tab[2 * r] = w0;
                tab[2 * r + 1] = w1;
                tidx[r] = (w1 & 0xFF) * SF + T0 + ((w1 >> 8) & 0xFFFF);  // (home x tokens + token: the gathered x row)
            }
        }
        nsl = sp = 0;
    };
    for (uint32_t e = 0; e < EPC && R1 > R0; ++e) {
        const uint32_t e_lo = reg[e], e_end = reg[e] + tot[e], e_pad = reg[e] + (tot[e] + 31) / 32 * 32;
        if (e_pad <= R0 || e_lo >= R1) continue;
        for (uint32_t r = (e_end > R0 ? e_end : R0); r < (e_pad < R1 ? e_pad : R1); ++r) {  // padding rows
            tidx[r - R0] = 0;
            tab[2 * (r - R0)] = tab[2 * (r - R0) + 1] = 0;
        }
        const uint32_t* xp = reinterpret_cast<const uint32_t*>(B + O_XP + e * XP);
        for (uint32_t j = 0; j < M; ++j) {
            const uint32_t* cj = cin + j * (MSG / 4);
            const uint32_t base = e_lo + xp[2 + j], n = (j + 1 < M ? xp[3 + j] : tot[e]) - xp[2 + j];
            const uint32_t lo = base > R0 ? base : R0, hi = base + n < R1 ? base + n : R1;
            if (lo < hi) {
                const uint32_t src = B + O_BUCK + (cj[EPC + e] + (lo - base)) * 8, cnt_ = hi - lo;
                const uint32_t bytes = ((src & 15) + cnt_ * 8 + 15) & ~15u;
                if (nsl == MAXSL || sp + bytes > STG) drain();  // (staging full)
                noc_async_read(get_noc_addr(px[j], py[j], src & ~15u), B + O_STG + sp, bytes);
                sl[nsl++] = Sl{lo, cnt_, B + O_STG + sp + (src & 15)};
                sp += bytes;
            }
        }
    }
    drain();
    ts[5] = get_timestamp_32b();
    const InterleavedAddrGen<true> ig = {.bank_base_address = tidx_addr, .page_size = ROWS_CAP * 4};
    const InterleavedAddrGen<true> tg = {.bank_base_address = tab_addr, .page_size = 256};
    const uint32_t W1 = R1 < ROWS_CAP ? R1 : ROWS_CAP;
    if (W1 > R0) {
        noc_async_write(B + O_TIDX, get_noc_addr(0, ig) + R0 * 4, (W1 - R0) * 4);
        for (uint32_t pg = R0 / 32; pg < W1 / 32; ++pg) {
            noc_async_write(B + O_TAB + (pg - R0 / 32) * 256, get_noc_addr(pg, tg), 256);
        }
    }
    if (NT < RC32) {  // (the rows past the plan: zeros, a 1/M share each)
        const uint32_t z0 = NT + p * (RC32 - NT) / M, z1 = NT + (p + 1) * (RC32 - NT) / M;
        for (uint32_t b = z0 * 128; b < z1 * 128; b += ZB) {
            noc_async_write(B + O_ZERO, get_noc_addr(0, ig) + b, z1 * 128 - b < ZB ? z1 * 128 - b : ZB);
        }
        for (uint32_t pg = z0; pg < z1; ++pg) noc_async_write(B + O_ZERO, get_noc_addr(pg, tg), 256);
    }
    // (3) n_inj and my home tokens' words onto core 0's plan (atomic increments; core 0 zeroed it before its counts
    // went out)
    const uint32_t pl0 = B + O_PLAN;
    for (uint32_t d = 0; d < 8; ++d) {
        if (ninj[d]) noc_semaphore_inc(get_noc_addr(px[0], py[0], pl0 + 4 * d), ninj[d]);
    }
    if (home_me && nt) {
        for (uint32_t d = 0; d < 8; ++d) {
            if (nret[d]) noc_semaphore_inc(get_noc_addr(px[0], py[0], pl0 + 4 * (8 + d)), nret[d]);
        }
        for (uint32_t ac = 0; ac < A; ++ac) {
            if (nloc[ac]) noc_semaphore_inc(get_noc_addr(px[0], py[0], pl0 + 4 * (16 + ac)), nloc[ac]);
            for (uint32_t s = 0; s < N; ++s) {
                const uint32_t v = want[ac * N + s];
                if (v) noc_semaphore_inc(get_noc_addr(px[0], py[0], pl0 + 4 * (32 + ac * 32 + s)), v);
            }
        }
    }
    ts[6] = get_timestamp_32b();
#if __DET__  // (preprocessor: SeDyn / SE_RPS exist only with det's include)
    {
        const uint32_t sem_d = get_semaphore(3);
        if (home_me) {
            // rows per column expert: all chunks (every source's schedule), and the chunks before mine (each of my rows'
            // place in its expert's row list: homes before mine, then my home's earlier tokens)
            // (two counts a word, summed packed: < 65536 rows an expert, no carry between the halves)
            uint32_t gsp[NGP / 2], bsp[NGP / 2];
            for (uint32_t i = 0; i < NGP / 2; ++i) gsp[i] = bsp[i] = 0;
            for (uint32_t j = 0; j < M; ++j) {
                const uint32_t* gj = reinterpret_cast<const uint32_t*>(B + O_CIN + j * MSG + 2 * EPC * 4);
                if (j == p) {
                    for (uint32_t i = 0; i < NGP / 2; ++i) bsp[i] = gsp[i];
                }
                for (uint32_t i = 0; i < NGP / 2; ++i) gsp[i] += gj[i];
            }
            ts[8] = get_timestamp_32b();
            uint32_t* sch = reinterpret_cast<uint32_t*>(B + O_SCH);
            uint32_t* tf = reinterpret_cast<uint32_t*>(B + O_TF);
            static_assert(VM < 255 && EPC < 256, "entry lists in bytes");
            uint8_t vfirst[N * EPC], vnext[N * VM];  // (each expert's entries in schedule order)
            static_assert(sizeof(SeDyn) <= 4096, "the schedule scratch");
            SeDyn& dy = *reinterpret_cast<SeDyn*>(B + O_DYN);
            for (uint32_t s = 0; s < N; ++s) {
                se_dyn_from_counts(dy, EPC, SE_RPS, [&](uint32_t e) {
                    const uint32_t g = s * EPC + e;
                    return (gsp[g / 2] >> (16 * (g & 1))) & 0xFFFF;
                });
                for (uint32_t e = 0; e < EPC; ++e) vfirst[s * EPC + e] = 0xFF;
                for (uint32_t v = dy.n_act; v-- > 0;) {
                    vnext[s * VM + v] = vfirst[s * EPC + dy.eid[v]];
                    vfirst[s * EPC + dy.eid[v]] = static_cast<uint8_t>(v);
                }
                uint32_t fin = 0;
                for (uint32_t v = 0; v < dy.n_act; ++v) {  // (first row | rows << 16: < 65536 rows a source)
                    sch[s * VM + v] = dy.off[v] | (dy.cnt[v] << 16);
                    fin += dy.cnt[v] > WR ? dy.cnt[v] : WR;
                    tf[s * VM + v] = fin;
                }
            }
            ts[9] = get_timestamp_32b();
            uint16_t base[NGP];  // (each expert's rows in the chunks before mine)
            for (uint32_t g = 0; g < NGP; ++g) base[g] = (bsp[g / 2] >> (16 * (g & 1))) & 0xFFFF;
            uint8_t* ordl = reinterpret_cast<uint8_t*>(B + O_ORDL);
            for (uint32_t i = 0; i < CE; ++i) ordl[i] = 0xFF;
            for (uint32_t lt = 0; lt < nt; ++lt) {
                uint32_t kt[K], nk = 0;
                uint8_t ks[K];
                for (uint32_t k = 0; k < K; ++k) {
                    const uint32_t g = grow[lt * K + k];
                    if (g == 0xFF) continue;
                    const uint32_t r = base[g]++, s = g / EPC;
                    for (uint32_t v = vfirst[g]; v != 0xFF; v = vnext[s * VM + v]) {
                        const uint32_t w_ = sch[s * VM + v], o_ = w_ & 0xFFFF;
                        if (r >= o_ && r < o_ + (w_ >> 16)) {
                            const uint32_t key = tf[s * VM + v];
                            uint32_t i = nk++;
                            for (; i > 0 && (kt[i - 1] > key || (kt[i - 1] == key && ks[i - 1] > s)); --i) {
                                kt[i] = kt[i - 1];
                                ks[i] = ks[i - 1];
                            }
                            kt[i] = key;
                            ks[i] = static_cast<uint8_t>(s);
                            break;
                        }
                    }
                }
                for (uint32_t i = 0; i < nk; ++i) ordl[lt * K + i] = ks[i];
            }
            ts[10] = get_timestamp_32b();
            // my tokens' order lists to every accumulator owner (its copy of this chip's home tokens, token-major)
            for (uint32_t q = 0; q < NOWN; ++q) {
                noc_async_write(B + O_ORDL, get_noc_addr(px[q], py[q], B + O_ORDT + (p % HCH) * CE), CE);
            }
            noc_async_write_barrier();
            for (uint32_t q = 0; q < NOWN; ++q) noc_semaphore_inc(get_noc_addr(px[q], py[q], sem_d), 1);
        }
        if (p < NOWN) {  // an accumulator's page: its tokens t = a + j A in order, K sources each (0xFF: none)
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_d), HCH);
            invalidate_l1_cache();
            ts[11] = get_timestamp_32b();
            const uint8_t* ordt = reinterpret_cast<const uint8_t*>(B + O_ORDT);
            uint32_t* ff = reinterpret_cast<uint32_t*>(B + O_FF);
            for (uint32_t i = 0; i < ZB / 4; ++i) ff[i] = 0xFFFFFFFFu;
            const InterleavedAddrGen<true> og = {.bank_base_address = dord_addr, .page_size = DPG};
            uint32_t own = 0;
            for (uint32_t ac = p; ac < A; ac += M, ++own) {
                uint8_t* pg = reinterpret_cast<uint8_t*>(B + O_PG + own * DPG);
                uint32_t j = 0;
                if constexpr (K % 2 == 0) {  // (records 2-byte aligned: halfwords)
                    uint16_t* d16 = reinterpret_cast<uint16_t*>(pg);
                    const uint16_t* s16 = reinterpret_cast<const uint16_t*>(ordt);
                    for (uint32_t t = ac; t < SV; t += A, ++j) {
                        for (uint32_t k = 0; k < K / 2; ++k) d16[j * (K / 2) + k] = s16[t * (K / 2) + k];
                    }
                } else {
                    for (uint32_t t = ac; t < SV; t += A, ++j) {
                        for (uint32_t k = 0; k < K; ++k) pg[j * K + k] = ordt[t * K + k];
                    }
                }
                // the records, then 0xFF to the page end: up to the next 64 B here, the rest from the 0xFF buffer
                const uint32_t b0 = j * K, b1 = (b0 + 63) / 64 * 64 < DPG ? (b0 + 63) / 64 * 64 : DPG;
                for (uint32_t b = b0; b < b1; ++b) pg[b] = 0xFF;
                noc_async_write(B + O_PG + own * DPG, get_noc_addr(ac, og), b1);
                for (uint32_t b = b1; b < DPG; b += ZB) {
                    noc_async_write(B + O_FF, get_noc_addr(ac, og) + b, DPG - b < ZB ? DPG - b : ZB);
                }
            }
        }
    }
#endif
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    noc_semaphore_inc(get_noc_addr(px[0], py[0], sem_c), 1);
    if (p == 0) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_c), M);
        invalidate_l1_cache();
        plan[600] = off;
        plan[601] = off > ROWS_CAP ? 1 : 0;
        for (uint32_t e = 0; e < EPC; ++e) {
            cg_[E0 + e] = tot[e];
            rg_[E0 + e] = reg[e];
        }
        const InterleavedAddrGen<true> cg = {.bank_base_address = cnt_addr, .page_size = NG * 4};
        const InterleavedAddrGen<true> rg = {.bank_base_address = reg_addr, .page_size = NG * 4};
        const InterleavedAddrGen<true> pg = {.bank_base_address = plan_addr, .page_size = 4096};
        noc_async_write(B + O_CNT, get_noc_addr(0, cg), NG * 4);
        noc_async_write(B + O_REG, get_noc_addr(0, rg), NG * 4);
        noc_async_write(B + O_PLAN, get_noc_addr(0, pg), 4096);
        noc_async_write_barrier();
    }
    ts[7] = get_timestamp_32b();
    if (tdbg) {  // (phase timestamps: start, read, sort, counts out, counts in, placed, rows out + atomics, end)
        uint32_t* d = reinterpret_cast<uint32_t*>(B + O_DBG);
        for (uint32_t i = 0; i < 16; ++i) d[i] = ts[i];
        const InterleavedAddrGen<true> dg = {.bank_base_address = tdbg, .page_size = 64};
        noc_async_write(B + O_DBG, get_noc_addr(p, dg), 64);
        noc_async_write_barrier();
    }
    noc_async_atomic_barrier();
}
"""


def _mcc_geometry(S, N, cores):
    """(M, CT): the column multi-core planner's core count and tokens per chunk for ``cores`` requested cores (chunks
    of 32-token multiples inside one home each; M = N ceil(S / CT) <= cores, >= N)."""
    hch = max(1, cores // N)
    ct = -(-(-(-S // hch)) // 32) * 32
    return N * -(-S // ct), ct


def _mcc_l1_bytes(M, CT, N, S, K, EPC, NG, A, rows_cap, idw, det, VM):
    """L1 of the column multi-core planner (the kernel's layout; it static_asserts against this)."""
    al = lambda v: (v + 63) // 64 * 64  # noqa: E731
    hch, ce = -(-S // CT), CT * K
    ngp = (NG + 1) // 2 * 2
    msg = al(2 * EPC * 4 + (ngp * 2 if det else 0))
    rc32 = rows_cap // 32
    rpc = -(-rc32 // M) * 32 + 32
    own = -(-A // M)
    tbc = 2 * 256 * idw + 2 * 512
    b = (CT // 32 + 2) * tbc + 2 * al(ce * 8) + msg + M * msg + EPC * al((M + 2) * 4)
    b += al(_mc2_stg(M, EPC, rows_cap)) + al(512 * 12) + al(544 * 4) + al(rpc * 4) + al(rpc * 8) + 2048 + 4096
    b += 2 * al(NG * 4) + 64
    if det:
        b += al(2 * NG * 4) + al(N * VM * 8) + al(N * VM) + al(N * 4) + al(N * VM * 4) + 4096 + al(ce) + al(hch * ce)
        b += own * det_page_bytes(S, K) + 2048
    return b + 64


def first_hop_idx(src, dst, C, R=None, wrap=(False, False)):
    """DIRS index (0 X+, 1 X-, 2 Y+, 3 Y-) of the first hop src -> dst: Y first, then X; wrap = (X, Y) rings, the
    shorter way, ties Y- / X+ (the fabric's routing tables)."""
    sr, sc, dr, dc = src // C, src % C, dst // C, dst % C
    if sr != dr:
        if wrap[1]:
            f2 = 2 * ((dr - sr) % R)  # (SPC_YTIE: the opposite row splits by column parity, as the planner's YT)
            return 2 if f2 < R or (f2 == R and os.environ.get("SPC_YTIE") == "1" and (sc + dc) % 2) else 3
        return 2 if dr > sr else 3
    if wrap[0]:
        return 0 if 2 * ((dc - sc) % C) <= C else 1
    return 0 if dc > sc else 1


def link_of(acc, src, home, L):
    """The link of every row from chip src to accumulator acc of chip home (one ring per source at each accumulator)."""
    return (acc + src + home) % L


def plan_reference(
    ids,
    wbits,
    me,
    R,
    C,
    S,
    K,
    EPC,
    NG,
    A,
    L,
    P,
    lut,
    rows_cap,
    wrap=(False, False),
    token_major=False,
    id_offset=0,
    token_base=0,
):
    """Host model of the planner for chip `me`: ids / wbits [N, S, K] (global expert ids, bf16 bits); token_major: an
    expert's rows in (token, home, k) order (the multi-core planner's default, run.token_major), else (home, token, k).
    id_offset / token_base (combine_planner_prepare's column / pass options): ids [N, SF, K] with SF tokens per home,
    the S tokens from token_base planned, id_offset subtracted, ids outside [0, N EPC) skipped (R, C: the planned
    group, e.g. R x 1 for a column)."""
    N = R * C
    E0 = me * EPC
    SF = ids.shape[1]
    skip = id_offset != 0 or token_base != 0 or SF != S
    ids = ids.long() - id_offset
    if skip:  # (only the pass's tokens; another column's ids out of range)
        ids = ids[:, token_base : token_base + S]
        wbits = wbits[:, token_base : token_base + S]
        ids = torch.where((ids >= 0) & (ids < N * EPC), ids, torch.full_like(ids, 1 << 20))
    lc = [0] * EPC
    flat = ids.reshape(-1).tolist()
    for g in flat:
        if E0 <= g < E0 + EPC:
            lc[g - E0] += 1
    cnt, reg = [0] * NG, [0] * NG
    fill, off = [0] * EPC, 0
    for e in range(EPC):
        cnt[E0 + e], reg[E0 + e], fill[e] = lc[e], off, off
        off += (lc[e] + 31) // 32 * 32
    tidx = [0] * rows_cap
    tab = [0] * (2 * rows_cap)
    plan = [0] * 1024
    plan[P_ROWS], plan[P_OVER] = off, int(off > rows_cap)
    idsl, wl = ids.tolist(), wbits.tolist()
    S_ = ids.shape[1]
    scan = (
        [(o, t) for t in range(S_) for o in range(N)] if token_major else [(o, t) for o in range(N) for t in range(S_)]
    )
    for o, t in scan:
        dh = 0 if o == me else first_hop_idx(me, o, C, R, wrap)
        for k in range(K):
            g, wb = idsl[o][t][k], wl[o][t][k]
            src = g // EPC
            if skip and src >= N:
                continue
            l = link_of(t % A, src, o, L)
            if src == me:
                e = g - E0
                r = fill[e]
                fill[e] += 1
                if r < rows_cap:
                    tidx[r] = o * SF + token_base + t
                    if o == me:
                        dest = P + t % A
                    else:
                        dest = lut[dh * 2 + l]
                        plan[P_NINJ + dest] += 1
                    tab[2 * r] = dest | (wb << 16)
                    tab[2 * r + 1] = o | (t << 8) | (e << 24)
            if o == me:
                if src == me:
                    plan[P_NLOC + t % A] += 1
                else:
                    plan[P_WANT + (t % A) * 32 + src] += 1
                    plan[P_NRET + lut[first_hop_idx(me, src, C, R, wrap) * 2 + l]] += 1
    return dict(counts=cnt, regions=reg, token_index=tidx, table=tab, plan=plan)


def port_luts(mesh_device, num_links, torus=(False, False), per_direction=False, axis=None):
    """Per chip coord: lut[direction * 2 + link] = port id (the chip's ports in DIRS / link order, as
    sparse_combine_rf_prepare numbers them), and the port count. per_direction: one port per direction, driving all
    its links (sparse_combine_rf_prepare ports_per_direction): every link of a direction maps to that port's id.
    axis=0: the column combine's ports only (Y directions; sparse_combine_rf_prepare axis=0)."""
    from .sparse_all_gather_2d import DIRS, mesh_plan

    chips, _, _ = mesh_plan(mesh_device, num_links, torus)
    if axis is not None:
        chips = axis_ports(chips, axis)
    luts, n = {}, {}
    for ch, ports in chips.items():
        lut, j = [0xFF] * 8, 0
        for di, d in enumerate(DIRS):
            if d in ports:
                for li in range(num_links):
                    lut[di * 2 + li] = j
                    j += 0 if per_direction else 1
                j += 1 if per_direction else 0
        luts[ch], n[ch] = lut, j
    return luts, n


def axis_ports(chips, axis):
    """mesh_plan's chips with only the ports along mesh ``axis`` (0: Y, the column; 1: X, the row)."""
    keep = "Y" if axis == 0 else "X"
    return {ch: {d: p for d, p in ports.items() if d[0] == keep} for ch, ports in chips.items()}


def combine_planner_prepare(
    mesh_device,
    *,
    S,
    K,
    EPC,
    A,
    L,
    port_luts,
    n_ports,
    rows_cap,
    core=(0, 0),
    cores=None,
    row_pages=False,
    joint=False,
    tokens_per_page=1,
    torus=(False, False),
    det=None,
    token_major=True,
    axis=None,
    id_offset=None,
    tokens=None,
    token_base=0,
    ids_uint32=False,
    tile_in=False,
):
    """port_luts[coord] = 8 port ids (direction * 2 + link; 0xFF none), n_ports[coord] = P. Returns run(ids, wts) ->
    dict of the per-chip output tensors (counts, regions, token_index, table, plan), allocated once. ids / wts: one
    row-major page [1, N S K] each, or with row_pages (the multi-core planner) [N S, K] row-major (a page per token),
    or with joint (and row_pages) one [N S, 2 K] uint16 row-major tensor, each token's ids then its weights' bits
    (run(t, t)).

    Column / pass options (defaults = the whole-mesh plan above, unchanged); the one-core planner, or with ``cores``
    (and tile_in) the column multi-core planner (P.11), whose outputs are bit-identical to the one-core planner's:
      axis=0       the combine along each mesh column (sparse_combine_rf_prepare axis=0): chip (r, c) plans as chip r
                   of an R x 1 mesh (its column), homes and sources are the column's chips (header home = r), ids /
                   wts hold the column's tokens [R tokens S K] (e.g. all-gathered over axis 0), expert ids are the
                   column's (global id - id_offset(r, c), in [0, R EPC); others are skipped: another column's
                   experts), counts / regions are [1, R EPC] (the flat expert's ids local to the column).
      id_offset    callable (r, c) -> the id subtracted from every routing id on that chip (axis=0: the column's first
                   global expert); ids that fall outside the chip group's range are skipped.
      tokens       tokens per home in ids and in the gathered x (default S): with token_base, plan the S tokens
                   [token_base, token_base + S) of every home (a pass; tokens past ``tokens`` are skipped). The
                   header's token and the accumulator are pass-local (t - token_base: home rows [0, S)), the token
                   index is home * tokens + token (the full gathered x).
      ids_uint32   the ids page holds uint32 ids (default uint16; ttnn.reshape refuses uint16 row-major tensors).
      tile_in      ids / wts are TILE [tokens, K] tensors (K <= 16; e.g. the router's top-k gathered as tiles), read
                   tile by tile: no untilize and no reshape to one page (a ttnn.reshape of [2048, 6] to one page took
                   ~1 ms on the device).
      cores        with axis=0 (default None: the one-core planner on ``core``): the column multi-core planner on these
                   logical cores (the first M = N ceil(S / CT) of them, CT the 32-token-multiple chunk of a home's
                   tokens that spreads S over cores // N chunks): chunk counting sorts, count exchange, row slices,
                   det order lists per chunk; needs tile_in. run.mcc = (M, CT)."""
    from .sparse_all_gather_2d import mesh_plan

    R, C = tuple(mesh_device.shape)
    R_m, C_m = R, C  # (the mesh; R, C below: the planned group's shape)
    if axis is not None:
        assert axis == 0, "axis=0 only (the column)"
        assert cores is None or tile_in, "axis=0 with cores (the column multi-core planner): tile_in only"
        C = 1
    SF = S if tokens is None else int(tokens)
    T0 = int(token_base)
    skipf = axis is not None or id_offset is not None or SF != S or T0 != 0
    colmc = cores is not None and axis is not None  # (the column multi-core planner)
    assert cores is None or not skipf or colmc, "id_offset / tokens / token_base: the one-core / column planners"
    N = R * C
    NG = N * EPC
    assert rows_cap % 32 == 0 and N * S * K < (1 << 16) * 64 and N <= 32 and A <= 16
    _, _, (wrap_x, wrap_y) = mesh_plan(mesh_device, L, torus)
    yt = int(os.environ.get("SPC_YTIE") == "1")  # (the combine's probe, CombineProbes.ytie: both sides route alike)
    wsub = lambda src: (  # noqa: E731
        src.replace("__WX__", str(int(wrap_x))).replace("__WY__", str(int(wrap_y))).replace("__YT__", str(yt))
    )
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R_m, C_m), dims=(0, 1))
    zeros = lambda w: ttnn.from_torch(  # noqa: E731
        torch.zeros(R_m, C_m, 1, w, dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    outs = dict(
        counts=zeros(NG),
        regions=zeros(NG),
        token_index=zeros(rows_cap),
        table=ttnn.from_torch(
            torch.zeros(R_m, C_m, rows_cap // 32, 64, dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        ),
        plan=zeros(1024),
    )
    # det (deterministic combine): the flat expert's schedule parameters - dict(rps=, max_e=, gu_nreg=, pin=,
    # pin_ratio=, optional w_rows=) from its plan (sched_rps / sched_max_e / sched_gu_nreg) and call (pin). Adds
    # outs["dord"]: per accumulator, per owned home token (t = accumulator + j A), the source chips of its K rows in
    # the order sparse_combine_rf_prepare(deterministic=True) adds them (uint8, 0xFF past the token's rows; a page of
    # det_page_bytes(S, K) each). The order: by the row's predicted finish on its chip (the flat schedule, an entry
    # costing max(w_rows, its rows)), then source chip - each source's production order, so the accumulator can
    # always take some ring's head row; a token waits only on its own earlier rows, never on a global step.
    VM = 2 * det["max_e"] if det is not None else 1
    if det is not None:
        assert (
            cores is not None or skipf
        ), "det: the multi-core planner, or the one-core planner with the column options"
        assert N < 255, "det: source chips fit a byte"
        outs["dord"] = ttnn.from_torch(
            torch.zeros(
                R_m, C_m, 16, det_page_bytes(S, K) // 4, dtype=torch.int32
            ),  # (16 pages: a probe planner's serve any A)
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        detinc = "\n".join(
            [
                f"#define SE_RPS {det['rps']}",
                f"#define SE_MAX_E {det['max_e']}",
                f"#define SE_GU_NREG {det['gu_nreg']}",
                "#define SE_META_BYTES 65536",
            ]  # (no meta CB here)
            + (
                [
                    f"#define SE_PIN_MIN {det['pin']}",
                    "#define SE_PIN_SMALL 2",
                    f"#define SE_PIN_RATIO {det['pin_ratio']}",
                ]
                if det["pin"]
                else []
            )
            + ['#include "ttnn/ttnn/bringup/flat_routed_expert_ttnn/device/kernels/se_dyn.hpp"']
        )
    T = N * SF * K
    assert not ids_uint32 or cores is None or colmc, "ids_uint32: the one-core / column planners"
    assert not tile_in or ((cores is None or colmc) and K <= 16 and rows_cap * 8 >= 6144), "tile_in: K <= 16"
    in_bytes = ((T * (4 if ids_uint32 else 2) + 63) // 64) * 64 + ((T * 2 + 63) // 64) * 64 + 64
    stage_bytes = ((T // K + 31) // 32) * (1024 * (4 if ids_uint32 else 2) + 2048) + 64
    out_bytes = 2 * ((NG * 4 + 63) // 64) * 64 + rows_cap * 12 + 4096 + 64
    # (det, one-core planner: the schedule scratch, counts, entries and the order pages, cb 3)
    al_ = lambda v: (v + 63) // 64 * 64  # noqa: E731
    det_bytes = (
        4096 + al_(6 * NG) + (3 * N * VM + 32) * 4 + 2 * al_(N * VM) + al_(N * EPC) + A * det_page_bytes(S, K) + 64
        if det is not None
        else 0
    )
    c = ttnn.CoreCoord(*core)
    cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c)])
    if colmc:  # the column multi-core planner (P.11)
        M, CT = _mcc_geometry(S, N, len(cores))
        assert M <= len(cores) and N <= 32 and A <= 16 and K <= 16
        assert S < (1 << 16) and EPC < 256
        mcores = list(cores)[:M]
        mcs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*x), ttnn.CoreCoord(*x)) for x in mcores])
        vv = [mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(*x)) for x in mcores]
        pxy = [w for v in vv for w in (v.x, v.y)]
        idw = 4 if ids_uint32 else 2
        mcc_bytes = _mcc_l1_bytes(M, CT, N, S, K, EPC, NG, A, rows_cap, idw, det is not None, VM)
        tdbg_t = ttnn.from_torch(
            torch.zeros(R_m, C_m, M, 16, dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
    elif cores is not None:  # the multi-core planner
        M = len(cores)
        CH = (T + M - 1) // M
        GS = (EPC + M - 1) // M
        assert (S * K) % M == 0
        HS = S * K // M
        assert not row_pages or CH % K == 0
        assert not joint or row_pages
        HR = S if row_pages else 0  # (row pages: chunk ids + weights land in the bucket region, the home ids in the
        # (slice staging: 64 B a token row, ids and weights staged once each, or once joint)
        assert not row_pages or 2 * K <= 64
        buckb = _bucket_bytes(CH, K, row_pages, joint)

        V2 = os.environ.get("FC_PLAN_V", "2") == "2"
        assert V2 or (CH < (1 << 13) and GS <= 8), (CH, GS)
        mcs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*x), ttnn.CoreCoord(*x)) for x in cores])
        vv = [mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(*x)) for x in cores]
        pxy = [w for v in vv for w in (v.x, v.y)]
        # token-major rows (an expert's rows by token, then home): V2 without det, tokens split evenly over the cores
        TM = bool(token_major and V2 and det is None and S % M == 0)
        idb = N * ((S // M * K * 2 + 63) // 64 * 64 + 64) if TM else None
        mc_bytes = (
            _mc2_l1_bytes(M, CH, HS, EPC, NG, rows_cap, HR, idb, K, row_pages, joint)
            if V2
            else _mc_l1_bytes(M, CH, HS, EPC, NG, rows_cap)
        )
        if det is not None:
            assert V2, "det: the V2 multi-core planner"
            mc_bytes += _mc2_det_bytes(M, A, N, NG, VM, S, K)
        tdbg_t = ttnn.from_torch(
            torch.zeros(R, C, M, 16, dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )

    def build_mc(ids, wts, per_chip=False):
        mesh_desc, progs = ttnn.MeshProgramDescriptor(), {}
        for r in range(R):
            for cc in range(C):
                me = r * C + cc
                pd = ttnn.ProgramDescriptor()
                pd.cbs = [
                    ttnn.CBDescriptor(
                        total_size=mc_bytes,
                        core_ranges=mcs,
                        format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, mc_bytes)],
                    )
                ]
                pd.semaphores = [
                    ttnn.SemaphoreDescriptor(id=i, core_ranges=mcs, initial_value=0)
                    for i in ((0, 1, 2, 3) if det is not None else (0, 1, 2))
                ]
                rt = ttnn.RuntimeArgs()
                head = [
                    int(ids.buffer_address()),
                    int(wts.buffer_address()),
                    int(outs["counts"].buffer_address()),
                    int(outs["regions"].buffer_address()),
                    int(outs["token_index"].buffer_address()),
                    int(outs["table"].buffer_address()),
                    int(outs["plan"].buffer_address()),
                ] + list(port_luts[(r, cc)])
                rect = [min(v.x for v in vv), min(v.y for v in vv), max(v.x for v in vv), max(v.y for v in vv)]
                for pi, x in enumerate(cores):
                    rt[x[0]][x[1]] = (
                        head
                        + [pi]
                        + pxy
                        + [int(tdbg_t.buffer_address())]
                        + rect
                        + ([int(outs["dord"].buffer_address())] if det is not None else [])
                    )
                pd.kernels = [
                    ttnn.KernelDescriptor(
                        kernel_source=wsub(_PLANNER_MC2 if V2 else _PLANNER_MC)
                        .replace("__RPAGE__", str((4 if joint else 2) * K if row_pages else 0))
                        .replace("__BUCKB__", str(buckb))
                        .replace("__RJOINT__", str(int(joint)))
                        .replace("__RPT__", str(tokens_per_page))
                        .replace("__DET__", str(int(det is not None)))
                        .replace("__TM__", str(int(TM)))
                        .replace("__DPG__", str(det_page_bytes(S, K)))
                        .replace("__WR__", str(int(det.get("w_rows", 160)) if det is not None else 1))
                        .replace("__VM__", str(VM))
                        .replace("__DETINC__", detinc if det is not None else ""),
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=mcs,
                        compile_time_args=[me, R, C, S, K, EPC, NG, A, L, n_ports[(r, cc)], rows_cap, M, CH]
                        + ([HS, _mc2_stg(M, EPC, rows_cap, HR)] if V2 else [GS, HS]),
                        runtime_args=rt,
                        config=ttnn.DataMovementConfigDescriptor(
                            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                        ),
                    )
                ]
                mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, cc), ttnn.MeshCoordinate(r, cc))] = pd
                progs[(r, cc)] = pd
        return progs if per_chip else mesh_desc

    def build_mcc(ids, wts, per_chip=False):
        mesh_desc, progs = ttnn.MeshProgramDescriptor(), {}
        for r in range(R_m):
            for cc in range(C_m):
                eoff = int(id_offset(r, cc)) if id_offset is not None else 0
                pd = ttnn.ProgramDescriptor()
                pd.cbs = [
                    ttnn.CBDescriptor(
                        total_size=mcc_bytes,
                        core_ranges=mcs,
                        format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, mcc_bytes)],
                    )
                ]
                pd.semaphores = [
                    ttnn.SemaphoreDescriptor(id=i, core_ranges=mcs, initial_value=0)
                    for i in ((0, 1, 2, 3) if det is not None else (0, 1, 2))
                ]
                head = [
                    int(ids.buffer_address()),
                    int(wts.buffer_address()),
                    int(outs["counts"].buffer_address()),
                    int(outs["regions"].buffer_address()),
                    int(outs["token_index"].buffer_address()),
                    int(outs["table"].buffer_address()),
                    int(outs["plan"].buffer_address()),
                ] + list(port_luts[(r, cc)])
                rt = ttnn.RuntimeArgs()
                for pi, x in enumerate(mcores):
                    rt[x[0]][x[1]] = (
                        head
                        + [pi]
                        + pxy
                        + [int(tdbg_t.buffer_address())]
                        + ([int(outs["dord"].buffer_address())] if det is not None else [])
                    )
                pd.kernels = [
                    ttnn.KernelDescriptor(
                        kernel_source=wsub(_PLANNER_MCC)
                        .replace("__IDW__", str(idw))
                        .replace("__DET__", str(int(det is not None)))
                        .replace("__DPG__", str(det_page_bytes(S, K)))
                        .replace("__WR__", str(int(det.get("w_rows", 160)) if det is not None else 1))
                        .replace("__VM__", str(VM))
                        .replace("__DETINC__", detinc if det is not None else ""),
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=mcs,
                        compile_time_args=[r, N, S, K, EPC, NG, A, L, n_ports[(r, cc)], rows_cap, M, CT, eoff, SF, T0]
                        + [_mc2_stg(M, EPC, rows_cap), mcc_bytes],
                        runtime_args=rt,
                        config=ttnn.DataMovementConfigDescriptor(
                            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                        ),
                    )
                ]
                mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, cc), ttnn.MeshCoordinate(r, cc))] = pd
                progs[(r, cc)] = pd
        return progs if per_chip else mesh_desc

    def build(ids, wts, per_chip=False):
        if colmc:
            return build_mcc(ids, wts, per_chip=per_chip)
        if cores is not None:
            return build_mc(ids, wts, per_chip=per_chip)
        mesh_desc, progs = ttnn.MeshProgramDescriptor(), {}
        for r in range(R_m):
            for cc in range(C_m):
                me = r * C + cc if axis is None else r  # (axis=0: chip r of its column)
                eoff = int(id_offset(r, cc)) if id_offset is not None else 0
                pd = ttnn.ProgramDescriptor()
                pd.cbs = (
                    [
                        ttnn.CBDescriptor(
                            total_size=in_bytes,
                            core_ranges=cs,
                            format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, in_bytes)],
                        ),
                        ttnn.CBDescriptor(
                            total_size=out_bytes,
                            core_ranges=cs,
                            format_descriptors=[ttnn.CBFormatDescriptor(1, ttnn.uint32, out_bytes)],
                        ),
                    ]
                    + (
                        [  # (tile_in: every ids / weights tile staged at once)
                            ttnn.CBDescriptor(
                                total_size=stage_bytes,
                                core_ranges=cs,
                                format_descriptors=[ttnn.CBFormatDescriptor(2, ttnn.uint32, stage_bytes)],
                            )
                        ]
                        if tile_in
                        else []
                    )
                    + (
                        [  # (det: the order table's scratch)
                            ttnn.CBDescriptor(
                                total_size=det_bytes,
                                core_ranges=cs,
                                format_descriptors=[ttnn.CBFormatDescriptor(3, ttnn.uint32, det_bytes)],
                            )
                        ]
                        if det is not None
                        else []
                    )
                )
                rt = ttnn.RuntimeArgs()
                rt[c.x][c.y] = [
                    int(ids.buffer_address()),
                    int(wts.buffer_address()),
                    int(outs["counts"].buffer_address()),
                    int(outs["regions"].buffer_address()),
                    int(outs["token_index"].buffer_address()),
                    int(outs["table"].buffer_address()),
                    int(outs["plan"].buffer_address()),
                ] + list(port_luts[(r, cc)])
                if det is not None:
                    rt[c.x][c.y] = list(rt[c.x][c.y]) + [int(outs["dord"].buffer_address())]
                pd.kernels = [
                    ttnn.KernelDescriptor(
                        kernel_source=wsub(_PLANNER)
                        .replace("__DETINC1__", detinc if det is not None else "")
                        .replace("__DET1__", str(int(det is not None)))
                        .replace("__VM1__", str(VM))
                        .replace("__WR1__", str(int(det.get("w_rows", 160)) if det is not None else 1))
                        .replace("__DPG1__", str(det_page_bytes(S, K)))
                        .replace("__SKIPF__", "true" if skipf else "false")
                        .replace("__COLOPT__", str(int(skipf)))
                        .replace("__IDW__", "4" if ids_uint32 else "2")
                        .replace("__TILEIN__", "true" if tile_in else "false"),
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=cs,
                        compile_time_args=[me, R, C, S, K, EPC, NG, A, L, n_ports[(r, cc)], rows_cap]
                        + ([eoff, SF, T0] if skipf else []),
                        runtime_args=rt,
                        config=ttnn.DataMovementConfigDescriptor(
                            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                        ),
                    )
                ]
                mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, cc), ttnn.MeshCoordinate(r, cc))] = pd
                progs[(r, cc)] = pd
        return progs if per_chip else mesh_desc

    state = {}

    def run(ids, wts):
        k = (int(ids.buffer_address()), int(wts.buffer_address()))
        if state.get("key") != k:
            state["key"], state["desc"] = k, build(ids, wts)
            ttnn.freeze_program_hashes(state["desc"])  # (hash stored once: not re-hashed per call)
        ttnn.generic_op(
            [ids, wts, outs["counts"], outs["regions"], outs["token_index"], outs["table"], outs["plan"]]
            + ([outs["dord"]] if "dord" in outs and (cores is None or colmc) else []),
            state["desc"],
        )
        return outs

    run.outs = outs
    run.tdbg = tdbg_t if cores is not None else None
    run.token_major = bool(cores is not None and not colmc and TM)
    run.mcc = (M, CT) if colmc else None
    if colmc:  # (the per-chip programs: several passes' planners in one op, each on its own cores)
        run.programs = lambda ids, wts: build_mcc(ids, wts, per_chip=True)
        run.cores = [ttnn.CoreCoord(*x) for x in mcores]
        run.tensors = [outs["counts"], outs["regions"], outs["token_index"], outs["table"], outs["plan"]]
        run.tensors += [outs["dord"]] if "dord" in outs else []
    elif cores is not None:  # (the per-chip programs, for merging another op onto the cores it leaves)
        run.programs = lambda ids, wts: build_mc(ids, wts, per_chip=True)
        run.cores = [ttnn.CoreCoord(*x) for x in cores]
        run.tensors = [outs["counts"], outs["regions"], outs["token_index"], outs["table"], outs["plan"]]
    elif axis is not None or skipf:  # (the one-core planner's per-chip programs: several passes' planners in one op)
        run.programs = lambda ids, wts: build(ids, wts, per_chip=True)
        run.cores = [c]
        run.tensors = [outs["counts"], outs["regions"], outs["token_index"], outs["table"], outs["plan"]]
        run.tensors += [outs["dord"]] if "dord" in outs else []
    return run
