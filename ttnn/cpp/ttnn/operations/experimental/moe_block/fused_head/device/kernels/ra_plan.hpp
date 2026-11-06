// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The routing exchange's planner (ra::plan), parametrized by the RAP_* defines.
// A fragment the including kernel inlines: it relies on what that kernel declared before it (included once a
// kernel).
#pragma once

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/ra_accessors.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/combine_plan_words.hpp"

namespace ra {

constexpr uint32_t S = RAP_S, K = RAP_K, EPC = RAP_EPC, R = RAP_R, C = RAP_C, N = R * C, ME = RAP_ME;
constexpr uint32_t A = RAP_A, L = RAP_L, P = RAP_P, TPP = RAP_TPP, M = RAP_M, NG = N * EPC, E0 = ME * EPC;
constexpr uint32_t HBP = RAP_HBP, BB = RAP_BB, BBP = RAP_BBP, PCAP = RAP_PCAP, NPORT = RAP_NPORT;
constexpr uint32_t ROWS_CAP = RAP_ROWS_CAP, ENT_CAP = RAP_ENT_CAP;  // (ENT_CAP: an instance's entries, worst case)
constexpr uint32_t INST_ROWS = ENT_CAP + 32 * EPC < ROWS_CAP ? ENT_CAP + 32 * EPC : ROWS_CAP;  // (an instance's rows)
constexpr uint32_t ROWB = 4 * K, PB = TPP * ROWB, PSZ = (PB + 63) / 64 * 64, NPG = S / TPP;
constexpr uint32_t NOWN = (A + M - 1) / M;  // accumulators a core owns at most
constexpr uint32_t al16(uint32_t v) { return (v + 15) / 16 * 16; }
constexpr uint32_t al64(uint32_t v) { return (v + 63) / 64 * 64; }
constexpr uint32_t TSM = NOWN * ((S + A - 1) / A);  // my home tokens at most
constexpr uint32_t O_RT = 0, O_EX = al64(NPG * PSZ), O_NQ = O_EX + al64(4 * M * N), O_PRE = O_NQ + al64(4 * N);
constexpr uint32_t O_STG = O_PRE + N * BBP, O_HDR = O_STG + al64(4 * TSM * K + 16 * N);
constexpr uint32_t O_ENT = O_HDR + N * HBP, O_RUN = O_ENT + al64(4 * ENT_CAP + 128 * N * M);
constexpr uint32_t O_ET = O_RUN + al64(4 * N * M);
constexpr uint32_t O_TIDX = O_ET + al64(2 * EPC * S), O_TAB = O_TIDX + al64(4 * INST_ROWS);
constexpr uint32_t O_WANT = O_TAB + al64(8 * INST_ROWS), O_PIN = O_WANT + al64(128 * NOWN);
constexpr uint32_t O_PLAN = O_PIN + M * 64, O_CNT = O_PLAN + moe_block_plan_words::kPlanHeadBytes, O_RGN = O_CNT + al64(4 * (EPC + 32));
constexpr uint32_t O_ROWS = O_RGN + al64(4 * (EPC + 32));
constexpr uint32_t NOWND = (N + M - 1) / M;  // (destinations an instance owns the header totals of: d % M == q)
constexpr uint32_t O_CX = O_ROWS + 64, O_PSUM = O_CX + N * BB, O_TOT = O_PSUM + NOWND * M * BB;
constexpr uint32_t LMW = 1024;  // local-memory counters: per global expert, later per (my expert, token) if they fit
static_assert(NG <= LMW, "local counters: NG up to 1024");
static_assert(S <= 2048 && EPC <= 64, "an entry: token (11 bits) | expert (6 bits) | weight (15 bits)");
static_assert(A <= moe_block_plan_words::kAccMax && N <= moe_block_plan_words::kMaxChips, "the plan's want[A][N]");
constexpr auto rt_args = TensorAccessorArgs<RAP_CT0>();
constexpr auto sl_args = TensorAccessorArgs<rt_args.next_compile_time_args_offset()>();

// torus axes (rings: the shorter way; a tie as the fabric: by the source's parity - sparse_combine_l1.tie_mode 2)
constexpr uint32_t WX = RAP_WX, WY = RAP_WY;
FORCE_INLINE uint32_t first_hop(uint32_t src, uint32_t dst) {  // DIRS index: 0 X+, 1 X-, 2 Y+, 3 Y-; Y first, then X
    const uint32_t sr = src / C, sc = src % C, dr = dst / C, dc = dst % C;
    if (sr != dr) {
        if constexpr (WY) {  // (a tie: an odd source row goes Y+, an even one Y-)
            const uint32_t f2 = 2 * ((dr + R - sr) % R);
            return f2 < R || (f2 == R && (sr & 1)) ? 2 : 3;
        }
        return dr > sr ? 2 : 3;
    }
    if constexpr (WX) {
        return 2 * ((dc + C - sc) % C) < C || (2 * ((dc + C - sc) % C) == C && !(sc & 1)) ? 0 : 1;
    }
    return dc > sc ? 0 : 1;
}

FORCE_INLINE void plan(uint32_t a0, uint32_t B) {
    size_t a = a0;
    const uint32_t rt_addr = get_arg_val<uint32_t>(a++), sl_addr = get_arg_val<uint32_t>(a++);
    const uint32_t c0 = get_arg_val<uint32_t>(a++), c1 = get_arg_val<uint32_t>(a++);
    const uint32_t calls_addr = get_arg_val<uint32_t>(a++);
    const uint32_t cnt_addr = get_arg_val<uint32_t>(a++), reg_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tidx_addr = get_arg_val<uint32_t>(a++), tab_addr = get_arg_val<uint32_t>(a++);
    const uint32_t plan_addr = get_arg_val<uint32_t>(a++);
    uint32_t lut[8];
    for (uint32_t i = 0; i < 8; ++i) {
        lut[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t q = get_arg_val<uint32_t>(a++);
    a += 2;  // (static expert ranges: unused, the ranges follow the call's rows)
    uint32_t px[M], py[M];
    for (uint32_t i = 0; i < M; ++i) {
        px[i] = get_arg_val<uint32_t>(a++);
        py[i] = get_arg_val<uint32_t>(a++);
    }
    uint32_t bb[M], exs[M], gos[M];
    for (uint32_t i = 0; i < M; ++i) {
        bb[i] = get_arg_val<uint32_t>(a++);
    }
    for (uint32_t i = 0; i < M; ++i) {
        exs[i] = get_arg_val<uint32_t>(a++);
    }
    for (uint32_t i = 0; i < M; ++i) {
        gos[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t dst_arg = a;  // per destination chip: its port core (x, y) and its packet buffer's offset there
    a += 3 * N;
    const uint32_t port_arg = a;  // the port cores (x, y)
    a += 3 * NPORT;
    const uint32_t ready_sem = get_arg_val<uint32_t>(a++);  // (port cores')
    const uint32_t go_sem = get_arg_val<uint32_t>(a++), parts_sem = get_arg_val<uint32_t>(a++),
                   ex_sem = get_arg_val<uint32_t>(a++);
    // (the RISC's L1 data cache - off by default: every L1 load is then ~10+ cycles; write-through, so only NoC
    // arrivals need the invalidations this kernel already does; off again before returning, for the firmware)
    set_l1_data_cache<true>();

#ifndef RAP_SV0  // (an earlier token loop's planner plans no token: no rows to read)
    const auto rt = TensorAccessor(rt_args, rt_addr, PB);
    for (uint32_t pg = 0; pg < NPG; ++pg) {
        noc_async_read(rt.get_noc_addr(pg), B + O_RT + pg * PSZ, PB);
    }
#endif
#if defined(RT_VALID) && !defined(RAP_SV0)
    // (a padded chunk's actual_start / actual_end, into O_WANT: local scratch of step 4 - nothing writes it before)
    const uint32_t vl1 = rt_valid_slot(B + O_WANT);
    static_assert(O_PIN - O_WANT >= 128, "the scalars' slots: 64 B aligned in O_WANT, 128 B");
    rt_valid_issue(vl1);
#endif
    volatile tt_l1_ptr uint32_t* calls = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(calls_addr);
    const uint32_t par = calls[0] & 1;
    calls[0] = calls[0] + 1;
    alignas(4)
        uint16_t lm[LMW];  // (local memory: entries per global expert, then cursors; later per (my expert, token))
    for (uint32_t i = 0; i < NG; ++i) {
        lm[i] = 0;
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
#ifdef RAP_SV0
    // (token loops, an earlier loop: no token planned - the exchange runs empty, the chips' barrier between loops;
    // the last loop plans every loop's tokens)
    constexpr uint32_t SV = 0;
#elif defined(RT_VALID)
    // (my real tokens of every token loop: 0 .. SV - 1; the rest are padding, unplanned)
    const uint32_t SV = rt_valid_owned(vl1);
#else
    constexpr uint32_t SV = S;
#endif
    // (1) my home tokens (t % A owned by me): my entries per global expert
    auto row = [&](uint32_t t) __attribute__((always_inline)) {
        return reinterpret_cast<const uint32_t*>(B + O_RT + (t / TPP) * PSZ + (t % TPP) * ROWB);
    };
    uint32_t n_my = 0;
    for (uint32_t aa = q; aa < A; aa += M) {
        for (uint32_t t = aa; t < SV; t += A) {
            const uint32_t* r2 = row(t);
            for (uint32_t kk = 0; kk < K / 2; ++kk) {
                const uint32_t v = r2[kk];
                ++lm[v & 0xFFFF];
                ++lm[v >> 16];
            }
            n_my += K;
        }
    }
    // (2) my per-destination totals to every planner core
    uint32_t* nq = reinterpret_cast<uint32_t*>(B + O_NQ);
    for (uint32_t d = 0; d < N; ++d) {
        uint32_t s = 0;
        for (uint32_t e = 0; e < EPC; ++e) {
            s += lm[d * EPC + e];
        }
        nq[d] = s;
    }
    for (uint32_t r = 0; r < M; ++r) {
        noc_async_write(B + O_NQ, get_noc_addr(px[r], py[r], bb[r] + O_EX + q * N * 4), N * 4);
    }
    {  // my per-expert counts of destination d to its owner instance d % M (slot q): the owner sums the M slices'
       // into the packet header's totals - the sender then only sends
        uint16_t* cx = reinterpret_cast<uint16_t*>(B + O_CX);
        for (uint32_t d = 0; d < N; ++d) {
            for (uint32_t e = 0; e < BB / 2; ++e) {
                cx[d * (BB / 2) + e] = e < EPC ? lm[d * EPC + e] : 0;
            }
        }
        for (uint32_t d = 0, r = 0; d < N; ++d, r = r + 1 == M ? 0 : r + 1) {
            noc_async_write(B + O_CX + d * BB, get_noc_addr(px[r], py[r], bb[r] + O_PSUM + ((d / M) * M + q) * BB), BB);
        }
    }
    noc_async_write_barrier();
    for (uint32_t r = 0; r < M; ++r) {
        if (r != q) {
            noc_semaphore_inc(get_noc_addr(px[r], py[r], exs[r]), 1);
        }
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ex_sem), M - 1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ex_sem) = 0;
    invalidate_l1_cache();
    // (3) my entries sorted by global expert into staging (each destination's run aligned as its place in the packet,
    // mod 16), straight into the port cores' packet buffers
    const uint32_t* ex = reinterpret_cast<const uint32_t*>(B + O_EX);
    {
        uint32_t doff[N], stg[N];
        uint32_t ptr = B + O_STG;
        for (uint32_t d = 0; d < N; ++d) {
            uint32_t o = 0;
            for (uint32_t r = 0; r < q; ++r) {
                o += ex[r * N + d];
            }
            doff[d] = HBP + 4 * o;
            ptr += (doff[d] - ptr) & 15u;
            stg[d] = ptr;
            uint16_t* pre = reinterpret_cast<uint16_t*>(B + O_PRE + d * BBP);
            uint32_t c = 0, w = (ptr - (B + O_STG)) / 4;
            for (uint32_t e = 0; e < EPC; ++e) {
                pre[e] = static_cast<uint16_t>(c);
                const uint32_t n = lm[d * EPC + e];
                lm[d * EPC + e] = static_cast<uint16_t>(w);  // (cursor: word index in the staging)
                w += n;
                c += n;
            }
            pre[EPC] = static_cast<uint16_t>(c);
            ptr += 4 * c;
        }
        uint32_t* sw = reinterpret_cast<uint32_t*>(B + O_STG);
        for (uint32_t aa = q; aa < A; aa += M) {
            for (uint32_t t = aa; t < SV; t += A) {
                const uint32_t* r2 = row(t);
                for (uint32_t kk = 0; kk < K / 2; ++kk) {  // (token | expert << 11 | weight's 15 low bits << 17)
                    const uint32_t v = r2[kk], w = r2[K / 2 + kk], g0 = v & 0xFFFF, g1 = v >> 16;
                    sw[lm[g0]++] = t | ((g0 % EPC) << 11) | ((w & 0x7FFF) << 17);
                    sw[lm[g1]++] = t | ((g1 % EPC) << 11) | ((w >> 16) << 17);
                }
            }
        }
        for (uint32_t d = q; d < N; d += M) {  // (my destinations' header totals: the M slices' counts, two 16-bit
                                               // lanes a word - a lane's sum <= the destination's entries <= S K)
            const uint32_t* cs = reinterpret_cast<const uint32_t*>(B + O_PSUM + (d / M) * M * BB);
            uint32_t* tw = reinterpret_cast<uint32_t*>(B + O_TOT + (d / M) * BB);
            for (uint32_t i = 0; i < BB / 4; ++i) {
                uint32_t acc = 0;
                for (uint32_t r = 0; r < M; ++r) {
                    acc += cs[r * (BB / 4) + i];
                }
                tw[i] = acc;
            }
        }
        for (uint32_t d = 0; d < N; ++d) {
            const uint32_t x = get_arg_val<uint32_t>(dst_arg + 3 * d), y = get_arg_val<uint32_t>(dst_arg + 3 * d + 1);
            const uint32_t pb = get_arg_val<uint32_t>(dst_arg + 3 * d + 2);  // (the port's CB 0: the same base)
            if (nq[d]) {
                noc_async_write(stg[d], get_noc_addr(x, y, pb + doff[d]), 4 * nq[d]);
            }
            noc_async_write(B + O_PRE + d * BBP, get_noc_addr(x, y, pb + 64 + BB + q * BBP), BBP);
            noc_inline_dw_write(get_noc_addr(x, y, pb + 4 * q), nq[d]);
            if (d % M == q) {
                noc_async_write(B + O_TOT + (d / M) * BB, get_noc_addr(x, y, pb + 64), BB);
            }
        }
        noc_async_write_barrier();
        for (uint32_t i = 0; i < NPORT; ++i) {
            const uint32_t x = get_arg_val<uint32_t>(port_arg + 3 * i), y = get_arg_val<uint32_t>(port_arg + 3 * i + 1);
            noc_semaphore_inc(get_noc_addr(x, y, get_arg_val<uint32_t>(port_arg + 3 * i + 2)), 1);
        }
    }
    // (4) while the packets fly: my home tokens' rows (local: I am their home) - n_local / want / n_ret; want rows
    // straight to the plan, n_local to core 0
    uint32_t nr[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    {
        uint8_t hop[N * L];  // n_ret's port: the first hop toward a source, its link
        for (uint32_t s = 0; s < N; ++s) {
            for (uint32_t l = 0; l < L; ++l) {
                hop[s * L + l] = s == ME ? 0 : static_cast<uint8_t>(lut[first_hop(ME, s) * 2 + l]);
            }
        }
        uint32_t* wr = reinterpret_cast<uint32_t*>(B + O_WANT);
        const auto pgg = TensorAccessor(ra_acc::plan, plan_addr, moe_block_plan_words::kPlanBytes);
        for (uint32_t i = 0, aa = q; aa < A; ++i, aa += M) {
            uint32_t want[N], nloc = 0;
            for (uint32_t s = 0; s < N; ++s) {
                want[s] = 0;
            }
            for (uint32_t t = aa; t < SV; t += A) {
                const uint32_t* r2 = row(t);
                for (uint32_t kk = 0; kk < K / 2; ++kk) {
                    const uint32_t v = r2[kk], s0 = (v & 0xFFFF) / EPC, s1 = (v >> 16) / EPC;
                    if (s0 == ME) {
                        ++nloc;
                    } else {
                        ++want[s0];
                        ++nr[hop[s0 * L + (aa + s0 + ME) % L]];
                    }
                    if (s1 == ME) {
                        ++nloc;
                    } else {
                        ++want[s1];
                        ++nr[hop[s1 * L + (aa + s1 + ME) % L]];
                    }
                }
            }
            for (uint32_t s = 0; s < moe_block_plan_words::kMaxChips; ++s) {
                wr[i * 32 + s] = s < N ? want[s] : 0;
            }
            noc_async_write(
                B + O_WANT + i * 128,
                pgg.get_noc_addr(0) + (moe_block_plan_words::kPlanWant + aa * moe_block_plan_words::kMaxChips) * 4,
                128);
            noc_inline_dw_write(
                get_noc_addr(px[0], py[0], bb[0] + O_PLAN + (moe_block_plan_words::kPlanNloc + aa) * 4), nloc);
        }
    }
    // (5) every packet of the call: core 0 waits, then wakes the others
    {
        DeviceZoneScopedN("SE-RPWAIT");
        if (q == 0) {
            const uint32_t ca = par ? c1 : c0;
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ca), N);
            noc_semaphore_inc(get_noc_addr(ca), 0u - N);
            for (uint32_t r = 1; r < M; ++r) {
                noc_semaphore_inc(get_noc_addr(px[r], py[r], gos[r]), 1);
            }
        } else {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem), 1);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem) = 0;
        }
    }
    // (6) the headers (every expert's total -> regions; my experts: a contiguous range balanced by rows), then every
    // packet's entries
    const auto sl = TensorAccessor(sl_args, sl_addr, PCAP);
    for (uint32_t o = 0; o < N; ++o) {
        noc_async_read(sl.get_noc_addr(par * N + o), B + O_HDR + o * HBP, HBP);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    uint32_t tot[EPC], reg[EPC], ent[N];
    // (every expert's total over the headers: two 16-bit lanes a word - a lane's sum <= N S, one entry a token)
    static_assert(N * S < 65536, "packed expert totals");
    constexpr uint32_t EW = (EPC + 1) / 2;
    uint32_t tw[EW];
    for (uint32_t i = 0; i < EW; ++i) {
        tw[i] = 0;
    }
    for (uint32_t o = 0; o < N; ++o) {
        const uint32_t* ht =
            reinterpret_cast<const uint32_t*>(B + O_HDR + o * HBP + 64);  // (EPC uint16: BB >= 2 EW words)
        for (uint32_t i = 0; i < EW; ++i) {
            tw[i] += ht[i];
        }
    }
    for (uint32_t e = 0; e < EPC; ++e) {
        tot[e] = (tw[e / 2] >> (16 * (e & 1))) & 0xFFFF;
    }
    uint32_t off = 0, T = 0;
    for (uint32_t e = 0; e < EPC; ++e) {
        T += tot[e];
        reg[e] = off;
        off += (tot[e] + 31) / 32 * 32;
    }
    // my experts: from boundary q to q + 1 - boundary j the expert boundary nearest j T / M entries (whole experts: the
    // nearest, not the next - an instance's work is its entries)
    auto bound = [&](uint32_t j) __attribute__((always_inline)) -> uint32_t {
        if (j == 0) {
            return 0;
        }
        if (j >= M) {
            return EPC;
        }
        for (uint32_t e = 0, c = 0; e < EPC; ++e) {
            const uint32_t c1 = c + tot[e];
            if (c1 * M >= j * T) {
                return 2 * j * T <= (c + c1) * M ? e : e + 1;
            }
            c = c1;
        }
        return EPC;
    };
    const uint32_t e0 = bound(q), e1 = bound(q + 1);
    const uint32_t R0 = e0 < EPC ? reg[e0] : off, R1 = e1 < EPC ? reg[e1] : off;
    // my experts' entries only: per source, per slice, its run of my experts (a slice is sorted by expert) - each read
    // keeps its source's 64 B alignment; run (o, r)'s entry x at run[o M + r] + 4 x. An instance's entries are at most
    // T / M + one expert's (the boundaries are the nearest expert edges): ENT_CAP, the build's worst case - packed
    // routing included (the rest of the chip's entries are other instances')
    uint32_t* run = reinterpret_cast<uint32_t*>(B + O_RUN);
    {
        uint32_t nall = 0, pos = B + O_ENT;
        for (uint32_t o = 0; o < N; ++o) {
            const uint32_t hb = B + O_HDR + o * HBP;
            const uint32_t* nqo = reinterpret_cast<const uint32_t*>(hb);
            ent[o] = 0;
            for (uint32_t r = 0, base = 0; r < M; base += nqo[r], ++r) {
                const uint16_t* pre = reinterpret_cast<const uint16_t*>(hb + 64 + BB + r * BBP);
                const uint32_t x0 = base + pre[e0], n = pre[e1] - pre[e0];
                run[o * M + r] = 0;
                if (!n || nall + n > ENT_CAP) {
                    continue;  // (past the worst case: dropped - off > ROWS_CAP or P_OVER)
                }
                const uint32_t src = HBP + 4 * x0, a0 = src & ~63u, len = (src - a0 + 4 * n + 63) & ~63u;
                noc_async_read(sl.get_noc_addr(par * N + o) + a0, pos, len);
                run[o * M + r] = pos + (src - a0) - 4 * x0;
                ent[o] = 1;
                pos += len;
                nall += n;
            }
        }
    }
    // my experts' counts / regions (core 0: the 64-byte span of the page that holds them, the rest of the span other
    // chips' experts: zero) - now, under the entries' reads and the fill
    const uint32_t b0 = (E0 * 4) & ~63u, b1 = ((E0 + EPC) * 4 + 63) & ~63u;
    if (q == 0) {
        uint32_t* cw = reinterpret_cast<uint32_t*>(B + O_CNT);
        uint32_t* rw = reinterpret_cast<uint32_t*>(B + O_RGN);
        for (uint32_t i = 0; i < (b1 - b0) / 4; ++i) {
            cw[i] = rw[i] = 0;
        }
        for (uint32_t e = 0; e < EPC; ++e) {
            cw[E0 + e - b0 / 4] = tot[e];
            rw[E0 + e - b0 / 4] = reg[e];
        }
        noc_async_write(B + O_CNT, TensorAccessor(ra_acc::counts, cnt_addr, NG * 4).get_noc_addr(0) + b0, b1 - b0);
        noc_async_write(B + O_RGN, TensorAccessor(ra_acc::regions, reg_addr, NG * 4).get_noc_addr(0) + b0, b1 - b0);
    }
    const uint32_t W1c = R1 < ROWS_CAP ? R1 : ROWS_CAP, W1 = W1c < R0 + INST_ROWS ? W1c : R0 + INST_ROWS;
    const uint32_t ER = e1 - e0;
    uint16_t* et = ER * S <= LMW ? lm : reinterpret_cast<uint16_t*>(B + O_ET);  // per (my expert, token)
    {  // (32-bit stores: et is 4-byte aligned, S even)
        uint32_t* e32 = reinterpret_cast<uint32_t*>(et);
        for (uint32_t i = 0; i < ER * S / 2; ++i) {
            e32[i] = 0;
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    // my experts' entries: sources in order, each source's slices in order, each slice's run of my experts (one
    // contiguous range: a slice is sorted by expert); an entry carries its token and expert
    // rows: per expert, ordered by (token, home, k): count per (expert, token), prefix, then each entry its row
    for (uint32_t o = 0; o < N; ++o) {
        if (!ent[o]) {
            continue;
        }
        const uint32_t hb = B + O_HDR + o * HBP;
        const uint32_t* nqo = reinterpret_cast<const uint32_t*>(hb);
        for (uint32_t r = 0, base = 0; r < M; base += nqo[r], ++r) {
            if (!run[o * M + r]) {
                continue;
            }
            const uint32_t* eo = reinterpret_cast<const uint32_t*>(run[o * M + r]);
            const uint16_t* pre = reinterpret_cast<const uint16_t*>(hb + 64 + BB + r * BBP);
            const uint32_t x1 = base + pre[e1];
            for (uint32_t x = base + pre[e0]; x < x1; ++x) {
                const uint32_t w = eo[x], e = ((w >> 11) & 63) - e0, t = w & 0x7FF;
                if (e >= ER || t >= S) {
                    continue;  // (a slice's run holds my experts' entries only: never - bounds et on bad data)
                }
                ++et[e * S + t];
            }
        }
    }
    for (uint32_t e = e0; e < e1; ++e) {
        uint16_t* ete = et + (e - e0) * S;
        uint32_t c = reg[e] - R0;
        for (uint32_t t = 0; t < S; ++t) {
            const uint32_t n = ete[t];
            ete[t] = static_cast<uint16_t>(c);
            c += n;
        }
    }
    uint32_t* tidx = reinterpret_cast<uint32_t*>(B + O_TIDX);
    uint32_t* tab = reinterpret_cast<uint32_t*>(B + O_TAB);
    uint32_t ninj[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t o = 0; o < N; ++o) {
        if (!ent[o]) {
            continue;
        }
        const uint32_t hb = B + O_HDR + o * HBP;
        const uint32_t* nqo = reinterpret_cast<const uint32_t*>(hb);
        const uint32_t os = o * S, lk = ME + o;
        uint32_t hp[L];  // (the port toward home o, per link)
        for (uint32_t l = 0; l < L; ++l) {
            hp[l] = o == ME ? 0 : lut[first_hop(ME, o) * 2 + l];
        }
        for (uint32_t r = 0, base = 0; r < M; base += nqo[r], ++r) {
            if (!run[o * M + r]) {
                continue;
            }
            const uint32_t* eo = reinterpret_cast<const uint32_t*>(run[o * M + r]);
            const uint16_t* pre = reinterpret_cast<const uint16_t*>(hb + 64 + BB + r * BBP);
            const uint32_t x1 = base + pre[e1];
            for (uint32_t x = base + pre[e0]; x < x1; ++x) {
                const uint32_t w = eo[x], t = w & 0x7FF, e = (w >> 11) & 63;
                if (e - e0 >= ER || t >= S) {
                    continue;  // (as the count above)
                }
                const uint32_t rr = et[(e - e0) * S + t]++;
                if (rr >= INST_ROWS || R0 + rr >= ROWS_CAP) {
                    continue;  // (past the worst case only: never)
                }
                const uint32_t aa = t % A;
                uint32_t dest;
                if (o == ME) {
                    dest = P + aa;
                } else {
                    dest = hp[(aa + lk) % L];
                    ++ninj[dest];
                }
                tidx[rr] = os + t;
                tab[2 * rr] = dest | ((w >> 17) << 16);
                tab[2 * rr + 1] = o | (t << 8) | (e << 24);
            }
        }
    }
    for (uint32_t e = e0; e < e1; ++e) {  // (padding rows: zero)
        const uint32_t nx = e + 1 < EPC ? reg[e + 1] : off, p1 = nx < W1 ? nx : W1;
        for (uint32_t r = reg[e] + tot[e]; r < p1; ++r) {
            tidx[r - R0] = 0;
            tab[2 * (r - R0)] = 0;
            tab[2 * (r - R0) + 1] = 0;
        }
    }
    if (W1 > R0) {
        const auto tg = TensorAccessor(ra_acc::table, tab_addr, 256);
        noc_async_write(
            B + O_TIDX,
            TensorAccessor(ra_acc::token_index, tidx_addr, ROWS_CAP * 4).get_noc_addr(0) + R0 * 4,
            (W1 - R0) * 4);
        for (uint32_t pg = R0 / 32; pg < W1 / 32; ++pg) {
            noc_async_write(B + O_TAB + (pg - R0 / 32) * 256, tg.get_noc_addr(pg), 256);
        }
    }
    // (7) n_inj / n_ret partials to core 0; core 0 writes counts / regions, the plan's head words (n_inj, n_ret,
    // n_local) and its rows block (combine_plan_words.hpp)
    {
        uint32_t* pin = reinterpret_cast<uint32_t*>(B + O_PIN + q * 64);
        for (uint32_t i = 0; i < 8; ++i) {
            pin[i] = ninj[i];
            pin[8 + i] = nr[i];
        }
        if (q != 0) {
            noc_async_write(B + O_PIN + q * 64, get_noc_addr(px[0], py[0], bb[0] + O_PIN + q * 64), 64);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(px[0], py[0], parts_sem), 1);
        }
    }
    if (q == 0) {
        noc_async_write_barrier();  // (my own n_local inline writes, into my plan words)
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(parts_sem), M - 1);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(parts_sem) = 0;
        invalidate_l1_cache();
        uint32_t* plan = reinterpret_cast<uint32_t*>(B + O_PLAN);  // (n_local[]: from the owners)
        for (uint32_t i = 0; i < moe_block_plan_words::kPlanNloc; ++i) {
            plan[i] = 0;
        }
        for (uint32_t aa = A; aa < moe_block_plan_words::kPlanWant - moe_block_plan_words::kPlanNloc; ++aa) {
            plan[moe_block_plan_words::kPlanNloc + aa] = 0;  // (no accumulator: no owner writes it)
        }
        for (uint32_t r = 0; r < M; ++r) {
            const uint32_t* pr = reinterpret_cast<const uint32_t*>(B + O_PIN + r * 64);
            for (uint32_t i = 0; i < 16; ++i) {
                plan[i] += pr[i];
            }
        }
        uint32_t* rows = reinterpret_cast<uint32_t*>(B + O_ROWS);  // (the plan's rows block: 16 words)
        for (uint32_t i = 0; i < 16; ++i) {
            rows[i] = 0;
        }
        rows[8] = off;
        rows[9] = off > ROWS_CAP ? 1 : 0;
        const uint64_t plan_noc =
            TensorAccessor(ra_acc::plan, plan_addr, moe_block_plan_words::kPlanBytes).get_noc_addr(0);
        noc_async_write(B + O_PLAN, plan_noc, moe_block_plan_words::kPlanHeadBytes);
        noc_async_write(B + O_ROWS, plan_noc + moe_block_plan_words::kPlanRowsBlock * 4, 64);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    set_l1_data_cache<false>();
}
}  // namespace ra
