// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// tt/glued_derive.py's kernel (NC.7): verbatim but for the accessor block (@ACCESSORS@: one x_args for every
// output) and the section switches as defines (D_*, CKEEP_FAST, NOUT, ALIGN_ROWS, TAB_ARG, P<i>, HPAGE_BYTES)
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CHUNK = get_compile_time_arg_val(0);
constexpr uint32_t SP = get_compile_time_arg_val(1);
constexpr uint32_t TP = get_compile_time_arg_val(2);
constexpr uint32_t R = get_compile_time_arg_val(3);
constexpr uint32_t BLOCK =
    get_compile_time_arg_val(4);  // decode's kv block (PREFILL_END=cap: cap = end // BLOCK * BLOCK)
constexpr uint32_t NC = get_compile_time_arg_val(5);
constexpr uint32_t A0 = get_compile_time_arg_val(6);
constexpr uint32_t A1 = get_compile_time_arg_val(7);
constexpr uint32_t A2 = get_compile_time_arg_val(8);
constexpr uint32_t A3 = get_compile_time_arg_val(9);
constexpr uint32_t A4 = get_compile_time_arg_val(10);
constexpr uint32_t A5 = get_compile_time_arg_val(11);
constexpr uint32_t WIN = 128;
constexpr uint32_t SENT = 0xFFFFFFFFu;
constexpr uint32_t NSEG = 6;
constexpr auto h_args = TensorAccessorArgs<12>();
constexpr auto x_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
// NC.7: every output is DRAM interleaved (checked by the op): one accessor config for all (the Python route chained one
// TensorAccessorArgs a output, all equal); the names the sections use
constexpr auto o0_args = x_args;
constexpr auto o1_args = x_args;
constexpr auto vl0_args = x_args;
constexpr auto vlb0_args = x_args;
constexpr auto vlc0_args = x_args;
constexpr auto cap0_args = x_args;
constexpr auto tab_args = x_args;
constexpr uint32_t HPAGE = HPAGE_BYTES;

#ifndef ALIGN_ROWS
#define ALIGN_ROWS 0
#endif

struct Seg {
    uint32_t start, rows, off, lane, gws;
};

FORCE_INLINE void zero_words(uint32_t addr, uint32_t words) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < words; ++i) {
        p[i] = 0;
    }
}

void kernel_main() {
    const uint32_t hdr = get_arg_val<uint32_t>(0);
    const uint32_t q = get_arg_val<uint32_t>(1 + NOUT);
    const uint32_t k = get_arg_val<uint32_t>(2 + NOUT);
    (void)k;
    const uint32_t hoff = get_arg_val<uint32_t>(3 + NOUT);
    cb_reserve_back(0, 1);
    const uint32_t base = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t buf = base + 128;
    const auto H = TensorAccessor(h_args, hdr, HPAGE);
    noc_async_read(H.get_noc_addr(0, hoff), base, 128);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* hw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    Seg s[NSEG];
    uint32_t ns = 0, used = 0;
    const uint32_t n = hw[0] < NSEG ? hw[0] : NSEG;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t st = hw[2 + 4 * i], en = hw[3 + 4 * i], gw = hw[4 + 4 * i];
        const uint32_t cap = en / BLOCK * BLOCK;
        const uint32_t rows = cap > st ? cap - st : 0;
        // PB.9G (ALIGN_ROWS = rows a chip): a segment owns whole chips (its length rounded up, at least one chip)
        const uint32_t len = en > st ? en - st : 1;
        constexpr uint32_t AR = ALIGN_ROWS ? ALIGN_ROWS : 1;  // (no division by a zero constant)
        const uint32_t ext = ALIGN_ROWS ? (len + AR - 1) / AR * AR : rows;
        if (rows) {
            s[ns].start = st;
            s[ns].rows = rows;
            s[ns].off = used;
            s[ns].lane = i;
            s[ns].gws = gw;
            ++ns;
        }
        used += ext;
    }
    // this chip's entries (pass p = its p-th): segment, local rows [a, b)
    const uint32_t q0 = q * R;
    uint32_t ne = 0, ej[NSEG], ea[NSEG], eb[NSEG];
    for (uint32_t j = 0; j < ns; ++j) {
        const uint32_t lo = s[j].off, hi = s[j].off + s[j].rows;
        if (lo < q0 + R && hi > q0) {
            ej[ne] = j;
            ea[ne] = (lo > q0 ? lo : q0) - q0;
            eb[ne] = (hi < q0 + R ? hi : q0 + R) - q0;
            ++ne;
        }
    }
    volatile tt_l1_ptr uint32_t* u = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);

#if D_KEEP
    {
        const auto O0 = TensorAccessor(o0_args, get_arg_val<uint32_t>(1), P0);
        const auto O1 = TensorAccessor(o1_args, get_arg_val<uint32_t>(2), P1);
        volatile tt_l1_ptr uint16_t* t16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf);
        for (uint32_t t = 0; t < R / 32; ++t) {
            zero_words(buf, 512);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t i = 32 * t + r;  // (a row of a segment's computed rows; PB.9G: not its chip's pad)
                bool in = false;
                for (uint32_t e = 0; e < ne; ++e) {
                    in = in || (ea[e] <= i && i < eb[e]);
                }
                if (in) {
                    t16[((r >> 4) << 9) + ((r & 15) << 4)] = 0x3F80;
                }
            }
            noc_async_write(buf, O0.get_noc_addr(t), 2048);
            noc_async_write_barrier();
        }
        constexpr uint32_t W = CHUNK / SP;
        const uint32_t rr = q / TP;
        uint32_t nr = 0;  // (the SP row's real rows as a prefix: its last real row + 1, tt/packed.real_prefix)
        for (uint32_t j = 0; j < ns; ++j) {
            const uint32_t lo = s[j].off, hi = s[j].off + s[j].rows;
            if (lo < (rr + 1) * W && hi > rr * W) {
                const uint32_t e = (hi < (rr + 1) * W ? hi : (rr + 1) * W) - rr * W;
                nr = e > nr ? e : nr;
            }
        }
        u[0] = rr * W + nr;
        noc_async_write(buf, O1.get_noc_addr(0), 4);
        noc_async_write_barrier();
    }
#endif

#if D_REGION
    {
        const auto O0 = TensorAccessor(o0_args, get_arg_val<uint32_t>(1), P0);
        const uint32_t mask = get_arg_val<uint32_t>(4 + NOUT);
        const uint32_t lo = R * k / NC, hi = R * (k + 1) / NC;
        constexpr uint32_t BR = 8;
        for (uint32_t i0 = lo; i0 < hi; i0 += BR) {
            const uint32_t nb = (hi - i0) < BR ? (hi - i0) : BR;
            for (uint32_t b = 0; b < nb; ++b) {
                const uint32_t i = i0 + b;
                volatile tt_l1_ptr uint32_t* row = u + b * WIN;
                int32_t jj = -1;
                for (uint32_t e = 0; e < ne; ++e) {
                    if (ea[e] <= i && i < eb[e]) {
                        jj = static_cast<int32_t>(ej[e]);
                    }
                }
                if (jj < 0) {
                    row[0] = WIN + i;
                    for (uint32_t x = 1; x < WIN; ++x) {
                        row[x] = SENT;
                    }
                    continue;
                }
                const Seg& g = s[jj];
                const int32_t d = static_cast<int32_t>(q0 + i - g.off);
                const int32_t cls = ((mask >> jj) & 1u) ? 0 : static_cast<int32_t>(g.start < WIN ? g.start : WIN);
                int32_t x0 = d - static_cast<int32_t>(WIN - 1);
                if (x0 < -cls) {
                    x0 = -cls;
                }
                const int32_t carry0 = static_cast<int32_t>(WIN + R + WIN * g.lane + WIN);
                const int32_t own0 = static_cast<int32_t>(WIN) - static_cast<int32_t>(q0) + static_cast<int32_t>(g.off);
                for (uint32_t x = 0; x < WIN; ++x) {
                    const int32_t X = x0 + static_cast<int32_t>(x);
                    if (X > d) {
                        row[x] = SENT;
                    } else if (X < 0) {
                        row[x] = static_cast<uint32_t>(carry0 + X);
                    } else {
                        // own rows (mx >= q0): 128 + mx - q0; the previous chip's: mx - (q0 - 128): the same formula
                        row[x] = static_cast<uint32_t>(own0 + X);
                    }
                }
            }
            for (uint32_t b = 0; b < nb; ++b) {
                noc_async_write(buf + b * WIN * 4, O0.get_noc_addr(i0 + b), WIN * 4);
            }
            noc_async_write_barrier();
        }
    }
#endif

#if D_CKEEP && CKEEP_FAST
    {
        // PB.GAP (PACKED_CWRITE_FAST=on): the same tiles as below. Compressed row i is kept when its model row i RAT is
        // in a segment's computed rows [off, off + rows), i.e. i in [ceil(off / RAT), ceil((off + rows) / RAT)): a
        // tile's 32 rows as a bit mask (one OR of shifted runs a segment), only column 0 written (rows r < 16 at 16 r,
        // rows 16..31 at 512 + 16 (r - 16); the rest of the L1 batch is zeroed once), CK_B tiles behind one write
        // barrier (was: 2 KB zeroed, 32 x segments compares and a barrier a tile: 1.13 ms for 4096 rows on the 8x4)
        const auto O0 = TensorAccessor(o0_args, get_arg_val<uint32_t>(1), P0);
        constexpr uint32_t RAT = CHUNK / A0;  // (model rows a compressed row)
        constexpr uint32_t CK_B = 8;          // tiles a batch (8 x 2 KB: the CB's 16 KB behind the header slot)
        constexpr uint32_t NT = A0 / 32;
        uint32_t lo[NSEG], hi[NSEG];
        for (uint32_t j = 0; j < ns; ++j) {
            lo[j] = (s[j].off + RAT - 1) / RAT;
            hi[j] = (s[j].off + s[j].rows + RAT - 1) / RAT;
        }
        zero_words(buf, 512 * CK_B);
        for (uint32_t t0 = 0; t0 < NT; t0 += CK_B) {
            const uint32_t nb = (NT - t0) < CK_B ? (NT - t0) : CK_B;
            for (uint32_t b = 0; b < nb; ++b) {
                const uint32_t i0 = 32 * (t0 + b);
                uint32_t mask = 0;
                for (uint32_t j = 0; j < ns; ++j) {
                    const uint32_t a = lo[j] > i0 ? lo[j] - i0 : 0;
                    const uint32_t e = hi[j] > i0 ? (hi[j] - i0 < 32 ? hi[j] - i0 : 32) : 0;
                    if (a < e) {
                        const uint32_t run = e - a == 32 ? 0xFFFFFFFFu : ((1u << (e - a)) - 1u);
                        mask |= run << a;
                    }
                }
                volatile tt_l1_ptr uint16_t* t16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf + b * 2048);
                for (uint32_t r = 0; r < 16; ++r) {
                    t16[r << 4] = ((mask >> r) & 1u) ? 0x3F80 : 0;
                    t16[512 + (r << 4)] = ((mask >> (r + 16)) & 1u) ? 0x3F80 : 0;
                }
                noc_async_write(buf + b * 2048, O0.get_noc_addr(t0 + b), 2048);
            }
            noc_async_write_barrier();
        }
    }
#elif D_CKEEP
    {
        const auto O0 = TensorAccessor(o0_args, get_arg_val<uint32_t>(1), P0);
        volatile tt_l1_ptr uint16_t* t16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf);
        constexpr uint32_t RAT = CHUNK / A0;  // (model rows a compressed row)
        for (uint32_t t = 0; t < A0 / 32; ++t) {
            zero_words(buf, 512);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t m = (32 * t + r) * RAT;  // (a compressed row of a segment's computed rows)
                bool in = false;
                for (uint32_t j = 0; j < ns; ++j) {
                    in = in || (s[j].off <= m && m < s[j].off + s[j].rows);
                }
                if (in) {
                    t16[((r >> 4) << 9) + ((r & 15) << 4)] = 0x3F80;
                }
            }
            noc_async_write(buf, O0.get_noc_addr(t), 2048);
            noc_async_write_barrier();
        }
    }
#endif

#if D_IDX
    {
        // A0 the indexer's chunk, A1 its ratio, A2 the key capacity, A3 its block, A4 ctopk
        uint32_t lens[NSEG], nbv[NSEG];
        for (uint32_t j = 0; j < ns; ++j) {
            uint32_t v = (s[j].start + A0) / A1;
            v = (v + 31) / 32 * 32;
            lens[j] = v < A2 ? v : A2;
            nbv[j] = lens[j] / A3;
        }
        (void)nbv;
        for (uint32_t p = 0; p < NSEG; ++p) {
            const bool e = p < ne;
            const uint32_t j = e ? ej[p] : 0;
            u[0] = e ? lens[j] : 32;
            noc_async_write(buf, TensorAccessor(vl0_args, get_arg_val<uint32_t>(1 + p), 4).get_noc_addr(0), 4);
            noc_async_write_barrier();
#if D_BSEL
            // PB.BM (INDEXER_BLOCK_SELECT): the chip's block count of the block pre-selection (lens / A3; an idle
            // chip's 32)
            u[0] = (e ? lens[j] : 32) / A3;
            noc_async_write(buf, TensorAccessor(vlb0_args, get_arg_val<uint32_t>(7 + p), 4).get_noc_addr(0), 4);
            noc_async_write_barrier();
#endif
#if D_CAND
            const uint32_t v = e ? nbv[j] : 0;
            u[0] = (e && v >= A4) ? v : A4;
            noc_async_write(buf, TensorAccessor(vlc0_args, get_arg_val<uint32_t>(7 + p), 4).get_noc_addr(0), 4);
            noc_async_write_barrier();
            const uint32_t cv = (e && v < A4) ? v : 0;
            volatile tt_l1_ptr uint16_t* t16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf);
            const auto C = TensorAccessor(cap0_args, get_arg_val<uint32_t>(13 + p), 2048);
            for (uint32_t t = 0; t < A4 / 32; ++t) {
                zero_words(buf, 512);
                for (uint32_t c = 0; c < 32; ++c) {
                    t16[((c >> 4) << 8) + (c & 15)] = (32 * t + c < cv) ? 0x7F80 : 0xFF80;
                }
                noc_async_write(buf, C.get_noc_addr(t), 2048);
                noc_async_write_barrier();
            }
#endif
        }
        // the ChipRows piece tables (tt/glued_derive.slot): [count, (source row, destination row, rows) x count]
        zero_words(buf, 64 * 32);
        auto put = [&](uint32_t t, uint32_t np, const uint32_t* w) {
            volatile tt_l1_ptr uint32_t* tb = u + t * 32;
            tb[0] = np;
            for (uint32_t x = 0; x < 3 * np; ++x) {
                tb[1 + x] = w[x];
            }
        };
        for (uint32_t p = 0; p < NSEG; ++p) {
            const bool e = p < ne;
            const uint32_t j = e ? ej[p] : 0;
            const uint32_t a = e ? ea[p] : 0, b = e ? eb[p] : 0;
            const uint32_t offr = e ? s[j].off % R : 0;
            uint32_t w[6];
            const uint32_t c = e ? (R - offr) % R : 0;
            if (c) {
                w[0] = 0;
                w[1] = c;
                w[2] = R - c;
                w[3] = R - c;
                w[4] = 0;
                w[5] = c;
                put(p, 2, w);
            } else {
                w[0] = 0;
                w[1] = 0;
                w[2] = R;
                put(p, 1, w);
            }
            uint32_t k0[3] = {a, a, b - a};
            const uint32_t t = (a + R - offr) % R;
            const uint32_t first = (b - a) < (R - t) ? (b - a) : (R - t);
            uint32_t k1[6] = {t, a, first, 0, a + first, b - a - first};
            const uint32_t n1 = first < b - a ? 2 : 1;
            if (e) {
                put(6 + p, 1, k0);
                put(12 + p, n1, k1);
            }
#if D_CAND
            if (e) {
                const uint32_t L = nbv[j] >= A4 ? 1 : 0;
                put(18 + 12 * L + p, 1, k0);
                put(18 + 12 * L + 6 + p, n1, k1);
            }
#endif
        }
        const auto T = TensorAccessor(tab_args, get_arg_val<uint32_t>(TAB_ARG), 64 * 32 * 4);
        noc_async_write(buf, T.get_noc_addr(0), 64 * 32 * 4);
        noc_async_write_barrier();
    }
#endif

#if D_SRV
    {
        // A0 segments written; per-run values: their segment indices
        constexpr uint32_t W = CHUNK / SP;
        const auto O0 = TensorAccessor(o0_args, get_arg_val<uint32_t>(1), P0);
        const uint32_t rr = q / TP;
        for (uint32_t x = 0; x < A0; ++x) {
            const uint32_t j = get_arg_val<uint32_t>(4 + NOUT + x);
            // PB.9G: an encoder replay segment (gws > start) writes [gws, cap) as a chunk at gws: gws's server order,
            // its rows from gws - start on (TtCompressor._write_from's shifted rows)
            const uint32_t lo = s[j].gws > s[j].start ? s[j].gws : s[j].start;
            const uint32_t s0 = lo % CHUNK;
            const uint32_t intra = s0 % W, cst = (s0 / W) % SP;
            const uint32_t kk = (rr + SP - cst) % SP;
            const uint32_t off = s[j].off + (lo - s[j].start);
            for (uint32_t jj = 0; jj < W; ++jj) {
                const uint32_t d = kk == 0 ? (jj < W - intra ? jj : CHUNK - W + jj) : kk * W - intra + jj;
                u[jj] = (d + off) % CHUNK;
            }
            noc_async_write(buf, O0.get_noc_addr(0, x * W * 4), W * 4);
            noc_async_write_barrier();
        }
    }
#endif
}
