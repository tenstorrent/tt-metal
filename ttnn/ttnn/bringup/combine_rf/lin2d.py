# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""lin2d - y [M, N] = x [M, K] @ W [K, N] for M = 32 x (grid rows) rows (K3: 320; bf16 x, bf8 W): a 2D multicast
matmul laid out around what the Blackhole NoC / DRAM measured (zz_scratch_mcast_bw.py, zz_scratch_dram_mix.py,
12 x 10 grid):
  - multicast: one sender per column 52 GB/s, two on one NoC into one column 29 (they collide), NOC0 + NOC1 100;
  - a row multicast and a column multicast on the same NoC stall each other (crossing at every core: stalls of
    tens of us) -> x is delivered first, then W (XBAR);
  - DRAM reads on NOC1 from a row: ~100 GB/s vs ~385 on NOC0 -> every DRAM read is the BRISC's (NOC0); a NOC1
    sender's pieces are read by its core's BRISC and handed over in L1 (the LANDED / SENT semaphores);
  - ~390 GB/s is the ceiling for cores away from the DRAM (pinning a reader to one bank does not help).

  grid     GY rows x GX columns; core (cx, cy): output row tile cy x PN(cx) tile columns from n0(cx), in NP passes of
           PNP(cx) columns (<= 8 fp32 DST tiles, full sync), each accumulated over all of K and packed once
  x        resident in L1 (CB 0, K / 32 tiles of row cy), multicast once along the row by its two ends (left core
           NOC0 even K blocks, right core NOC1 odd: XS = 2), before any W
  W        step i = p NBLK + j (pass p, K block j of KB tiles): the KB x PNP tiles of column cx's pass-p columns; each
           column's steps go out as two streams, q0 (even steps, NOC0 multicast by a BRISC) and q1 (odd, NOC1 by an
           NCRISC, read by its core's BRISC), into the W ring (CB 1, NB slots); NSTW pieces read ahead; credits per
           ring slot (receivers -> the stream's reader). A stream's pieces sit in one DRAM bank in order (a bank-sharded
           tensor per group of equal-width columns, streams balanced over the banks). Readers (w_streams): the
           column's top (q0) / bottom (q1) core, the ones that are a bank's optimal core (or its row neighbour) reading
           that bank (L2D_NEAR, default; -8% time). L2D_FEED=1 makes the 16 bank-adjacent cores the readers of all 24
           streams (multicasting from outside the column): correct but slower - the in-transit routes (NOC0 east then
           south, NOC1 north then west) pile up to 5 streams on a link, e.g. every q1 along the bottom row
  RISCs    BRISC (NOC0): DRAM reads, the NOC0 sends, the receive loop (valid counters -> CB pushes, credits);
           NCRISC (NOC1): the NOC1 sends, then the y writes (bf16 TILE, interleaved DRAM or L1)
K3 (bh-lb-09): latent down 7168 -> 3584 93 us, up 3584 -> 7168 89 us standalone (ttnn mm2d 117 / 122); bound by the
DRAM reads. PROBE bits (L2D_PROBE) skip pieces for diagnosis; 512
records a per-core event trace (zz_scratch_k3_dense_mm.py::_lin2d_trace).
"""

import collections
import os

import torch

import ttnn

_COMMON = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t KT = get_compile_time_arg_val(0), KB = get_compile_time_arg_val(1), PNPX = get_compile_time_arg_val(2);
constexpr uint32_t NBLK = get_compile_time_arg_val(3), NP = get_compile_time_arg_val(4), NB = get_compile_time_arg_val(5);
constexpr uint32_t GX = get_compile_time_arg_val(6), GY = get_compile_time_arg_val(7), WTB = get_compile_time_arg_val(8);
constexpr uint32_t PNP = get_compile_time_arg_val(9), NT = get_compile_time_arg_val(10);
constexpr uint32_t PROBE = get_compile_time_arg_val(11);  // (1: no DRAM reads, 4: 64 B multicasts)
constexpr uint32_t NBANK = get_compile_time_arg_val(12), NST = get_compile_time_arg_val(13);
// x senders per row (XS): 2 = both ends, alternating (left NOC0 even, right NOC1 odd); 1 = the right core on NOC1
constexpr uint32_t XS = get_compile_time_arg_val(14), Q1S = get_compile_time_arg_val(15);  // (Q1S: q1 slots' offset)
// XBAR: the W multicasts start once every core of the column has all of x (XDQ: count of them, on the W readers), so
// the x row multicasts never cross W column multicasts on a NoC
constexpr uint32_t XBAR = get_compile_time_arg_val(16), NSTW = get_compile_time_arg_val(17);  // (NST: x staging slots)
// WPRE = 0: the W senders read nothing before every core of the column has x (the x reads get the DRAM to themselves)
constexpr uint32_t WPRE = get_compile_time_arg_val(18), WBMAX = get_compile_time_arg_val(21);
// GLU: the weight's tile columns are gate / up pairs; the output (NT tiles a row) is the activation, PNP / 2 a pass
constexpr uint32_t GLU = get_compile_time_arg_val(19), PO = GLU ? PNP / 2 : PNP;
// RES: y = r + x @ W (r [M, N] bf16 TILE: the NCRISC reads the core's tiles into CB 5, compute starts DST from them)
constexpr uint32_t RES = get_compile_time_arg_val(20);
constexpr auto x_args = TensorAccessorArgs<22>();
constexpr auto y_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr auto d_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();  // (PROBE 512: event trace)
constexpr auto r_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
constexpr uint32_t XB = KB * 2048, WB = KB * PNPX * WTB, NS = NP * NBLK, WPT = KB * PNPX;
// W streams: each column's steps go out as two streams, q0 (even steps, NOC0 multicast by a BRISC) and q1 (odd, NOC1
// by an NCRISC, read by its core's BRISC); a stream's reader core holds its pieces in one DRAM bank, in order
// (bank-adjacent readers: L2D_FEED=1, or the column's top / bottom core). Staging (CB 9, grid-wide): q0 slots
// [0, NSTW), q1 [Q1S, Q1S + NSTW) (Q1S = NSTW when a core can read both, else 0), WBMAX apart (the largest piece)
// semaphores: XV0 / XV1 x valid (per x sender), WV0 / WV1 W valid (per W stream of the column), LANDED / SENT the
// BRISC -> NCRISC hand-over (x 4 / 6, W 5 / 7), XDQ0 / XDQ1 (on a W stream's reader: cores of its column with all of x),
// WC (on a W stream's reader: credits per own ring slot, NB / 2 each for its q0 and q1 stream)
constexpr uint32_t XV = 0, WV = 2, XDQ = 8, WCB = 10;
// multicast value words (16 B apart) in CB 10: [0, 128) the BRISC's, [128, 256) the NCRISC's
constexpr uint32_t LANDED = 4, SENT = 6;
FORCE_INLINE volatile tt_l1_ptr uint32_t* sem(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(i));
}
FORCE_INLINE volatile tt_l1_ptr uint32_t* word(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); }
// PROBE 512: wall clock (low word) per event into CB 11 (BRISC entries [0, 256), NCRISC [256, 512)), dumped at the end
FORCE_INLINE void tr(uint32_t i) {
    if constexpr (PROBE & 512) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(11))[i] =
            *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    }
}
FORCE_INLINE void tr_dump(uint32_t d_addr, uint32_t page, uint32_t half) {
    if constexpr (PROBE & 512) {
        const auto dg = TensorAccessor(d_args, d_addr, 2048);
        noc_async_write(get_write_ptr(11) + half * 1024, dg.get_noc_addr(page, half * 1024), 1024);
        noc_async_write_barrier();
    }
}
// sender q's k-th own piece (x block / W step) and own count, S senders (S = 1: q = the only one)
constexpr uint32_t own_at(uint32_t S, uint32_t q, uint32_t k) { return S == 2 ? 2 * k + q : k; }
constexpr uint32_t n_own(uint32_t S, uint32_t n, uint32_t q) { return S == 2 ? (n + 1 - q) / 2 : n; }
// the sender (q) and own index (k) of piece i
constexpr uint32_t q_of(uint32_t S, uint32_t q1, uint32_t i) { return S == 2 ? (i & 1) : q1; }
constexpr uint32_t k_of(uint32_t S, uint32_t i) { return S == 2 ? (i >> 1) : i; }
// W credits per ring slot (a shared counter would let a fast core's credits stand in for a slow one's): the q stream's
// k-th piece (step s = 2 k + q) goes once every core of its column freed slot s % NB for the (s / NB)-th time
FORCE_INLINE bool w_credit_ok(uint32_t q, uint32_t k) {
    const uint32_t st = 2 * k + q;
    if (XBAR && *sem(XDQ + q) < GY) return false;
    return st < NB || (PROBE & 128) || *sem(WCB + q * (NB / 2) + (st % NB) / 2) >= GY * (st / NB);
}
"""

_BRISC = (
    _COMMON
    + r"""
// one stream read by this BRISC into staging slots (a read transaction id each); sent by this BRISC (send) or by the
// NCRISC (feed: landed / sent exchanged through the LANDED / SENT semaphores)
struct Stream {
    bool on = false, send = false, isw = false;
    uint32_t nst = NST, q = 0, n = 0, issued = 0, landed = 0, sent = 0, stage = 0, stride = 0, piece = 0, trid0 = 0;
    uint32_t tcx = 0, w_addr = 0, bank = 0, o = 0;  // (W: the target column's NoC x, the pieces' tensor / bank / start)
};

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t cx = get_arg_val<uint32_t>(2), cy = get_arg_val<uint32_t>(3);
    const uint32_t lx = get_arg_val<uint32_t>(4), rx = get_arg_val<uint32_t>(5), ry = get_arg_val<uint32_t>(6);  // row
    const uint32_t ty = get_arg_val<uint32_t>(7), by = get_arg_val<uint32_t>(8), cxv = get_arg_val<uint32_t>(9);  // col
    // the readers of this column's q0 / q1 streams (credits and the x-done count go to them)
    const uint32_t s0x = get_arg_val<uint32_t>(12), s0y = get_arg_val<uint32_t>(13);
    const uint32_t s1x = get_arg_val<uint32_t>(14), s1y = get_arg_val<uint32_t>(15);
    const auto xg = TensorAccessor(x_args, x_addr, 2048);
    const uint32_t xres = get_write_ptr(0), wring = get_write_ptr(1), words = get_write_ptr(10);
    // x: the left core sends the even blocks (NOC0), the right core feeds the odd ones to its NCRISC (NOC1)
    Stream S[3];
    Stream& X = S[0];
    X.on = (XS == 2 && cx == 0) || cx == GX - 1, X.send = XS == 2 && cx == 0, X.q = X.send ? 0 : 1;
    X.n = n_own(XS, NBLK, X.q);
    X.stage = get_write_ptr(8), X.stride = XB, X.piece = XB, X.trid0 = 1;
    // W streams this core reads: q0 (sent here, NOC0) and q1 (handed to the NCRISC); args per stream at 16 / 23:
    // on, target column NoC x, tensor address, piece bytes, bank, first piece, count
    for (uint32_t q = 0; q < 2; ++q) {
        Stream& T = S[1 + q];
        const uint32_t a = 16 + 7 * q;
        T.on = get_arg_val<uint32_t>(a), T.send = q == 0, T.isw = true, T.q = q, T.nst = NSTW;
        T.tcx = get_arg_val<uint32_t>(a + 1), T.w_addr = get_arg_val<uint32_t>(a + 2), T.piece = get_arg_val<uint32_t>(a + 3);
        T.bank = get_arg_val<uint32_t>(a + 4), T.o = get_arg_val<uint32_t>(a + 5), T.n = get_arg_val<uint32_t>(a + 6);
        T.stage = get_write_ptr(9) + q * Q1S * WBMAX, T.stride = WBMAX, T.trid0 = 1 + NST + q * NSTW;
    }
    uint32_t xr = 0, r = 0, c = NB;  // (x blocks / W steps pushed; next W step to credit)
    tr(255);
    auto streams_busy = [&]() {
        for (auto& T : S)
            if (T.on && (T.send ? T.sent < T.n : T.landed < T.n)) return true;
        return false;
    };
    while (xr < NBLK || r < NS || c < NS || streams_busy()) {
        for (auto& T : S) {
            if (!T.on) continue;
            if (!T.send) {  // (the NCRISC's progress: the staging slots it has multicast and flushed)
                invalidate_l1_cache();
                T.sent = *sem(SENT + T.isw);
            }
            while (T.issued < T.n && T.issued < T.sent + T.nst && (WPRE || !T.isw || *sem(XDQ + T.q) >= GY)) {
                const uint32_t k = T.issued, slot = k % T.nst, dst = T.stage + slot * T.stride;
                if (T.send && !(PROBE & 256)) noc_async_writes_flushed();  // (the slot's earlier multicast has left)
                noc_async_read_set_trid(T.trid0 + slot);
                if (!(PROBE & 1)) {
                    if (T.isw) {
                        if (!(PROBE & 16)) noc_async_read(get_noc_addr_from_bank_id<true>(T.bank, T.w_addr + (T.o + k) * T.piece), dst, T.piece);
                    } else if (!(PROBE & 8)) {
                        const uint32_t j = own_at(XS, T.q, k);
                        for (uint32_t t = 0; t < KB; ++t) noc_async_read(xg.get_noc_addr(cy * KT + j * KB + t), dst + t * 2048, 2048);
                    }
                }
                ++T.issued;
            }
            while (T.landed < T.issued && ncrisc_noc_read_with_transaction_id_flushed(noc_index, T.trid0 + T.landed % T.nst)) {
                if (!T.isw) tr(224 + T.landed);
                ++T.landed;
                if (!T.send) *sem(LANDED + T.isw) = T.landed;
            }
            if (T.send && T.sent < T.landed) {
                const uint32_t k = T.sent, slot = k % T.nst;
                invalidate_l1_cache();
                if (!T.isw || w_credit_ok(0, k)) {
                    const uint32_t src = T.stage + slot * T.stride, bytes = (PROBE & (T.isw ? 36 : 68)) ? 64 : T.piece;
                    const uint32_t vw = words + (T.isw ? 64 : 0) + (k & 3) * 16;
                    *word(vw) = k + 1;
                    if (T.isw) {  // (into the target column; loopback when this core is in it)
                        const uint64_t d = get_noc_multicast_addr(T.tcx, ty, T.tcx, by, wring + ((2 * k) % NB) * T.piece);
                        const uint64_t v = get_noc_multicast_addr(T.tcx, ty, T.tcx, by, get_semaphore(WV));
                        if (T.tcx == cxv) {
                            noc_async_write_multicast_loopback_src(src, d, bytes, GY);
                            noc_semaphore_set_multicast_loopback_src(vw, v, GY);
                        } else {
                            noc_async_write_multicast(src, d, bytes, GY);
                            noc_semaphore_set_multicast(vw, v, GY);
                        }
                    } else {
                        noc_async_write_multicast_loopback_src(src, get_noc_multicast_addr(lx, ry, rx, ry, xres + own_at(XS, 0, k) * XB), bytes, GX);
                        noc_semaphore_set_multicast_loopback_src(vw, get_noc_multicast_addr(lx, ry, rx, ry, get_semaphore(XV)), GX);
                    }
                    tr((T.isw ? 160 : 192) + T.sent);
                    ++T.sent;
                }
            }
        }
        // receive: x blocks (resident, pass 0 only) and W steps in order
        invalidate_l1_cache();
        if (xr < NBLK && *sem(XV + q_of(XS, 1, xr)) >= k_of(XS, xr) + 1) {
            cb_reserve_back(0, KB);
            cb_push_back(0, KB);
            tr(64 + xr);
            if (++xr == NBLK && XBAR) {
                noc_semaphore_inc(get_noc_addr(s0x, s0y, get_semaphore(XDQ)), 1);
                noc_semaphore_inc(get_noc_addr(s1x, s1y, get_semaphore(XDQ + 1)), 1);
            }
        }
        if (r < NS && *sem(WV + (r & 1)) >= (r >> 1) + 1) {
            cb_reserve_back(1, WPT);
            cb_push_back(1, WPT);
            tr(r);
            ++r;
        }
        // credit step c (>= NB) to its stream's reader once the slot of step c - NB is free here
        if (c < NS && c < r + NB && cb_pages_reservable_at_back(1, WPT * (c - r + 1))) {
            const uint32_t q = c & 1;
            noc_semaphore_inc(get_noc_addr(q ? s1x : s0x, q ? s1y : s0y, get_semaphore(WCB + q * (NB / 2) + (c % NB) / 2)), 1);
            tr(96 + c - NB);
            ++c;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    tr_dump(get_arg_val<uint32_t>(10), cy * GX + cx, 0);
}
"""
)

_NCRISC = (
    _COMMON
    + r"""
void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t cx = get_arg_val<uint32_t>(1), cy = get_arg_val<uint32_t>(2), n0 = get_arg_val<uint32_t>(3);
    // NOC1 multicasts start at the far end: the row from the right core to the left, the column bottom to top
    const uint32_t rx = get_arg_val<uint32_t>(4), lx = get_arg_val<uint32_t>(5), ry = get_arg_val<uint32_t>(6);
    const uint32_t by = get_arg_val<uint32_t>(7), ty = get_arg_val<uint32_t>(8), cxv = get_arg_val<uint32_t>(9);
    const auto yg = TensorAccessor(y_args, y_addr, 2048);
    const uint32_t xres = get_write_ptr(0), wring = get_write_ptr(1), words = get_write_ptr(10);
    const uint32_t xst = get_write_ptr(8), wst = get_write_ptr(9);
    if constexpr (RES) {  // (the residual's tiles of every pass, up front: the y tiles' positions)
        const auto rg = TensorAccessor(r_args, get_arg_val<uint32_t>(11), 2048);
        cb_reserve_back(5, NP * PNP);
        const uint32_t l1 = get_write_ptr(5);
        for (uint32_t t = 0; t < NP * PNP; ++t) noc_async_read(rg.get_noc_addr(cy * NT + n0 + t), l1 + t * 2048, 2048);
        noc_async_read_barrier();
        cb_push_back(5, NP * PNP);
    }
    // the odd x blocks (right core) / this core's q1 W stream (args 12..14: target column NoC x, piece bytes, count):
    // the BRISC reads them into the staging
    const bool xo = cx == GX - 1;
    const uint32_t tcx = get_arg_val<uint32_t>(12), wb = get_arg_val<uint32_t>(13), nw = get_arg_val<uint32_t>(14);
    const uint32_t nx = xo ? n_own(XS, NBLK, 1) : 0;
    uint32_t xs = 0, ws = 0;
    while (xs < nx || ws < nw) {
        invalidate_l1_cache();
        if (xs < nx && xs < *sem(LANDED)) {
            const uint32_t vw = words + 128 + (xs & 3) * 16;
            *word(vw) = xs + 1;
            noc_async_write_multicast_loopback_src(xst + (xs % NST) * XB, get_noc_multicast_addr(rx, ry, lx, ry, xres + own_at(XS, 1, xs) * XB),
                                                   (PROBE & 68) ? 64 : XB, GX);
            noc_semaphore_set_multicast_loopback_src(vw, get_noc_multicast_addr(rx, ry, lx, ry, get_semaphore(XV + 1)), GX);
            if (!(PROBE & 256)) noc_async_writes_flushed();
            tr(256 + 192 + xs);
            *sem(SENT) = ++xs;
        }
        if (ws < nw && ws < *sem(LANDED + 1) && w_credit_ok(1, ws)) {
            const uint32_t vw = words + 192 + (ws & 3) * 16;
            *word(vw) = ws + 1;
            const uint32_t src = wst + (Q1S + ws % NSTW) * WBMAX, bytes = (PROBE & 36) ? 64 : wb;
            const uint64_t d = get_noc_multicast_addr(tcx, by, tcx, ty, wring + ((2 * ws + 1) % NB) * wb);
            const uint64_t v = get_noc_multicast_addr(tcx, by, tcx, ty, get_semaphore(WV + 1));
            if (tcx == cxv) {
                noc_async_write_multicast_loopback_src(src, d, bytes, GY);
                noc_semaphore_set_multicast_loopback_src(vw, v, GY);
            } else {
                noc_async_write_multicast(src, d, bytes, GY);
                noc_semaphore_set_multicast(vw, v, GY);
            }
            if (!(PROBE & 256)) noc_async_writes_flushed();
            tr(256 + 160 + ws);
            *sem(SENT + 1) = ++ws;
        }
    }
    for (uint32_t p = 0; p < NP; ++p) {  // (y: NP passes x PNP tiles)
        cb_wait_front(4, PO);
        const uint32_t l1 = get_read_ptr(4);
        for (uint32_t t = 0; t < PO; ++t) noc_async_write(l1 + t * 2048, yg.get_noc_addr(cy * NT + n0 / (GLU + 1) + p * PO + t), 2048);
        noc_async_write_barrier();
        cb_pop_front(4, PO);
    }
    noc_async_write_barrier();
    tr_dump(get_arg_val<uint32_t>(10), cy * GX + cx, 1);
}
"""
)

_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
// L2D_GLU (defined for the gate / up linear): SiTU-GLU 4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25) over the pass's
// DST tiles (gate t, up t + 1 -> t) on the PACK thread's SFPU, as the flat expert's se3_compute
#ifdef L2D_RES
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#endif
#if defined(L2D_GLU) && defined(TRISC_PACK)
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "ckernel_sfpu_situ_glu.h"
#endif
constexpr uint32_t KB = get_compile_time_arg_val(0), PNP = get_compile_time_arg_val(1), PNPX = get_compile_time_arg_val(2);
constexpr uint32_t NBLK = get_compile_time_arg_val(3), NP = get_compile_time_arg_val(4), PROBE = get_compile_time_arg_val(5);
void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 4);  // (matmul: in1 -> SrcA, in0 -> SrcB)
    matmul_block_init(0, 1, 0, PNP, 1, KB);
#ifdef L2D_GLU
    PACK((SFPU_BINARY_INIT_FN_NO_ARGS(situ_glu, sfpu::situ_glu_init)));
    constexpr uint32_t PO = PNP / 2;
#else
    constexpr uint32_t PO = PNP;
#endif
    for (uint32_t p = 0; p < NP; ++p) {
#ifdef L2D_RES
        // DST starts from the residual's tiles (unpack srcA: W's format -> bf16, then the matmul state again)
        reconfig_data_format_srca(1, 5);
        copy_tile_to_dst_init_short(5);
        tile_regs_acquire();
        cb_wait_front(5, (p + 1) * PNP);
        for (uint32_t t = 0; t < PNP; ++t) copy_tile(5, p * PNP + t, t);
        reconfig_data_format_srca(5, 1);
        matmul_block_init(0, 1, 0, PNP, 1, KB);
#else
        tile_regs_acquire();
#endif
        for (uint32_t j = 0; j < NBLK; ++j) {
            if (p == 0) cb_wait_front(0, (j + 1) * KB);  // (x: resident, block j arrives in pass 0)
            cb_wait_front(1, KB * PNPX);
            for (uint32_t k = 0; k < KB && !(PROBE & 2); ++k) matmul_block(0, 1, j * KB + k, k * PNPX, 0, false, PNP, 1, KB);
            cb_pop_front(1, KB * PNPX);
        }
        tile_regs_commit();
        cb_reserve_back(4, PO);
        tile_regs_wait();
#ifdef L2D_GLU
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
        for (uint32_t t = 0; t < PNP; t += 2) {
            PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_situ_glu,
                                   (DST_ACCUM_MODE, 8, sfpu::SituGluConfigKimi), t, t + 1, t, VectorMode::RC)));
        }
        for (uint32_t t = 0; t < PNP; t += 2) pack_tile(t, 4);
#else
        for (uint32_t t = 0; t < PNP; ++t) pack_tile(t, 4);
#endif
        tile_regs_release();
        cb_push_back(4, PO);
    }
    cb_pop_front(0, KB * NBLK);
#ifdef L2D_RES
    cb_pop_front(5, NP * PNP);
#endif
}
"""


def column_split(Nt, GX, unit):
    """Per-column output tiles: multiples of `unit` (the pass count), as even as possible."""
    assert Nt % unit == 0, (Nt, unit)
    base, rem = divmod(Nt // unit, GX)
    return [(base + (1 if i < rem else 0)) * unit for i in range(GX)]


def w_streams(mesh_device, GX, GY, NS, feed, nbank, col_bytes):
    """The W streams (column cx, q: q0 the even steps, q1 the odd) -> reader core, bank, piece count. feed: the readers
    next to the DRAM - each bank's optimal worker (NOC0) and its row neighbour, streaming only that bank (98% of the
    DRAM in zz_scratch_dram_mix.py::test_bank_readers); the streams balanced over the banks (min NoC hops), a reader
    holding at most one q0 and one q1 stream. Else the column's top (q0) / bottom
    (q1) core, bank (2 cx + q) % banks."""
    streams = [dict(cx=cx, q=q, n=(NS + 1 - q) // 2) for cx in range(GX) for q in (0, 1)]
    if not feed:  # (column ends; a column end that is a bank's reader core (or its row neighbour) reads that bank)
        near = {}
        if os.environ.get("L2D_NEAR", "1") == "1":
            for b, c in enumerate(mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)):
                near.setdefault((c.x, c.y), b)
                near.setdefault((c.x + 1, c.y), b)
        # balanced by bytes (the call is DRAM bound: the busiest bank sets its time); a column end keeps its own bank
        # only while that bank stays within the even share
        for st_ in streams:
            st_["bytes"] = st_["n"] * col_bytes[st_["cx"]]
        share = sum(st_["bytes"] for st_ in streams) / nbank
        load = collections.Counter()
        within = os.environ.get("L2D_NEAR_COL", "0") == "1"  # (also a bank core elsewhere in the column: no gain)
        for st_ in streams:
            end = (st_["cx"], 0 if st_["q"] == 0 else GY - 1)
            st_["reader"] = end
            # the in-column bank cores, least transit to the stream's multicast start first (q0: south to row 0,
            # q1: north to the last row; both wrap over the two non-worker rows)
            hops = lambda y: ((0 - y) if st_["q"] == 0 else (y - (GY - 1))) % (GY + 2)  # noqa: E731
            cands = [(st_["cx"], y) for y in range(GY) if (st_["cx"], y) in near and (within or (st_["cx"], y) == end)]
            for c in sorted(cands, key=lambda c: hops(c[1])):
                if load[near[c]] + st_["bytes"] <= share * (1 + 1e-9):
                    st_["reader"], st_["bank"] = c, near[c]
                    load[near[c]] += st_["bytes"]
                    break
        # the rest, largest first, to the least loaded bank (ties: nearest the rotation (2 cx + q) % banks)
        for st_ in sorted((t for t in streams if "bank" not in t), key=lambda t: -t["bytes"]):
            b0 = (2 * st_["cx"] + st_["q"]) % nbank
            st_["bank"] = min(range(nbank), key=lambda b: (load[b], (b - b0) % nbank))
            load[st_["bank"]] += st_["bytes"]
        return streams
    opt = list(mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0))
    assert len(opt) == nbank
    readers = {}
    for b, c in enumerate(opt):
        nx = c.x + 1 if c.x + 1 < GX else c.x - 1
        readers[b] = [(c.x, c.y), (nx, c.y)]
        assert all(0 <= y < GY for _, y in readers[b]), ("a bank reader outside the grid rows", readers[b], GY)
    # min-cost assignment (networkx flow): stream -> (reader, q) slot -> bank (<= ceil(streams / banks) each). Cost: the
    # NoC hops from the reader to the column's multicast start - NOC0 runs east then south (to the top row), NOC1 west
    # then north (to the bottom row), both wrapping (a q1 stream from a reader east of its column, a q0 from one west)
    import networkx as nx

    W_, H_ = GX + 3, GY + 2  # (the torus around the worker grid: DRAM / eth columns, rows; approximate)
    cost = lambda st_, r: (  # noqa: E731
        10 * ((st_["cx"] - r[0]) % W_) + (0 - r[1]) % H_
        if st_["q"] == 0
        else 10 * ((r[0] - st_["cx"]) % W_) + (r[1] - (GY - 1)) % H_
    )
    cap = -(-len(streams) // nbank)
    g = nx.DiGraph()
    for i, st_ in enumerate(streams):
        g.add_edge("s", ("st", i), capacity=1, weight=0)
        for b in range(nbank):
            for r in readers[b]:
                g.add_edge(("st", i), ("slot", r, st_["q"]), capacity=1, weight=cost(st_, r))
    for b in range(nbank):
        for r in readers[b]:
            for q in (0, 1):
                g.add_edge(("slot", r, q), ("bank", b), capacity=1, weight=0)
        g.add_edge(("bank", b), "t", capacity=cap, weight=0)
    flow = nx.max_flow_min_cost(g, "s", "t")
    for i, st_ in enumerate(streams):
        ((_, r, _q),) = [k for k, v in flow[("st", i)].items() if v]
        st_["reader"], st_["bank"] = r, next(b for b in range(nbank) if r in readers[b])
    return streams


def lin2d_prepare(
    mesh_device,
    *,
    M,
    K,
    N,
    kb=8,
    nb=4,
    nst=3,
    nstw=None,
    xs=None,
    ws=None,
    fidelity=None,
    w_dtype=ttnn.bfloat8_b,
    out_dtype=ttnn.bfloat16,
    glu=False,
    residual=False,
):
    """Returns (run, layout): run(x [M, K] bf16 TILE, upload_stacked(mesh, W, layout)) -> y [M, N] TILE DRAM."""
    grid = mesh_device.compute_with_storage_grid_size()
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    GX, GY = grid.x, Mt
    assert M % 32 == 0 and 2 <= Mt <= grid.y and nb % 2 == 0 and nb <= 8
    NP = 1
    G = 2 if glu else 1  # (glu: N = 2 I weight columns as gate / up tile pairs, y [M, I] = the activation)
    while (Nt % (NP * G)) or max(column_split(Nt, GX, NP * G)) // NP > 8:  # (a pass: <= 8 fp32 DST tiles, full sync)
        NP += 1
    pns = column_split(Nt, GX, NP * G)
    assert Kt % kb == 0, (Kt, kb)
    nblk = Kt // kb
    wtb = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.bfloat16: 2048}[w_dtype]
    fid = fidelity if fidelity is not None else getattr(ttnn.MathFidelity, os.environ.get("L2D_FID", "HiFi2"))
    nstw = nstw if nstw is not None else int(os.environ.get("L2D_NSTW", "4"))
    assert nst + 2 * nstw <= 15  # (a read transaction id per staging slot)
    xb = kb * 2048
    allc = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
    vv = {
        (cx, cy): mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(cx, cy))
        for cx in range(GX)
        for cy in range(GY)
    }
    vv_any = lambda c: mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(*c))  # noqa: E731
    n0s = [sum(pns[:i]) for i in range(GX)]
    groups = {}
    for cx, pn in enumerate(pns):
        groups.setdefault(pn, []).append(cx)
    st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    xs = xs if xs is not None else int(os.environ.get("L2D_XS", "2"))
    ws = ws if ws is not None else int(os.environ.get("L2D_WS", "2"))
    xbar = int(os.environ.get("L2D_XBAR", "1"))
    wpre = int(os.environ.get("L2D_WPRE", "1"))
    nst = int(os.environ.get("L2D_NST", str(nst)))  # (W multicasts wait until every core of the column has its x)
    assert xs in (1, 2) and nb <= 6
    probe = int(os.environ.get("L2D_PROBE", "0"))  # (1: no DRAM reads, 2: no matmuls, 4: 64 B multicasts)
    state = {}
    nbank = mesh_device.dram_grid_size().x
    feed = int(os.environ.get("L2D_FEED", "0"))
    NS = NP * nblk
    gidx = {cx: i for i, (pn, cols) in enumerate(sorted(groups.items())) for cx in cols}
    wbs = [kb * (pn // NP) * wtb for pn, _ in sorted(groups.items())]
    streams = w_streams(mesh_device, GX, GY, NS, feed, nbank, [wbs[gidx[cx]] for cx in range(GX)])
    wbmax = max(wbs)
    # a stream's pieces sit in its bank in order, after the earlier streams of its column group in that bank
    used = {}
    for st_ in streams:
        key = (gidx[st_["cx"]], st_["bank"])
        st_["o"], used[key] = used.get(key, 0), used.get(key, 0) + st_["n"]
    layout = dict(
        pns=pns, NP=NP, kb=kb, banks=nbank, xs=xs, groups=sorted(groups.items()), streams=streams, NS=NS, feed=feed
    )
    by_reader = {}
    for st_ in streams:
        by_reader.setdefault(st_["reader"], {})[st_["q"]] = st_
    q1s = nstw if any(len(v) == 2 for v in by_reader.values()) else 0  # (q1 staging overlays q0's unless shared)
    reader_of = {(st_["cx"], st_["q"]): st_["reader"] for st_ in streams}

    def build(x, wl, y, dbg, res):
        program = ttnn.ProgramDescriptor()
        fmt = lambda i, dt, page: [ttnn.CBFormatDescriptor(i, dt, page)]  # noqa: E731
        # the grid-wide CBs first (resident x: the same address across a row), then the W ring / output / W staging per
        # column group (exact piece sizes; the ring's address only has to match down a column). (A grid-wide CB after
        # per-group ones of different sizes is placed after the first group's only: it overlapped the wider group's.)
        program.cbs = [
            ttnn.CBDescriptor(total_size=Kt * 2048, core_ranges=allc, format_descriptors=fmt(0, ttnn.bfloat16, 2048)),
            ttnn.CBDescriptor(total_size=nst * xb, core_ranges=allc, format_descriptors=fmt(8, ttnn.bfloat16, 2048)),
            ttnn.CBDescriptor(total_size=512, core_ranges=allc, format_descriptors=fmt(10, ttnn.uint32, 512)),
            ttnn.CBDescriptor(
                total_size=(nstw + q1s) * wbmax, core_ranges=allc, format_descriptors=fmt(9, w_dtype, wtb)
            ),
        ] + (
            [ttnn.CBDescriptor(total_size=2048, core_ranges=allc, format_descriptors=fmt(11, ttnn.uint32, 2048))]
            if probe & 512
            else []
        )
        for pn, cols in sorted(groups.items()):
            crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, 0), ttnn.CoreCoord(cx, GY - 1)) for cx in cols])
            wbg = kb * (pn // NP) * wtb
            program.cbs += [
                ttnn.CBDescriptor(total_size=nb * wbg, core_ranges=crs, format_descriptors=fmt(1, w_dtype, wtb)),
                ttnn.CBDescriptor(
                    total_size=NP * (pn // NP // G) * 2048,
                    core_ranges=crs,
                    format_descriptors=fmt(4, out_dtype, 2048),
                ),
            ] + (
                [
                    ttnn.CBDescriptor(
                        total_size=pn * 2048, core_ranges=crs, format_descriptors=fmt(5, ttnn.bfloat16, 2048)
                    )
                ]
                if residual
                else []
            )
        program.semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=allc, initial_value=0) for i in range(10 + nb)]
        x_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        y_ct = list(ttnn.TensorAccessorArgs(y).get_compile_time_args())
        d_ct = list(ttnn.TensorAccessorArgs(dbg).get_compile_time_args())
        r_ct = list(ttnn.TensorAccessorArgs(res if res is not None else y).get_compile_time_args())
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
            processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
        )  # noqa: E731
        kernels = []
        for gi_, (pn, cols) in enumerate(sorted(groups.items())):
            w = wl[gi_]
            crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, 0), ttnn.CoreCoord(cx, GY - 1)) for cx in cols])
            pnp = pn // NP
            ct = (
                [Kt, kb, pnp, nblk, NP, nb, GX, GY, wtb, pnp, Nt // G, probe, nbank, nst]
                + [xs, q1s, xbar, nstw, wpre, int(glu), int(residual), wbmax]
                + x_ct
                + y_ct
                + d_ct
                + r_ct
            )
            b_rt, n_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for cx in cols:
                for cy in range(GY):
                    lx, rx, ry = vv[(0, cy)].x, vv[(GX - 1, cy)].x, vv[(cx, cy)].y
                    ty, by, cxv = vv[(cx, 0)].y, vv[(cx, GY - 1)].y, vv[(cx, cy)].x
                    ba = [
                        int(x.buffer_address()),
                        int(w.buffer_address()),
                        cx,
                        cy,
                        lx,
                        rx,
                        ry,
                        ty,
                        by,
                        cxv,
                        int(dbg.buffer_address()),
                        cols.index(cx),
                    ]
                    for q in (0, 1):  # (this column's stream readers)
                        rd = vv_any(reader_of[(cx, q)])
                        ba += [rd.x, rd.y]
                    mine = by_reader.get((cx, cy), {})
                    for q in (0, 1):  # (the streams this core reads)
                        st_ = mine.get(q)
                        if st_ is None:
                            ba += [0] * 7
                        else:
                            ba += [
                                1,
                                vv[(st_["cx"], 0)].x,
                                int(wl[gidx[st_["cx"]]].buffer_address()),
                                wbs[gidx[st_["cx"]]],
                                st_["bank"],
                                st_["o"],
                                st_["n"],
                            ]
                    na = [
                        int(y.buffer_address()),
                        cx,
                        cy,
                        n0s[cx],
                        rx,
                        lx,
                        ry,
                        by,
                        ty,
                        cxv,
                        int(dbg.buffer_address()),
                        int(res.buffer_address()) if res is not None else 0,
                    ]
                    st1 = by_reader.get((cx, cy), {}).get(1)
                    na += [vv[(st1["cx"], 0)].x, wbs[gidx[st1["cx"]]], st1["n"]] if st1 else [0, 0, 0]
                    b_rt[cx][cy], n_rt[cx][cy] = ba, na
                    c_rt[cx][cy] = []
            comp = ttnn.ComputeConfigDescriptor()
            comp.math_fidelity = fid
            comp.fp32_dest_acc_en = True
            comp.dst_full_sync_en = True  # (8 fp32 DST tiles: a pass's output row)
            comp.math_approx_mode = False
            kernels += [
                ttnn.KernelDescriptor(
                    kernel_source=_BRISC,
                    source_type=st,
                    core_ranges=crs,
                    compile_time_args=ct,
                    runtime_args=b_rt,
                    config=dm("RISCV_0", ttnn.NOC.NOC_0),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=_NCRISC,
                    source_type=st,
                    core_ranges=crs,
                    compile_time_args=ct,
                    runtime_args=n_rt,
                    config=dm("RISCV_1", ttnn.NOC.NOC_1),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=("#define L2D_GLU 1\n" if glu else "")
                    + ("#define L2D_RES 1\n" if residual else "")
                    + _COMPUTE,
                    source_type=st,
                    core_ranges=crs,
                    compile_time_args=[kb, pnp, pnp, nblk, NP, probe],
                    runtime_args=c_rt,
                    config=comp,
                ),
            ]
        program.kernels = kernels
        R, C = tuple(mesh_device.shape)
        mesh_desc = ttnn.MeshProgramDescriptor()
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = program
        return mesh_desc

    def run(x, w, out_memory_config=None, residual_tensor=None):
        assert (residual_tensor is not None) == bool(residual)
        res = residual_tensor
        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([M, N // G]),
            out_dtype,
            ttnn.TILE_LAYOUT,
            mesh_device,
            out_memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )
        if probe & 512 and "dbg" not in state:
            state["dbg"] = ttnn.allocate_tensor_on_device(
                ttnn.Shape([GX * GY, 512]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
            )
        dbg = state.get("dbg", y)
        wl = list(w) if isinstance(w, (list, tuple)) else [w]
        k = tuple(int(t.buffer_address()) for t in [x, y, dbg] + wl + ([res] if res is not None else []))
        if state.get("key") != k:
            state["key"], state["desc"] = k, build(x, wl, y, dbg, res)
        extra = ([dbg] if dbg is not y else []) + ([res] if res is not None else [])
        ttnn.generic_op([x] + wl + extra + [y], state["desc"])  # (the output last)
        run.dbg = state.get("dbg")
        return y

    return run, layout


def glu_weight(wg, wu):
    """Gate / up [K, I] each -> [K, 2 I] with tile columns gate 0, up 0, gate 1, up 1, .. (lin2d_prepare(glu=True))."""
    K, I = wg.shape
    return torch.stack([wg.reshape(K, I // 32, 32), wu.reshape(K, I // 32, 32)], dim=2).reshape(K, 2 * I)


def upload_stacked(mesh_device, w, layout, dtype=ttnn.bfloat8_b):
    """W [K, N] torch -> the device weights, one tensor per column group (equal pass widths PNP): stream (column cx, q)'s
    k-th piece (step s = 2 k + q = pass p, K block j: the [kb x PNP] tiles (k major) of its pass-p columns) is region
    o + k of the stream's bank (flat_expert._bank_sharded: region i in bank i % banks at (i // banks); replicated)."""
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import _bank_sharded

    pns, NP, kb, banks, NS = layout["pns"], layout["NP"], layout["kb"], layout["banks"], layout["NS"]
    nblk = NS // NP
    n0s = [sum(pns[:i]) for i in range(len(pns))]
    n_dev = mesh_device.get_num_devices()
    out = []
    for pn, cols in layout["groups"]:
        pnp = pn // NP
        mine = [st_ for st_ in layout["streams"] if st_["cx"] in cols]
        per_bank = {}
        for st_ in mine:
            per_bank[st_["bank"]] = max(per_bank.get(st_["bank"], 0), st_["o"] + st_["n"])
        slots = max(per_bank.values())
        zero = torch.zeros(kb * pnp * 32, 32, dtype=w.dtype)
        regs = [zero] * (slots * banks)
        for st_ in mine:
            for k in range(st_["n"]):
                step = 2 * k + st_["q"]
                p, j = divmod(step, nblk)
                c0 = (n0s[st_["cx"]] + p * pnp) * 32
                blk = w[j * kb * 32 : (j + 1) * kb * 32, c0 : c0 + pnp * 32]
                regs[(st_["o"] + k) * banks + st_["bank"]] = (
                    blk.reshape(kb, 32, pnp, 32).permute(0, 2, 1, 3).reshape(-1, 32)
                )
        out.append(_bank_sharded([regs] * n_dev, banks, dtype, mesh_device))
    return out
