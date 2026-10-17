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
  W        step i = p NBLK + j (pass p, K block j of KB tiles): the KB x PNP tiles of column cx's pass-p columns, one
           contiguous DRAM region (a bank-sharded tensor per group of equal-width columns, rotated over the banks
           by column); sent down the column by its two ends (top NOC0 even steps, bottom NOC1 odd: WS = 2) into the
           W ring (CB 1, NB slots), NSTW pieces read ahead; credits per ring slot (receivers -> the slot's sender)
  RISCs    BRISC (NOC0): DRAM reads, the NOC0 sends, the receive loop (valid counters -> CB pushes, credits);
           NCRISC (NOC1): the NOC1 sends, then the y writes (bf16 TILE, interleaved DRAM or L1)
K3 (bh-lb-09): latent down 7168 -> 3584 103 us, up 3584 -> 7168 95 us standalone (ttnn mm2d 117 / 122); bound by the
DRAM reads (x then W at ~340 GB/s; reads-only probe 95 us). PROBE bits (L2D_PROBE) skip pieces for diagnosis; 512
records a per-core event trace (zz_scratch_k3_dense_mm.py::_lin2d_trace).
"""

import os

import torch

import ttnn

# semaphores: x valid per sender XV0 / XV1, W valid WV0 / WV1 (counters), the feed handshake LANDED / SENT (x, W), XD
# (cores of the column with all of x, on the W senders), W
# credits per ring slot (on the slot's sender: even slots the top core, odd the bottom)
_COMMON = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t KT = get_compile_time_arg_val(0), KB = get_compile_time_arg_val(1), PNPX = get_compile_time_arg_val(2);
constexpr uint32_t NBLK = get_compile_time_arg_val(3), NP = get_compile_time_arg_val(4), NB = get_compile_time_arg_val(5);
constexpr uint32_t GX = get_compile_time_arg_val(6), GY = get_compile_time_arg_val(7), WTB = get_compile_time_arg_val(8);
constexpr uint32_t PNP = get_compile_time_arg_val(9), NT = get_compile_time_arg_val(10);
constexpr uint32_t PROBE = get_compile_time_arg_val(11);  // (1: no DRAM reads, 4: 64 B multicasts)
constexpr uint32_t NBANK = get_compile_time_arg_val(12), NST = get_compile_time_arg_val(13);
// senders per row (XS) / column (WS): 2 = both ends, alternating (x: left NOC0 even, right NOC1 odd; W: top NOC0 even,
// bottom NOC1 odd); 1 = one (x: the right core on NOC1, W: the top core on NOC0: no NoC carries crossing multicasts)
constexpr uint32_t XS = get_compile_time_arg_val(14), WS = get_compile_time_arg_val(15);
// XBAR: the W multicasts start once every core of the column has all of x (XD: count of them, on the W senders), so
// the x row multicasts never cross W column multicasts on a NoC
constexpr uint32_t XBAR = get_compile_time_arg_val(16), NSTW = get_compile_time_arg_val(17);  // (NST: x staging slots)
// WPRE = 0: the W senders read nothing before every core of the column has x (the x reads get the DRAM to themselves)
constexpr uint32_t WPRE = get_compile_time_arg_val(18);
// GLU: the weight's tile columns are gate / up pairs; the output (NT tiles a row) is the activation, PNP / 2 a pass
constexpr uint32_t GLU = get_compile_time_arg_val(19), PO = GLU ? PNP / 2 : PNP;
// RES: y = r + x @ W (r [M, N] bf16 TILE: the NCRISC reads the core's tiles into CB 5, compute starts DST from them)
constexpr uint32_t RES = get_compile_time_arg_val(20);
constexpr auto x_args = TensorAccessorArgs<21>();
constexpr auto y_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr auto d_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();  // (PROBE 512: event trace)
constexpr auto r_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
constexpr uint32_t XB = KB * 2048, WB = KB * PNPX * WTB, NS = NP * NBLK, WPT = KB * PNPX;
constexpr uint32_t XV = 0, WV = 2, XD = 8, WC = 9;  // (WC: per W ring slot, NB of them)
// multicast value words (16 B apart) in CB 10: [0, 128) the BRISC's, [128, 256) the NCRISC's; the feed handshake
// (local semaphores: zero at every launch) per stream (x 0, W 1): LANDED (BRISC: pieces read into the staging),
// SENT (NCRISC: multicast and flushed)
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
// W piece (column cx of the column group, step s): one contiguous region, exactly KB x PNP tiles (PNPX = PNP: one W
// tensor per group of equal-width columns); g = cx NSP + s in bank (g + cx) % NBANK at (g / NBANK) WB (rotated by the
// column: the columns' senders read different banks at the same step)
constexpr uint32_t NSP = (NS + NBANK - 1) / NBANK * NBANK;
FORCE_INLINE uint64_t w_piece(uint32_t w_addr, uint32_t cx, uint32_t step) {
    const uint32_t g = cx * NSP + step;
    return get_noc_addr_from_bank_id<true>((g + cx) % NBANK, w_addr + (g / NBANK) * WB);
}
// sender q's k-th own piece (x block / W step) and own count, S senders (S = 1: q = the only one)
constexpr uint32_t own_at(uint32_t S, uint32_t q, uint32_t k) { return S == 2 ? 2 * k + q : k; }
constexpr uint32_t n_own(uint32_t S, uint32_t n, uint32_t q) { return S == 2 ? (n + 1 - q) / 2 : n; }
// the sender (q) and own index (k) of piece i
constexpr uint32_t q_of(uint32_t S, uint32_t q1, uint32_t i) { return S == 2 ? (i & 1) : q1; }
constexpr uint32_t k_of(uint32_t S, uint32_t i) { return S == 2 ? (i >> 1) : i; }
// W credits per ring slot (a shared counter would let a fast core's credits stand in for a slow one's): step
// s = 2 k + q goes once every column core freed slot s % NB for the (s / NB)-th time
FORCE_INLINE bool w_credit_ok(uint32_t q, uint32_t k) {
    const uint32_t st = own_at(WS, q, k);
    if (XBAR && *sem(XD) < GY) return false;
    return st < NB || (PROBE & 128) || *sem(WC + st % NB) >= GY * (st / NB);
}
"""

_BRISC = (
    _COMMON
    + r"""
// one stream read by this BRISC into a staging ring of NST slots (read transaction id per slot); sent either by
// this BRISC (send) or by the NCRISC (feed: landed / sent exchanged through L1 words)
struct Stream {
    bool on = false, send = false, isw = false;
    uint32_t nst = NST, q = 0, n = 0, issued = 0, landed = 0, sent = 0, stage = 0, piece = 0, trid0 = 0, hs = 0;
};

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0), w_addr = get_arg_val<uint32_t>(1);
    const uint32_t cx = get_arg_val<uint32_t>(2), cy = get_arg_val<uint32_t>(3);
    const uint32_t lx = get_arg_val<uint32_t>(4), rx = get_arg_val<uint32_t>(5), ry = get_arg_val<uint32_t>(6);  // row
    const uint32_t ty = get_arg_val<uint32_t>(7), by = get_arg_val<uint32_t>(8), cxv = get_arg_val<uint32_t>(9);  // col
    const auto xg = TensorAccessor(x_args, x_addr, 2048);
    const uint32_t xres = get_write_ptr(0), wring = get_write_ptr(1), words = get_write_ptr(10);
    // the left core sends the even x blocks (NOC0), the right core feeds the odd ones to its NCRISC (NOC1 multicast);
    // the top core sends the even W steps, the bottom core feeds the odd ones
    Stream S[2];
    Stream& X = S[0];
    Stream& W = S[1];
    X.on = (XS == 2 && cx == 0) || cx == GX - 1, X.send = XS == 2 && cx == 0, X.q = X.send ? 0 : 1;
    X.n = n_own(XS, NBLK, X.q);
    X.stage = get_write_ptr(8), X.piece = XB, X.trid0 = 1, X.hs = 0;
    W.on = cy == 0 || (WS == 2 && cy == GY - 1), W.send = cy == 0, W.q = cy == 0 ? 0 : 1, W.n = n_own(WS, NS, W.q);
    W.isw = true;
    W.stage = get_write_ptr(9), W.piece = WB, W.trid0 = 1 + NST, W.hs = 16, W.nst = NSTW;
    const uint32_t cg = get_arg_val<uint32_t>(11);  // (the column's index in its group: the W tensor's piece order)
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
            while (T.issued < T.n && T.issued < T.sent + T.nst && (WPRE || !T.isw || *sem(XD) >= GY)) {
                const uint32_t k = T.issued, slot = k % T.nst, dst = T.stage + slot * T.piece;
                if (T.send && !(PROBE & 256)) noc_async_writes_flushed();  // (the slot's earlier multicast has left)
                noc_async_read_set_trid(T.trid0 + slot);
                if (!(PROBE & 1)) {
                    if (T.isw) {
                        if (!(PROBE & 16)) noc_async_read(w_piece(w_addr, cg, own_at(WS, T.q, k)), dst, WB);
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
                    const uint32_t src = T.stage + slot * T.piece, bytes = (PROBE & (T.isw ? 36 : 68)) ? 64 : T.piece;
                    const uint32_t vw = words + (T.hs * 4) + (k & 3) * 16;  // (x: words 0 .., W: 64 ..)
                    *word(vw) = k + 1;
                    if (T.isw) {
                        noc_async_write_multicast_loopback_src(src, get_noc_multicast_addr(cxv, ty, cxv, by, wring + (own_at(WS, 0, k) % NB) * WB), bytes, GY);
                        noc_semaphore_set_multicast_loopback_src(vw, get_noc_multicast_addr(cxv, ty, cxv, by, get_semaphore(WV)), GY);
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
                noc_semaphore_inc(get_noc_addr(cxv, ty, get_semaphore(XD)), 1);
                if (WS == 2) noc_semaphore_inc(get_noc_addr(cxv, by, get_semaphore(XD)), 1);
            }
        }
        if (r < NS && *sem(WV + q_of(WS, 0, r)) >= k_of(WS, r) + 1) {
            cb_reserve_back(1, WPT);
            cb_push_back(1, WPT);
            tr(r);
            ++r;
        }
        // credit step c (>= NB) to its sender once the slot of step c - NB is free here
        if (c < NS && c < r + NB && cb_pages_reservable_at_back(1, WPT * (c - r + 1))) {
            noc_semaphore_inc(get_noc_addr(cxv, q_of(WS, 0, c) ? by : ty, get_semaphore(WC + c % NB)), 1);
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
    // the odd x blocks (right core) / odd W steps (bottom core): the BRISC reads them into the staging
    const bool xo = cx == GX - 1, wo = WS == 2 && cy == GY - 1;
    const uint32_t nx = xo ? n_own(XS, NBLK, 1) : 0, nw = wo ? n_own(WS, NS, 1) : 0;
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
            noc_async_write_multicast_loopback_src(wst + (ws % NSTW) * WB, get_noc_multicast_addr(cxv, by, cxv, ty, wring + (own_at(WS, 1, ws) % NB) * WB),
                                                   (PROBE & 36) ? 64 : WB, GY);
            noc_semaphore_set_multicast_loopback_src(vw, get_noc_multicast_addr(cxv, by, cxv, ty, get_semaphore(WV + 1)), GY);
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
    assert nst + nstw <= 15  # (a read transaction id per staging slot)
    xb = kb * 2048
    allc = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
    vv = {
        (cx, cy): mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(cx, cy))
        for cx in range(GX)
        for cy in range(GY)
    }
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
    assert xs in (1, 2) and ws in (1, 2) and nb <= 7
    probe = int(os.environ.get("L2D_PROBE", "0"))  # (1: no DRAM reads, 2: no matmuls, 4: 64 B multicasts)
    state = {}
    nbank = mesh_device.dram_grid_size().x
    layout = dict(pns=pns, NP=NP, kb=kb, banks=nbank, xs=xs, ws=ws, groups=sorted(groups.items()))

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
        ] + (
            [ttnn.CBDescriptor(total_size=2048, core_ranges=allc, format_descriptors=fmt(11, ttnn.uint32, 2048))]
            if probe & 512
            else []
        )
        for pn, cols in sorted(groups.items()):
            crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, 0), ttnn.CoreCoord(cx, GY - 1)) for cx in cols])
            wbg = kb * (pn // NP) * wtb
            program.cbs += (
                [
                    ttnn.CBDescriptor(total_size=nb * wbg, core_ranges=crs, format_descriptors=fmt(1, w_dtype, wtb)),
                    ttnn.CBDescriptor(
                        total_size=NP * (pn // NP // G) * 2048,
                        core_ranges=crs,
                        format_descriptors=fmt(4, out_dtype, 2048),
                    ),
                ]
                + (
                    [
                        ttnn.CBDescriptor(
                            total_size=pn * 2048, core_ranges=crs, format_descriptors=fmt(5, ttnn.bfloat16, 2048)
                        )
                    ]
                    if residual
                    else []
                )
                + [
                    ttnn.CBDescriptor(total_size=nstw * wbg, core_ranges=crs, format_descriptors=fmt(9, w_dtype, wtb)),
                ]
            )
        program.semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=allc, initial_value=0) for i in range(9 + nb)]
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
                + [xs, ws, xbar, nstw, wpre, int(glu), int(residual)]
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
                    b_rt[cx][cy] = [
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
                    n_rt[cx][cy] = [
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
    """W [K, N] torch -> the device weights, one tensor per column group (equal pass widths PNP): column cg of the
    group, step s = (pass p, K block j): the [kb x PNP] tiles (k major) of its pass-p columns, one contiguous region,
    g = cg NSP + s in bank (g + cg) % banks at (g // banks) (flat_expert._bank_sharded; replicated over the mesh)."""
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import _bank_sharded

    pns, NP, kb, banks = layout["pns"], layout["NP"], layout["kb"], layout["banks"]
    K = w.shape[0]
    nblk = K // 32 // kb
    NS = NP * nblk
    NSP = -(-NS // banks) * banks
    n0s = [sum(pns[:i]) for i in range(len(pns))]
    n_dev = mesh_device.get_num_devices()
    out = []
    for pn, cols in layout["groups"]:
        pnp = pn // NP
        regs = [None] * (len(cols) * NSP)
        zero = torch.zeros(kb * pnp * 32, 32, dtype=w.dtype)
        for cg, cx in enumerate(cols):
            for p in range(NP):
                c0 = (n0s[cx] + p * pnp) * 32
                for j in range(nblk):
                    blk = w[j * kb * 32 : (j + 1) * kb * 32, c0 : c0 + pnp * 32]
                    g = cg * NSP + p * nblk + j
                    regs[(g // banks) * banks + (g + cg) % banks] = (
                        blk.reshape(kb, 32, pnp, 32).permute(0, 2, 1, 3).reshape(-1, 32)
                    )
        out.append(_bank_sharded([[r if r is not None else zero for r in regs]] * n_dev, banks, dtype, mesh_device))
    return out
