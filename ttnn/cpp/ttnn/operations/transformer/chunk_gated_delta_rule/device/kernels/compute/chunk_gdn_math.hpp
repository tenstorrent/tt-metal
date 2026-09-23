// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Shared math bodies for the chunk_gdn prep and scan compute kernels (and any future fused
// kernel). The kernel .cpp files are thin: compile-arg reads, constexpr CB maps, and loops
// calling prep_chunk / scan_step below. CB ids are passed in as plain uint32_t (via the
// GdnPrepCbs / GdnScanCbs structs for the composition functions), so the same bodies run under
// different CB maps.
//
// THE BIT-EXACTNESS CONTRACT: the seven prep intermediates are rounded at *pack* time; the DRAM
// round trip after that is a byte copy. A kernel composed from these bodies is bit-identical to
// the phased path end-to-end iff it
//   (1) reuses these math code bodies unchanged (no op reordering, loop merging, or moving of
//       init call sites),
//   (2) keeps the same pack-to-CB boundaries at the same CB data formats,
//   (3) keeps HiFi4 / fp32-dest-acc / no-approx-mode and the same matmul ki-accumulation order,
//   (4) keeps the same exp configuration.
// Any deliberate violation (e.g. keeping T_inv in DEST unpacked) is a reviewed PCC-downgrade,
// not a silent change.

#pragma once

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

// GDN_HOIST_RECONFIG (a per-kernel define, set by the fused factory on its producer compute
// kernel only): hoist the packer/unpacker format reconfigs out of the WY hot path (invert16 /
// invert_block — all-fp32 regions where the per-call reconfigs are redundant register writes).
// Math ops, order, and pack boundaries are identical either way (bit-exact both settings; the
// phased path measured byte-identical outputs WITH the hoist). It is a per-path PERF switch:
// the hoist measured -15% on the fused producer's chunk rate but +22-37% on phased prep at
// low items-per-core shapes (BH=12/T=512) — a timing sensitivity, not a correctness issue —
// so only the fused producer opts in.
#ifdef GDN_HOIST_RECONFIG
inline constexpr bool kGdnHoistReconfig = true;
#else
inline constexpr bool kGdnHoistReconfig = false;
#endif

inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }

// out[Mt,Nt] = A[Mt,Kt] @ (tr ? B[Nt,Kt]^T : B[Kt,Nt]). Inputs must be available.
inline void mm(
    uint32_t a, uint32_t b, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t Nt, bool tr, bool skip_reconfig = false) {
    cb_reserve_back(o, Mt * Nt);
    if (!skip_reconfig) {
        pack_reconfig_data_format(o);  // mixed bf16/fp32 CBs: set packer to this output's format
        // matmul_tiles(a,b): in0=a->srcB, in1=b->srcA. Reconfig unpack src formats to match (the
        // op init only asserts formats, it does not set them), else fp32/bf16 CBs are read at the
        // wrong format and produce garbage. skip_reconfig=true is legal ONLY when the caller has
        // already configured the packer/unpackers for these operands' formats (all-fp32 regions).
        reconfig_data_format(b, a);
    }
    matmul_init(a, b, tr ? 1 : 0);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            for (uint32_t ki = 0; ki < Kt; ki++) {
                uint32_t bi = tr ? (ni * Kt + ki) : (ki * Nt + ni);
                matmul_tiles(a, b, mi * Kt + ki, bi, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_push_back(o, Mt * Nt);
}

// out = A (op) B elementwise, n tiles. op: 0 add, 1 sub, 2 mul.
inline void ew(uint32_t a, uint32_t b, uint32_t o, uint32_t n, int op) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);  // binary(a,b): a->srcA, b->srcB
    if (op == 0) {
        add_init(a, b);
    } else if (op == 1) {
        sub_init(a, b);
    } else {
        mul_init(a, b);
    }
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        if (op == 0) {
            add_tiles(a, b, i, i, 0);
        } else if (op == 1) {
            sub_tiles(a, b, i, i, 0);
        } else {
            mul_tiles(a, b, i, i, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

inline void expc(uint32_t in, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);  // unary: in->srcA
    copy_tile_to_dst_init_short(in);
    exp_tile_init();
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in, i, 0);
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out[Mt,Nt] = A[Mt,Nt] * col[Mt,1]  (broadcast the single column of `col` across N)
inline void bcast_cols_mul(uint32_t a, uint32_t col, uint32_t o, uint32_t Mt, uint32_t Nt) {
    cb_reserve_back(o, Mt * Nt);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, col);  // bcast(a,col): a->srcA, col->srcB
    mul_bcast_cols_init(a, col);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(a, col, mi * Nt + ni, mi, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_push_back(o, Mt * Nt);
}

// out[Mt,Nt] = A[Mt,Nt] - row[1,Nt]  (broadcast the single row of `row` across M)
inline void bcast_rows_sub(uint32_t a, uint32_t row, uint32_t o, uint32_t Mt, uint32_t Nt) {
    cb_reserve_back(o, Mt * Nt);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, row);  // bcast(a,row): a->srcA, row->srcB
    sub_bcast_rows_init(a, row);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            sub_tiles_bcast_rows(a, row, mi * Nt + ni, ni, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_push_back(o, Mt * Nt);
}

// out = A * scalar, n tiles. scalar is the [0,0] element of the single `scal` tile.
inline void bcast_scalar_mul(uint32_t a, uint32_t scal, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, scal);  // bcast(a,scal): a->srcA, scal->srcB
    mul_bcast_scalar_init(a, scal);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(a, scal, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out[0] = copy of src[src_tile] (single 32x32 tile). src must be available.
inline void cpy_t(uint32_t src, uint32_t src_tile, uint32_t o, bool skip_reconfig = false) {
    cb_reserve_back(o, 1);
    if (!skip_reconfig) {  // see mm() for the skip_reconfig contract
        pack_reconfig_data_format(o);
        reconfig_data_format_srca(src);
    }
    copy_tile_to_dst_init_short(src);
    tile_regs_acquire();
    copy_tile(src, src_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

// out[0] = a[ai] (op) b[bi], single tile. op: 0 add, 2 mul. (Like ew but with free tile indices.)
inline void ewt(uint32_t a, uint32_t ai, uint32_t b, uint32_t bi, uint32_t o, int op, bool skip_reconfig = false) {
    cb_reserve_back(o, 1);
    if (!skip_reconfig) {  // see mm() for the skip_reconfig contract
        pack_reconfig_data_format(o);
        reconfig_data_format(a, b);
    }
    if (op == 0) {
        add_init(a, b);
    } else {
        mul_init(a, b);
    }
    tile_regs_acquire();
    if (op == 0) {
        add_tiles(a, b, ai, bi, 0);
    } else {
        mul_tiles(a, b, ai, bi, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

// (I32 - Nq)^-1 for a strictly-lower 16-block Nq isolated in one 16-quadrant (rest zero),
// nilpotent at 16. Horner in 15 terms -> out (single tile); the other diagonal quadrant is I.
// Small block + short chain keeps fp32 bounded where a 32x32/31-term Horner cancels.
// cb_eye holds the identity (tile 0 = I32).
inline void invert16(uint32_t nq, uint32_t out, uint32_t tmp, uint32_t cb_eye) {
    // Hot path: 15 alternating single-tile matmul/add rounds per call, ~4 calls per chunk. All
    // four CBs are fp32, so the unpacker/packer format registers never change across the loop —
    // reconfigure ONCE up front instead of inside every mm()/ew() call (their per-call
    // reconfig_data_format/pack_reconfig are unconditional register writes). The MOP inits still
    // alternate per op class. Ops, operands, order, and pack boundaries are identical to the
    // plain mm/ew composition this replaces — bit-exact with it by construction.
    if (kGdnHoistReconfig) {
        pack_reconfig_data_format(out);    // out and tmp are both fp32: one packer config serves all
        reconfig_data_format(cb_eye, nq);  // all operands fp32: one unpack config serves mm and ew
    }
    auto add1 = [&](uint32_t a, uint32_t b, uint32_t o) {  // o = a + b, 1 tile
        cb_reserve_back(o, 1);
        if (!kGdnHoistReconfig) {  // per-call reconfigs, exactly as the plain ew() would issue
            pack_reconfig_data_format(o);
            reconfig_data_format(a, b);
        }
        add_init(a, b);
        tile_regs_acquire();
        add_tiles(a, b, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, 0);
        tile_regs_release();
        cb_push_back(o, 1);
    };
    add1(cb_eye, nq, out);  // out = I + Nq
    CircularBuffer(out).wait_front(1);
    for (uint32_t m = 2; m < 16; m++) {  // sum_{k<16} Nq^k
        cb_reserve_back(tmp, 1);
        if (!kGdnHoistReconfig) {  // per-call reconfigs, exactly as the plain mm() would issue
            pack_reconfig_data_format(tmp);
            reconfig_data_format(out, nq);
        }
        matmul_init(nq, out, 0);
        tile_regs_acquire();
        matmul_tiles(nq, out, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tmp, 0);
        tile_regs_release();
        cb_push_back(tmp, 1);
        CircularBuffer(tmp).wait_front(1);
        CircularBuffer(out).pop_front(1);
        add1(cb_eye, tmp, out);  // out = I + Nq @ out
        CircularBuffer(out).wait_front(1);
        CircularBuffer(tmp).pop_front(1);
    }
}

// Assemble the 2x2 tile-block matrix [[s0[t0], s1[t1]], [s2[t2], s3[t3]]] into o (4 tiles).
inline void asm4(
    uint32_t s0,
    uint32_t t0,
    uint32_t s1,
    uint32_t t1,
    uint32_t s2,
    uint32_t t2,
    uint32_t s3,
    uint32_t t3,
    uint32_t o) {
    const uint32_t src[4] = {s0, s1, s2, s3};
    const uint32_t tl[4] = {t0, t1, t2, t3};
    cb_reserve_back(o, 4);
    pack_reconfig_data_format(o);
    for (uint32_t i = 0; i < 4; i++) {
        reconfig_data_format_srca(src[i]);
        copy_tile_to_dst_init_short(src[i]);
        tile_regs_acquire();
        copy_tile(src[i], tl[i], 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, 4);
}

// Invert one 32x32 diagonal tile-block: out[0] = (I32 - negN)^-1, negN = src[tile] (strictly-lower
// 32x32). Mirrors FLA solve_tril: split into 16-quadrants negN = [[N00,0],[N10,N11]], invert the two
// diagonal 16-blocks (short, bounded Horners), and form the off-diagonal EXACTLY (one matmul chain,
// no power series). A single 32x32 Horner instead loses fp32 precision on harder blocks.
//   Bi00=(I-N00)^-1 (top-left), Bi11=(I-N11)^-1 (bottom-right), off=Bi11@N10@Bi00 (bottom-left).
//   out = [[Bi00,0],[off,Bi11]].
// cb_eye = identity; masks (single tiles): cb_mask[0]=Qtl, [1]=Qbr, [2]=Q10 (bottom-left).
// tmpN/tmpT = scratch. Private scratch A..D: single-tile-capable fp32 CBs, NOT drained by any
// writer while this runs, and none may alias src, out, tmpN, or tmpT.
inline void invert_block(
    uint32_t src,
    uint32_t tile,
    uint32_t out,
    uint32_t tmpN,
    uint32_t tmpT,
    uint32_t cb_eye,
    uint32_t cb_mask,
    uint32_t A,
    uint32_t B,
    uint32_t C,
    uint32_t D) {
    // Every CB this function touches is fp32, so one packer+unpacker format config up front
    // serves the whole body; the per-call reconfigs inside cpy_t/ewt/mm are skipped (they are
    // unconditional register writes and this body issues ~9 such calls per invocation, twice per
    // chunk). Op order and pack boundaries are unchanged — bit-exact with the unhoisted form.
    if (kGdnHoistReconfig) {
        pack_reconfig_data_format(out);
        reconfig_data_format(cb_eye, src);
    }
    cpy_t(src, tile, tmpN, kGdnHoistReconfig);
    CircularBuffer(tmpN).wait_front(1);  // negN -> tmpN[0]
    // Bi00 = (I-N00)^-1  (N00 = top-left quadrant of negN; top-right is already 0)
    ewt(tmpN, 0, cb_mask, 0, A, 2, kGdnHoistReconfig);
    CircularBuffer(A).wait_front(1);  // N00
    invert16(A, B, tmpT, cb_eye);
    CircularBuffer(B).wait_front(1);
    CircularBuffer(A).pop_front(1);  // Bi00 -> B
    // Bi11 = (I-N11)^-1  (N11 = bottom-right quadrant)
    ewt(tmpN, 0, cb_mask, 1, A, 2, kGdnHoistReconfig);
    CircularBuffer(A).wait_front(1);  // N11
    invert16(A, C, tmpT, cb_eye);
    CircularBuffer(C).wait_front(1);
    CircularBuffer(A).pop_front(1);  // Bi11 -> C
    // off = Bi11 @ N10 @ Bi00  (N10 = bottom-left quadrant; result lives only there)
    ewt(tmpN, 0, cb_mask, 2, A, 2, kGdnHoistReconfig);
    CircularBuffer(A).wait_front(1);  // N10
    CircularBuffer(tmpN).pop_front(1);
    mm(C, A, tmpT, 1, 1, 1, false, kGdnHoistReconfig);
    CircularBuffer(tmpT).wait_front(1);
    CircularBuffer(A).pop_front(1);  // Bi11@N10
    mm(tmpT, B, A, 1, 1, 1, false, kGdnHoistReconfig);
    CircularBuffer(A).wait_front(1);
    CircularBuffer(tmpT).pop_front(1);  // @Bi00 -> A(off)
    // out = Qtl*Bi00 + Qbr*Bi11 + off
    ewt(B, 0, cb_mask, 0, D, 2, kGdnHoistReconfig);
    CircularBuffer(D).wait_front(1);
    CircularBuffer(B).pop_front(1);  // Bi00_tl -> D
    ewt(C, 0, cb_mask, 1, B, 2, kGdnHoistReconfig);
    CircularBuffer(B).wait_front(1);
    CircularBuffer(C).pop_front(1);  // Bi11_br -> B
    ewt(D, 0, B, 0, C, 0, true);
    CircularBuffer(C).wait_front(1);
    CircularBuffer(D).pop_front(1);
    CircularBuffer(B).pop_front(1);
    ewt(C, 0, A, 0, out, 0, true);
    CircularBuffer(C).pop_front(1);
    CircularBuffer(A).pop_front(1);  // + off -> out
}

// out[1,Ct] row-form = transpose of col[Ct,1]; produces Ct tiles (each row0 = a 32-chunk of col).
inline void transpose_col(uint32_t in, uint32_t o, uint32_t Ct) {
    cb_reserve_back(o, Ct);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);  // unary: in->srcA
    transpose_init(in);
    for (uint32_t i = 0; i < Ct; i++) {
        tile_regs_acquire();
        transpose_tile(in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, Ct);
}

// OPT-A/B in-kernel L2-norm over K. rowsum_k: o[Mt,1(broadcast)] = sum over the full K dim of
// in[Mt,Kt], computed as in @ ones by reusing cb_ones tile 0 as the [K,1] contraction operand
// (avoids a dedicated ones-column constant). Mirrors the `mm` helper's reconfig/matmul discipline.
inline void rowsum_k(uint32_t in, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t cb_ones) {
    cb_reserve_back(o, Mt);
    pack_reconfig_data_format(o);
    reconfig_data_format(cb_ones, in);  // matmul(in, cb_ones): in->srcB, cb_ones->srcA
    matmul_init(in, cb_ones, 0);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        tile_regs_acquire();
        for (uint32_t ki = 0; ki < Kt; ki++) {
            matmul_tiles(in, cb_ones, mi * Kt + ki, 0, 0);  // reuse ones tile 0 for every ki
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, mi);
        tile_regs_release();
    }
    cb_push_back(o, Mt);
}

// inv_rms: o[i] = rsqrt(in[i] + eps) [* scale]. in holds per-row sum-of-squares (rowsum_k output);
// out is the per-row inverse-L2 factor (optionally pre-scaled, for folding q's scale into the norm).
// eps/scale arrive as fp32-bit-cast uint32 compile args.
inline void inv_rms(uint32_t in, uint32_t o, uint32_t n, uint32_t eps_bits, uint32_t scale_bits, bool do_scale) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in, i, 0);
        binop_with_scalar_tile_init();
        add_unary_tile(0, eps_bits);  // + eps
        rsqrt_tile_init();
        rsqrt_tile(0);  // 1/sqrt(sumsq + eps)
        if (do_scale) {
            binop_with_scalar_tile_init();
            mul_unary_tile(0, scale_bits);  // * scale (q only)
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// CB map for prep_chunk — one field per CB the body touches. dl/mask are the prep kernel's
// aliases (cb_dl = the vnew slot, cb_mask = the u slot); the map carries the resolved ids.
struct GdnPrepCbs {
    uint32_t q, k, v, g, beta;
    uint32_t eye, tril, ones, S;
    uint32_t decay, decay_exp, decayfac, lmask, Tinv, vbeta, kbeta;
    uint32_t w, qdecay, intra, s2, ointer, kdec_t, supd, stmp, final_s;
    uint32_t scr1, scr2, scr3, s3;
    uint32_t dl;    // alias of the vnew slot in prep (1 tile used)
    uint32_t mask;  // alias of the u slot in prep (3 quadrant-mask tiles)
};

// CB map for scan_step — one field per CB the body touches (the state CBs S/s2/s3/final are
// selected per chunk by the caller and passed as cur_S/dst).
struct GdnScanCbs {
    uint32_t dl, Tinv, out;
    uint32_t vbeta, kd, qdecay, intra;
    uint32_t vnew, ointer, kdec_t, supd, stmp;
    uint32_t scr1;
};

// PHASE A (prep): one state-independent (head, chunk) work-item. No recurrent state here; the
// sequential state scan lives in scan_step. Outputs (per chunk) v_beta, kd(->cb.w), T_inv,
// k_dec_t, q_decay, intra, dl are pushed to their CBs and streamed to DRAM by the prep writer.
inline void prep_chunk(
    const GdnPrepCbs& cb, uint32_t Ct, uint32_t Kt, uint32_t Vt, bool qk_norm, uint32_t scale_bits, uint32_t eps_bits) {
    const uint32_t cc = Ct * Ct;
    const uint32_t ck = Ct * Kt;
    const uint32_t cv = Ct * Vt;
    const uint32_t C = Ct * 32;

    WAIT(cb.q, ck);
    WAIT(cb.k, ck);
    WAIT(cb.v, cv);
    WAIT(cb.g, Ct);
    WAIT(cb.beta, Ct);

    // ---- OPT-B: in-kernel L2-norm of q,k over K (fold q's scale). Consumes the raw reader q/k
    // and produces normalized q->cb.supd, k->cb.stmp (both free in Ct==1). The rest of the chunk
    // then reads Q/Kk instead of cb.q/cb.k. scr1/scr2/scr3 are free here (used only later). ----
    uint32_t Q = cb.q, Kk = cb.k;
    if (qk_norm) {
        // q: q^2 -> rowsum_K -> rsqrt(+eps)*scale -> q_normed (cb.supd)
        ew(cb.q, cb.q, cb.scr1, ck, 2);
        WAIT(cb.scr1, ck);
        rowsum_k(cb.scr1, cb.scr2, Ct, Kt, cb.ones);
        WAIT(cb.scr2, Ct);
        POP(cb.scr1, ck);
        inv_rms(cb.scr2, cb.scr3, Ct, eps_bits, scale_bits, /*do_scale=*/true);
        WAIT(cb.scr3, Ct);
        POP(cb.scr2, Ct);
        bcast_cols_mul(cb.q, cb.scr3, cb.supd, Ct, Kt);
        WAIT(cb.supd, ck);
        POP(cb.scr3, Ct);
        POP(cb.q, ck);
        // k: same, no scale -> k_normed (cb.stmp)
        ew(cb.k, cb.k, cb.scr1, ck, 2);
        WAIT(cb.scr1, ck);
        rowsum_k(cb.scr1, cb.scr2, Ct, Kt, cb.ones);
        WAIT(cb.scr2, Ct);
        POP(cb.scr1, ck);
        inv_rms(cb.scr2, cb.scr3, Ct, eps_bits, scale_bits, /*do_scale=*/false);
        WAIT(cb.scr3, Ct);
        POP(cb.scr2, Ct);
        bcast_cols_mul(cb.k, cb.scr3, cb.stmp, Ct, Kt);
        WAIT(cb.stmp, ck);
        POP(cb.scr3, Ct);
        POP(cb.k, ck);
        Q = cb.supd;
        Kk = cb.stmp;
    }

    // ---- P1: v_beta, k_beta ----
    bcast_cols_mul(cb.v, cb.beta, cb.vbeta, Ct, Vt);
    WAIT(cb.vbeta, cv);
    bcast_cols_mul(Kk, cb.beta, cb.kbeta, Ct, Kt);
    WAIT(cb.kbeta, ck);
    POP(cb.beta, Ct);
    POP(cb.v, cv);

    // ---- P2: decay = tril@g, decay_exp, decay_row ----
    mm(cb.tril, cb.g, cb.decay, Ct, Ct, 1, false);
    WAIT(cb.decay, Ct);
    expc(cb.decay, cb.decay_exp, Ct);
    WAIT(cb.decay_exp, Ct);
    transpose_col(cb.decay, cb.scr1, Ct);  // decay_row in scr1
    WAIT(cb.scr1, Ct);

    // ---- L_mask = tril(exp(decay_i - decay_j)) ----
    bcast_cols_mul(cb.ones, cb.decay, cb.scr2, Ct, Ct);  // decay_i everywhere
    WAIT(cb.scr2, cc);
    bcast_rows_sub(cb.scr2, cb.scr1, cb.scr3, Ct, Ct);  // decay_i - decay_j
    WAIT(cb.scr3, cc);
    POP(cb.scr1, Ct);  // decay_row done
    POP(cb.scr2, cc);
    ew(cb.scr3, cb.tril, cb.scr2, cc, 2);  // *tril (zero upper)
    WAIT(cb.scr2, cc);
    POP(cb.scr3, cc);
    expc(cb.scr2, cb.scr3, cc);  // exp
    WAIT(cb.scr3, cc);
    POP(cb.scr2, cc);
    ew(cb.scr3, cb.tril, cb.lmask, cc, 2);  // *tril again -> L_mask
    WAIT(cb.lmask, cc);
    POP(cb.scr3, cc);

    // ---- decayfac = exp(g_sum - decay) ----
    // (dl = exp(g_sum) is recomputed at the scan from decayfac[0]*decay_exp[0] so its CB
    //  slot can be reused as the third ping-pong state buffer cb_s3.)
    mm(cb.ones, cb.g, cb.scr1, Ct, Ct, 1, false);  // g_sum in every row (col form)
    WAIT(cb.scr1, Ct);
    POP(cb.g, Ct);
    ew(cb.scr1, cb.decay, cb.scr2, Ct, 1);  // g_sum - decay
    WAIT(cb.scr2, Ct);
    POP(cb.scr1, Ct);
    POP(cb.decay, Ct);
    expc(cb.scr2, cb.decayfac, Ct);
    WAIT(cb.decayfac, Ct);
    POP(cb.scr2, Ct);

    // ---- N = strictly_lower(k_beta@k^T * L_mask); T_inv = (I + strictly_lower)^-1 ----
    // The WY inverse, mirroring FLA's solve_tril: block down to 16x16 (invert_block splits each
    // 32x32 tile into 16-quadrants), invert the small diagonal blocks with bounded Horners, and
    // merge off-diagonal blocks EXACTLY. This keeps every intermediate bounded, unlike a single
    // 32x32/full-matrix Horner whose deep power series loses fp32 precision on harder chunks.
    mm(cb.kbeta, Kk, cb.scr1, Ct, Kt, Ct, true);  // kk = k_beta @ k^T (Kk = normalized k)
    WAIT(cb.scr1, cc);
    ew(cb.scr1, cb.lmask, cb.scr2, cc, 2);  // kk_masked = kk * L_mask
    WAIT(cb.scr2, cc);
    POP(cb.scr1, cc);
    ew(cb.scr2, cb.eye, cb.scr1, cc, 2);  // diag(kk_masked)
    WAIT(cb.scr1, cc);
    // negN = diag - kk_masked = -(strictly_lower(kk_masked))  (= -A_strict, kept in cb.scr3)
    ew(cb.scr1, cb.scr2, cb.scr3, cc, 1);
    WAIT(cb.scr3, cc);
    POP(cb.scr1, cc);
    POP(cb.scr2, cc);

    // invert_block's private scratch A..D = cb.S/cb.final_s/cb.s2/cb.s3 — all fp32 and NOT drained
    // by the prep writer (unlike the output CBs cb.w/cb.qdecay/cb.intra, whose scratch pushes the
    // writer would wrongly consume). None alias src (cb.scr3), out, or the Ct==2 persistents
    // (cb.supd/cb.stmp).
    if (Ct == 1) {
        // Single 32x32 block: T_inv is just its inverse.
        invert_block(cb.scr3, 0, cb.Tinv, cb.scr1, cb.scr2, cb.eye, cb.mask, cb.S, cb.final_s, cb.s2, cb.s3);
        WAIT(cb.Tinv, cc);
        POP(cb.scr3, cc);
    } else if (Ct == 2) {
        // 2x2 tile-block lower-triangular. negN tiles: 0=(0,0), 2=(1,0), 3=(1,1); (0,1)=0.
        // Diagonal inverses Mi11, Mi22, then off-diagonal Mi21 = -Mi22 @ A21 @ Mi11.
        // (A21 = -negN21, so -Mi22@A21@Mi11 = Mi22 @ negN21 @ Mi11.)
        // Mi11 -> cb.supd, Mi22 -> cb.stmp, Mi21 -> cb.ointer (all free in prep).
        invert_block(cb.scr3, 0, cb.supd, cb.scr1, cb.scr2, cb.eye, cb.mask, cb.S, cb.final_s, cb.s2, cb.s3);  // Mi11
        invert_block(cb.scr3, 3, cb.stmp, cb.scr1, cb.scr2, cb.eye, cb.mask, cb.S, cb.final_s, cb.s2, cb.s3);  // Mi22
        cpy_t(cb.scr3, 2, cb.scr1);  // negN21 -> cb.scr1[0]
        WAIT(cb.scr1, 1);
        mm(cb.scr1, cb.supd, cb.scr2, 1, 1, 1, false);  // tmp = negN21 @ Mi11
        WAIT(cb.scr2, 1);
        POP(cb.scr1, 1);
        mm(cb.stmp, cb.scr2, cb.ointer, 1, 1, 1, false);  // Mi21 = Mi22 @ tmp
        WAIT(cb.ointer, 1);
        POP(cb.scr2, 1);
        POP(cb.scr3, cc);  // negN done
        // T_inv = [[Mi11, 0], [Mi21, Mi22]]  (cb.eye[1] is the zero block)
        asm4(cb.supd, 0, cb.eye, 1, cb.ointer, 0, cb.stmp, 0, cb.Tinv);
        WAIT(cb.Tinv, cc);
        POP(cb.supd, 1);
        POP(cb.stmp, 1);
        POP(cb.ointer, 1);
    } else {
        // Fallback (C>64, currently xfail): full-matrix Horner.
        ew(cb.eye, cb.scr3, cb.Tinv, cc, 0);
        WAIT(cb.Tinv, cc);
        for (uint32_t m = 2; m < C; m++) {
            mm(cb.scr3, cb.Tinv, cb.scr1, Ct, Ct, Ct, false);
            WAIT(cb.scr1, cc);
            POP(cb.Tinv, cc);
            ew(cb.eye, cb.scr1, cb.Tinv, cc, 0);
            WAIT(cb.Tinv, cc);
            POP(cb.scr1, cc);
        }
        POP(cb.scr3, cc);
    }

    // ---- un-premultiplied WY hand-off: output v_beta (cb.vbeta), kd=k_beta*decay_exp (cb.w),
    // T_inv (cb.Tinv). The scan computes v_new = T_inv @ (v_beta - kd@S), applying the inverse
    // AFTER the subtraction so its fp error is not amplified by the u - w@S cancellation.
    bcast_cols_mul(cb.kbeta, cb.decay_exp, cb.w, Ct, Kt);  // kd -> cb.w (output)
    WAIT(cb.w, ck);
    POP(cb.kbeta, ck);
    // cb.vbeta (v_beta) and cb.Tinv (T_inv) remain pushed for the writer; NOT popped here.

    // ---- intra = (q@k^T) * L_mask ; q_decay = q*decay_exp ; k_dec_t ----
    mm(Q, Kk, cb.scr1, Ct, Kt, Ct, true);  // qk = q @ k^T (Q/Kk = normalized q,k)
    WAIT(cb.scr1, cc);
    ew(cb.scr1, cb.lmask, cb.intra, cc, 2);
    WAIT(cb.intra, cc);
    POP(cb.scr1, cc);
    POP(cb.lmask, cc);
    bcast_cols_mul(Q, cb.decay_exp, cb.qdecay, Ct, Kt);
    WAIT(cb.qdecay, ck);
    POP(Q, ck);
    // decay_exp kept alive: reused at the scan to recompute dl = exp(g_sum).
    bcast_cols_mul(Kk, cb.decayfac, cb.scr1, Ct, Kt);  // k * exp(decay_last-decay)
    WAIT(cb.scr1, ck);
    POP(Kk, ck);
    // decayfac kept alive: reused at the scan to recompute dl = exp(g_sum).
    // k_dec_t = transpose(k_dec) [K,C]: transpose each [Ct,Kt] tile block into [Kt,Ct].
    cb_reserve_back(cb.kdec_t, Kt * Ct);
    pack_reconfig_data_format(cb.kdec_t);
    reconfig_data_format_srca(cb.scr1);  // unary: in->srcA
    transpose_init(cb.scr1);
    for (uint32_t ki = 0; ki < Kt; ki++) {
        for (uint32_t ci = 0; ci < Ct; ci++) {
            tile_regs_acquire();
            transpose_tile(cb.scr1, ci * Kt + ki, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb.kdec_t, ki * Ct + ci);
            tile_regs_release();
        }
    }
    cb_push_back(cb.kdec_t, Kt * Ct);
    POP(cb.scr1, ck);

    // ---- dl = exp(g_sum) = decayfac[i]*decay_exp[i] (same for all i); 1 tile, [0,0] holds dl.
    // The scan kernel uses it to decay the recurrent state: S <- S*dl + k_dec_t@v_new.
    ew(cb.decayfac, cb.decay_exp, cb.dl, 1, 2);
    WAIT(cb.dl, 1);
    POP(cb.decayfac, Ct);
    POP(cb.decay_exp, Ct);
    // u, w, k_dec_t, q_decay, intra, dl remain pushed in their CBs -> prep writer -> DRAM.
    // (They are NOT popped here; the writer drains them per chunk.)
}

// PHASE B (scan): one chunk of the sequential recurrence. cur_S = the state input CB for this
// chunk (reader-fed cb_S at chunk 0, then the compute-only ping-pong), dst = where the updated
// state goes (the other ping-pong CB, or the final-state CB on the last chunk).
inline void scan_step(const GdnScanCbs& cb, uint32_t cur_S, uint32_t dst, uint32_t Ct, uint32_t Kt, uint32_t Vt) {
    const uint32_t cc = Ct * Ct;
    const uint32_t ck = Ct * Kt;
    const uint32_t cv = Ct * Vt;
    const uint32_t kv = Kt * Vt;
    const uint32_t kc = Kt * Ct;

    // v_new = T_inv @ (v_beta - kd@S)  -- apply the inverse AFTER the subtraction so the WY
    // inverse's fp error is not amplified by the cancellation (vs the u - w@S form).
    WAIT(cb.kd, ck);
    WAIT(cur_S, kv);
    mm(cb.kd, cur_S, cb.scr1, Ct, Kt, Vt, false);  // kdS = kd @ S -> scr1
    WAIT(cb.scr1, cv);
    POP(cb.kd, ck);
    WAIT(cb.vbeta, cv);
    ew(cb.vbeta, cb.scr1, cb.ointer, cv, 1);  // diff = v_beta - kdS -> ointer
    WAIT(cb.ointer, cv);
    POP(cb.vbeta, cv);
    POP(cb.scr1, cv);
    WAIT(cb.Tinv, cc);
    mm(cb.Tinv, cb.ointer, cb.vnew, Ct, Ct, Vt, false);  // v_new = T_inv @ diff -> vnew
    WAIT(cb.vnew, cv);
    POP(cb.Tinv, cc);
    POP(cb.ointer, cv);

    // o = q_decay @ S + intra @ v_new
    WAIT(cb.qdecay, ck);
    mm(cb.qdecay, cur_S, cb.ointer, Ct, Kt, Vt, false);  // o_inter = q_decay @ S
    WAIT(cb.ointer, cv);
    POP(cb.qdecay, ck);
    WAIT(cb.intra, cc);
    mm(cb.intra, cb.vnew, cb.scr1, Ct, Ct, Vt, false);  // intra_v = intra @ v_new
    WAIT(cb.scr1, cv);
    POP(cb.intra, cc);
    ew(cb.ointer, cb.scr1, cb.out, cv, 0);  // o -> cb_out (drained by writer)
    POP(cb.ointer, cv);
    POP(cb.scr1, cv);

    // s_upd = k_dec_t @ v_new
    WAIT(cb.kdec_t, kc);
    mm(cb.kdec_t, cb.vnew, cb.supd, Kt, Ct, Vt, false);
    WAIT(cb.supd, kv);
    POP(cb.kdec_t, kc);
    POP(cb.vnew, cv);

    // S_new = cur_S * dl + s_upd  (dl scalar in cb_dl tile [0,0])
    WAIT(cb.dl, 1);
    bcast_scalar_mul(cur_S, cb.dl, cb.stmp, kv);
    WAIT(cb.stmp, kv);
    POP(cb.dl, 1);
    POP(cur_S, kv);
    ew(cb.stmp, cb.supd, dst, kv, 0);
    POP(cb.stmp, kv);
    POP(cb.supd, kv);
}
