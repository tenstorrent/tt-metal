// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Shared math functions for the chunk_gdn prep and scan compute kernels.
// The kernel .cpp files are calling prep_chunk / scan_step below. CB ids are passed in as plain
// uint32_t (via the GdnPrepCbs / GdnScanCbs structs for the composition functions), so the same
// bodies run under different CB maps.
//

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
#if defined(GDN_DECAY_SFPU)
// In-DST SFPU ops for the fused decay chain (GDN_DECAY_SFPU, set by the fused factory from the hashed decay_sfpu attr).
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/fill.h"
#endif

// GDN_HOIST_RECONFIG (a per-kernel define, set by the fused factory on the producer compute
// kernel only): hoist the packer/unpacker format reconfigs out of the WY hot path (invert16 /
// invert_block).
// It is a per-path PERF switch: the hoist helps on the fused producer,
// but hurts the phased kernel at some shapes.
#ifdef GDN_HOIST_RECONFIG
inline constexpr bool kGdnHoistReconfig = true;
#else
inline constexpr bool kGdnHoistReconfig = false;
#endif

// Sub-step device zones for the Tracy device profiler. Only in profiled builds where the
// profiler header was included before this one.
#if defined(PROFILE_KERNEL) && defined(DeviceZoneScopedN)
#define GDN_ZONE(name) DeviceZoneScopedN(name)
#else
#define GDN_ZONE(name)
#endif

// GDN_TINV_SFPU (a per-kernel define the prep factories set from the hashed `tinv` attr when it is
// GdnTinv::SFPU_FP32): replace the Ct == 1 WY inverse (invert_block's Horner quadrants, ~60 LLK calls) with
// ONE SFPU forward-substitution solve reading negN as fp32 (chunk_gdn_tinv_sfpu.hpp). It changes the
// arithmetic, so T_inv is PCC-class against the Horner path — but both the phased prep and the fused
// producer compile this same body for a given method, so fused == phased stays bit-exact per method.
#if defined(GDN_TINV_SFPU)
#if !defined(ARCH_BLACKHOLE)
#error "GDN_TINV_SFPU: the SFPU triangle solve is Blackhole-only"
#endif
#include "chunk_gdn_tinv_sfpu.hpp"
#endif

inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }

// out[Mt,Nt] = A[Mt,Kt] @ (tr ? B[Nt,Kt]^T : B[Kt,Nt]). Inputs must be available.
// Output tiles per DST acquire. With fp32 accumulation DST holds 8 tiles and the math/pack half-sync
// gives each side 4, so four independent output tiles ride one acquire/commit/wait/release round trip
// instead of four.
#ifndef GDN_DST_TILES
#define GDN_DST_TILES 1  // the prep kernel keeps the per-tile form: its Ct=2 binary sits at the 70,656 B limit
#endif
inline constexpr uint32_t kDstTiles = GDN_DST_TILES;

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
    if constexpr (kDstTiles == 1) {  // original per-tile form (byte-identical code for the prep kernel)
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
        return;
    }
    const uint32_t n_out = Mt * Nt;
    for (uint32_t t0 = 0; t0 < n_out; t0 += kDstTiles) {
        const uint32_t nb = (n_out - t0 < kDstTiles) ? (n_out - t0) : kDstTiles;
        tile_regs_acquire();
        for (uint32_t j = 0; j < nb; j++) {
            const uint32_t t = t0 + j;
            const uint32_t mi = t / Nt;
            const uint32_t ni = t - mi * Nt;
            for (uint32_t ki = 0; ki < Kt; ki++) {
                uint32_t bi = tr ? (ni * Kt + ki) : (ki * Nt + ni);
                matmul_tiles(a, b, mi * Kt + ki, bi, j);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nb; j++) {
            pack_tile(j, o, t0 + j);
        }
        tile_regs_release();
    }
    cb_push_back(o, Mt * Nt);
}

// out = A (op) B elementwise, n tiles. op: 0 add, 1 sub, 2 mul.
inline void ew(uint32_t a, uint32_t b, uint32_t o, uint32_t n, int op, bool skip_reconfig = false) {
    cb_reserve_back(o, n);
    if (!skip_reconfig) {
        pack_reconfig_data_format(o);
        reconfig_data_format(a, b);  // binary(a,b): a->srcA, b->srcB
    }
    if (op == 0) {
        add_init(a, b);
    } else if (op == 1) {
        sub_init(a, b);
    } else {
        mul_init(a, b);
    }
    if constexpr (kDstTiles == 1) {
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
        return;
    }
    for (uint32_t i0 = 0; i0 < n; i0 += kDstTiles) {
        const uint32_t nb = (n - i0 < kDstTiles) ? (n - i0) : kDstTiles;
        tile_regs_acquire();
        for (uint32_t j = 0; j < nb; j++) {
            if (op == 0) {
                add_tiles(a, b, i0 + j, i0 + j, j);
            } else if (op == 1) {
                sub_tiles(a, b, i0 + j, i0 + j, j);
            } else {
                mul_tiles(a, b, i0 + j, i0 + j, j);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nb; j++) {
            pack_tile(j, o, i0 + j);
        }
        tile_regs_release();
    }
    cb_push_back(o, n);
}

inline void expc(uint32_t in, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);  // unary: in->srcA
    copy_init(in);
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

// out[0] = copy of src[src_tile] (single 32x32 tile). src must be available.
inline void cpy_t(uint32_t src, uint32_t src_tile, uint32_t o, bool skip_reconfig = false) {
    cb_reserve_back(o, 1);
    if (!skip_reconfig) {  // see mm() for the skip_reconfig contract
        pack_reconfig_data_format(o);
        reconfig_data_format_srca(src);
    }
    copy_init(src);
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
        copy_init(src[i]);
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

#if defined(GDN_TINV_SFPU)
// T_inv = (I - negN)^-1 for ONE 32x32 tile by the SFPU forward substitution (RHS = I, so X = T_inv).
//   negN   : fp32 CB, tile 0 = -strictly_lower(N) (prep's cb.scr3), front-waited — exactly the
//            pre-negated factor the solve consumes (unit diagonal implicit). Read in place, no staging copy.
//   cb_eye : identity tile.
//   out    : cb.Tinv (fp32).
// Caller: WAIT(out, 1) before popping negN — L is read until T_inv is packed.
// kNeg (Ct == 1 prep): negN is instead kk*L_mask and the solve forms -tf32(kk*L) per element — bit-identical to
// solving prep's negN tile, without building it.
template <bool kNeg = false>
inline void sfpu_tinv(uint32_t negN, uint32_t cb_eye, uint32_t out) {
    CircularBuffer l(negN);
    cb_reserve_back(out, 1);
    // The solve loads/stores DEST rows in the SrcB-implied format: keep both source formats on the fp32
    // identity so DEST is read back as fp32.
    reconfig_data_format(cb_eye, cb_eye);
    pack_reconfig_data_format(out);
    copy_init(cb_eye);
    tile_regs_acquire();
    copy_tile(cb_eye, 0, 0);  // RHS = I -> DST[0]
    gdn_tinv_trisolve_tile_init();
    gdn_tinv_trisolve_tile<kNeg>(l, 0, /*idst_in=*/0, /*idst_out=*/1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(1, out, 0);
    tile_regs_release();
    cb_push_back(out, 1);
}
#endif

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

// In-kernel L2-norm over K. rowsum_k: o[Mt,1(broadcast)] = sum over the full K dim of
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
    copy_init(in);
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
    uint32_t mask;  // alias of the u slot in prep (3 quadrant-mask tiles; gb_flat: + selector tile 3)
    // ck-sized fp32 scratch (the prep kernel's cb_out slot, unused otherwise in prep). Holds the three
    // ck-tile pushes (q^2, k^2, k*decayfac) so scr1 only ever sees pushes of one size (cc == Ct at
    // Ct == 1). Mixed push sizes on one ring can run a push past fifo_limit (scr1 got 4,4,1,..,1,4).
    uint32_t sck;
};

// gb_flat head-column select: out[mi] = a[mi] @ b[b_tile] for mi < Mt (one matmul_tiles each, fp32
// dest, same init/reconfig sequence as mm()). mm() always addresses b's tile 0; this variant takes
// the selector's tile index because the selector lives as tile 3 of the mask CB (gb_flat keeps the
// CB region unchanged, see prep_chunk).
inline void sel_col(uint32_t a, uint32_t b, uint32_t b_tile, uint32_t o, uint32_t Mt) {
    cb_reserve_back(o, Mt);
    pack_reconfig_data_format(o);
    reconfig_data_format(b, a);  // matmul_tiles(a,b): in0=a->srcB, in1=b->srcA
    matmul_init(a, b, 0);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        tile_regs_acquire();
        matmul_tiles(a, b, mi, b_tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, mi);
        tile_regs_release();
    }
    cb_push_back(o, Mt);
}

// CB map for scan_step — one field per CB the body touches (the state CBs S/s2/s3/final are
// selected per chunk by the caller and passed as cur_S/dst).
struct GdnScanCbs {
    uint32_t dl, Tinv, out;
    uint32_t vbeta, kd, qdecay, intra;
    uint32_t vnew, ointer, kdec_t;
    uint32_t scr1;
};

// Column-vector SFPU helpers (prep_chunk_c1): the operand's data that matters is column 0 (consumers read it
// through a column broadcast), which lives in faces 0 and 2, so the SFPU pass skips faces 1 and 3
// (VectorMode::C). Column 0 is bit-identical to the full-tile op.
ALWI void gdn_exp_col_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_exponential,
        (false, DST_ACCUM_MODE, false, 8, true),
        idst,
        VectorMode::C,
        p_sfpu::kCONST_1_FP16B));
}
ALWI void gdn_rsqrt_col_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_rsqrt,
        (APPROX, 8, DST_ACCUM_MODE, false, false),
        idst,
        VectorMode::C));
}
ALWI void gdn_add_scalar_col_tile(uint32_t idst, uint32_t param) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, ADD_UNARY, 8, DST_ACCUM_MODE),
        idst,
        VectorMode::C,
        param));
}
ALWI void gdn_mul_scalar_col_tile(uint32_t idst, uint32_t param) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, MUL_UNARY, 8, DST_ACCUM_MODE),
        idst,
        VectorMode::C,
        param));
}
// exp over faces 0, 2, 3 of a lower-triangular-support tile: face 1 (rows 0-15, cols 16-31) is strictly upper,
// so its exp(0) = 1 is multiplied by tril's 0 right after; leaving the 0 in place gives the same +0.
ALWI void gdn_exp_lower_tile(uint32_t idst) {
#ifdef TRISC_MATH
    _llk_math_eltwise_sfpu_start_(idst);
    ckernel::sfpu::calculate_exponential<false, DST_ACCUM_MODE, false, 8, true>(p_sfpu::kCONST_1_FP16B);
    _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    ckernel::sfpu::calculate_exponential<false, DST_ACCUM_MODE, false, 8, true>(p_sfpu::kCONST_1_FP16B);
    _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    ckernel::sfpu::calculate_exponential<false, DST_ACCUM_MODE, false, 8, true>(p_sfpu::kCONST_1_FP16B);
    _llk_math_eltwise_sfpu_done_();
#endif
}

// ---- Ct == 1 prep (P3_FLAPREP) ----------------------------------------------------------------------
// Same per-element arithmetic as prep_chunk_generic<1,...> (same ops on the same CB-packed operands),
// but scheduled for the three-thread pipeline instead of one op per round trip:
//  * stage order: every stage starts with the next op of the longest dependency chain
//    (G -> decay -> P/decay_row -> D -> tril mask -> exp -> tril mask -> kk*L -> T_inv) and then
//    issues independent work (q/k norm, v_beta, k_beta, kd, q_decay, k_dec_t, dl) that hides the
//    pack -> L1 -> unpack latency of that op, so the unpacker rarely blocks on a CB wait;
//  * the q and k L2-norm chains run side by side (one op init per pair, 4 round trips instead of 8);
//  * ops that read the same inputs share one DST acquire and pack to several CBs;
//  * a 4-tile op (Kt/Vt == 4) rides one fp32 DST half instead of four acquire/pack round trips;
//  * column-vector SFPU ops (exp, rsqrt, +eps, *scale) run on faces 0/2 only, the D*tril exp on faces 0/2/3;
//  * SFPU T_inv: the solve reads kk*L and forms negN = -tf32(kk*L) per element (sfpu_tinv<true>), so the
//    diag(kk*L) and diag - kk*L ops are gone (Horner keeps them).
// Bit-exact with prep_chunk_generic<1, ...> (fused and phased; verified by torch.equal on o and final_state).
// CB use (all fp32, 1 tile unless noted; every CB holds one live value at a time):
//   sck(4): q^2 then k*decayfac   stmp(4): k^2 then k_n    supd(4): q_n     kbeta(4): k_beta
//   decay   decay_exp   decayfac: G (gb_flat) then decayfac   ointer: B (gb_flat)
//   scr1: g_sum, rk, kk*L   scr2: q sumsq, D [, diag]   scr3: k sumsq, D*tril [, negN]   ([] Horner only)
//   S: rq, kk   s2: decay_row, dl column   s3: g_sum - decay, qk   final_s: P, exp(D*tril)
// S/s2/s3/final_s are invert_block's private scratch on the Horner path; all are free by T_inv.
template <uint32_t Kt, uint32_t Vt, bool qk_norm, bool gb_flat>
inline void prep_chunk_c1(const GdnPrepCbs& cb, uint32_t scale_bits, uint32_t eps_bits) {
    static_assert(Kt <= 4 && Vt <= 4, "prep_chunk_c1: a Kt/Vt-tile op must fit one fp32 DST half (4 tiles)");
    constexpr uint32_t ck = Kt;
    constexpr uint32_t cv = Vt;

    WAIT(cb.q, ck);
    WAIT(cb.k, ck);
    WAIT(cb.v, cv);
    WAIT(cb.g, 1);
    WAIT(cb.beta, 1);

    const uint32_t G = gb_flat ? cb.decayfac : cb.g;
    const uint32_t B = gb_flat ? cb.ointer : cb.beta;
    const uint32_t Q = qk_norm ? cb.supd : cb.q;
    const uint32_t Kk = qk_norm ? cb.stmp : cb.k;

    // n tiles of a (op) bcast-col(b) -> o, one acquire. op: mul.
    auto bcast_cols_mul_n = [&](uint32_t a, uint32_t col, uint32_t o, uint32_t n) {
        cb_reserve_back(o, n);
        reconfig_data_format(a, col);
        mul_bcast_cols_init(a, col);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; i++) {
            mul_tiles_bcast_cols(a, col, i, 0, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; i++) {
            pack_tile(i, o, i);
        }
        tile_regs_release();
        cb_push_back(o, n);
    };
    // o = a (op) b, one tile each. op: 1 sub, 2 mul.
    auto ew1 = [&](uint32_t a, uint32_t b, uint32_t o, int op) {
        cb_reserve_back(o, 1);
        reconfig_data_format(a, b);
        if (op == 1) {
            sub_init(a, b);
        } else {
            mul_init(a, b);
        }
        tile_regs_acquire();
        if (op == 1) {
            sub_tiles(a, b, 0, 0, 0);
        } else {
            mul_tiles(a, b, 0, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, 0);
        tile_regs_release();
        cb_push_back(o, 1);
    };
    // o = exp(in), one tile (copy + SFPU exp, as expc). col: column-vector operand (faces 0, 2);
    // else a lower-triangular-support tile (faces 0, 2, 3).
    auto exp1 = [&](uint32_t in, uint32_t o, bool col) {
        cb_reserve_back(o, 1);
        reconfig_data_format_srca(in);
        copy_init(in);
        exp_tile_init();
        tile_regs_acquire();
        copy_tile(in, 0, 0);
        if (col) {
            gdn_exp_col_tile(0);
        } else {
            gdn_exp_lower_tile(0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, 0);
        tile_regs_release();
        cb_push_back(o, 1);
    };

    pack_reconfig_data_format(cb.decay);  // every CB this body packs is fp32

    // ---- S1: G/B head select (gb_flat), q^2 and k^2 ----
    {
        GDN_ZONE("c1 S1 sel+sq");
        if constexpr (gb_flat) {
            cb_reserve_back(G, 1);
            cb_reserve_back(B, 1);
            reconfig_data_format(cb.mask, cb.g);  // matmul_tiles(g, mask): in0=g->srcB, in1=mask->srcA
            matmul_init(cb.g, cb.mask, 0);
            tile_regs_acquire();
            matmul_tiles(cb.g, cb.mask, 0, 3, 0);
            matmul_tiles(cb.beta, cb.mask, 0, 3, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, G, 0);
            pack_tile(1, B, 0);
            tile_regs_release();
            cb_push_back(G, 1);
            cb_push_back(B, 1);
            POP(cb.g, 1);
            POP(cb.beta, 1);
        }
        if constexpr (qk_norm) {
            reconfig_data_format(cb.q, cb.q);
            mul_init(cb.q, cb.q);
            const uint32_t src[2] = {cb.q, cb.k};
            const uint32_t dst[2] = {cb.sck, cb.stmp};
            for (uint32_t s = 0; s < 2; s++) {
                cb_reserve_back(dst[s], ck);
                tile_regs_acquire();
                for (uint32_t i = 0; i < ck; i++) {
                    mul_tiles(src[s], src[s], i, i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < ck; i++) {
                    pack_tile(i, dst[s], i);
                }
                tile_regs_release();
                cb_push_back(dst[s], ck);
            }
        }
    }

    // ---- S2: decay = tril@G and g_sum = ones@G (one acquire); q/k sum of squares; v_beta ----
    {
        GDN_ZONE("c1 S2 decay+ss");
        WAIT(G, 1);
        cb_reserve_back(cb.decay, 1);
        cb_reserve_back(cb.scr1, 1);
        reconfig_data_format(G, cb.tril);  // matmul_tiles(tril, G): in0=tril->srcB, in1=G->srcA
        matmul_init(cb.tril, G, 0);
        tile_regs_acquire();
        matmul_tiles(cb.tril, G, 0, 0, 0);
        matmul_tiles(cb.ones, G, 0, 0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb.decay, 0);
        pack_tile(1, cb.scr1, 0);
        tile_regs_release();
        cb_push_back(cb.decay, 1);
        cb_push_back(cb.scr1, 1);
        POP(G, 1);
        if constexpr (qk_norm) {
            WAIT(cb.sck, ck);
            WAIT(cb.stmp, ck);
            cb_reserve_back(cb.scr2, 1);
            cb_reserve_back(cb.scr3, 1);
            reconfig_data_format(cb.ones, cb.sck);  // matmul_tiles(sq, ones): in0=sq->srcB, in1=ones->srcA
            matmul_init(cb.sck, cb.ones, 0);
            tile_regs_acquire();
            for (uint32_t ki = 0; ki < ck; ki++) {
                matmul_tiles(cb.sck, cb.ones, ki, 0, 0);
            }
            for (uint32_t ki = 0; ki < ck; ki++) {
                matmul_tiles(cb.stmp, cb.ones, ki, 0, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb.scr2, 0);
            pack_tile(1, cb.scr3, 0);
            tile_regs_release();
            cb_push_back(cb.scr2, 1);
            cb_push_back(cb.scr3, 1);
            POP(cb.sck, ck);
            POP(cb.stmp, ck);
        }
        WAIT(B, 1);
        bcast_cols_mul_n(cb.v, B, cb.vbeta, cv);  // v_beta (output)
        POP(cb.v, cv);
    }

    // ---- S3: from decay: decay_exp, decay_row, P = decay_i broadcast (one acquire); g_sum - decay;
    //          rq/rk = rsqrt(sumsq + eps) [* scale] for q and k (one acquire) ----
    {
        GDN_ZONE("c1 S3 exp+rsq");
        WAIT(cb.decay, 1);
        cb_reserve_back(cb.decay_exp, 1);
        cb_reserve_back(cb.s2, 1);
        cb_reserve_back(cb.final_s, 1);
        reconfig_data_format_srca(cb.decay);
        copy_init(cb.decay);
        exp_tile_init();
        tile_regs_acquire();
        copy_tile(cb.decay, 0, 0);
        gdn_exp_col_tile(0);
        transpose_init(cb.decay);
        transpose_tile(cb.decay, 0, 1);
        reconfig_data_format(cb.ones, cb.decay);
        mul_bcast_cols_init(cb.ones, cb.decay);
        mul_tiles_bcast_cols(cb.ones, cb.decay, 0, 0, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb.decay_exp, 0);
        pack_tile(1, cb.s2, 0);
        pack_tile(2, cb.final_s, 0);
        tile_regs_release();
        cb_push_back(cb.decay_exp, 1);
        cb_push_back(cb.s2, 1);
        cb_push_back(cb.final_s, 1);
        WAIT(cb.scr1, 1);
        ew1(cb.scr1, cb.decay, cb.s3, 1);  // g_sum - decay
        POP(cb.scr1, 1);
        POP(cb.decay, 1);
        if constexpr (qk_norm) {
            WAIT(cb.scr2, 1);
            WAIT(cb.scr3, 1);
            cb_reserve_back(cb.S, 1);
            cb_reserve_back(cb.scr1, 1);
            reconfig_data_format_srca(cb.scr2);
            copy_init(cb.scr2);
            tile_regs_acquire();
            copy_tile(cb.scr2, 0, 0);
            copy_tile(cb.scr3, 0, 1);
            binop_with_scalar_tile_init();
            gdn_add_scalar_col_tile(0, eps_bits);
            gdn_add_scalar_col_tile(1, eps_bits);
            rsqrt_tile_init();
            gdn_rsqrt_col_tile(0);
            gdn_rsqrt_col_tile(1);
            binop_with_scalar_tile_init();
            gdn_mul_scalar_col_tile(0, scale_bits);  // q only
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb.S, 0);
            pack_tile(1, cb.scr1, 0);
            tile_regs_release();
            cb_push_back(cb.S, 1);
            cb_push_back(cb.scr1, 1);
            POP(cb.scr2, 1);
            POP(cb.scr3, 1);
        }
    }

    // ---- S4: D = P - decay_row; decayfac = exp(g_sum - decay); q_n, k_n ----
    {
        GDN_ZONE("c1 S4 D+qkn");
        WAIT(cb.s2, 1);
        WAIT(cb.final_s, 1);
        cb_reserve_back(cb.scr2, 1);
        reconfig_data_format(cb.final_s, cb.s2);
        sub_bcast_rows_init(cb.final_s, cb.s2);
        tile_regs_acquire();
        sub_tiles_bcast_rows(cb.final_s, cb.s2, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb.scr2, 0);
        tile_regs_release();
        cb_push_back(cb.scr2, 1);
        POP(cb.s2, 1);
        POP(cb.final_s, 1);
        WAIT(cb.s3, 1);
        exp1(cb.s3, cb.decayfac, true);
        POP(cb.s3, 1);
        if constexpr (qk_norm) {
            WAIT(cb.S, 1);
            WAIT(cb.scr1, 1);
            bcast_cols_mul_n(cb.q, cb.S, cb.supd, ck);     // q_n (scale folded)
            bcast_cols_mul_n(cb.k, cb.scr1, cb.stmp, ck);  // k_n
            POP(cb.S, 1);
            POP(cb.scr1, 1);
            POP(cb.q, ck);
            POP(cb.k, ck);
        }
    }

    // ---- S5: D*tril; k_beta; k*decayfac; dl column = decayfac*decay_exp ----
    {
        GDN_ZONE("c1 S5 L2+kb");
        WAIT(cb.scr2, 1);
        ew1(cb.scr2, cb.tril, cb.scr3, 2);
        POP(cb.scr2, 1);
        WAIT(Kk, ck);
        bcast_cols_mul_n(Kk, B, cb.kbeta, ck);
        POP(B, 1);
        WAIT(cb.decayfac, 1);
        bcast_cols_mul_n(Kk, cb.decayfac, cb.sck, ck);  // k * exp(g_sum - decay)
        WAIT(cb.decay_exp, 1);
        ew1(cb.decayfac, cb.decay_exp, cb.s2, 2);
        POP(cb.decayfac, 1);
    }

    // ---- S6: exp(D*tril); kk = k_beta@k^T and qk = q@k^T ----
    {
        GDN_ZONE("c1 S6 L3+mm");
        WAIT(cb.scr3, 1);
        exp1(cb.scr3, cb.final_s, false);
        POP(cb.scr3, 1);
        WAIT(cb.kbeta, ck);
        WAIT(Q, ck);
        cb_reserve_back(cb.S, 1);
        cb_reserve_back(cb.s3, 1);
        reconfig_data_format(Kk, cb.kbeta);  // matmul_tiles(kbeta, Kk): in0=kbeta->srcB, in1=Kk->srcA
        matmul_init(cb.kbeta, Kk, 1);
        tile_regs_acquire();
        for (uint32_t ki = 0; ki < ck; ki++) {
            matmul_tiles(cb.kbeta, Kk, ki, ki, 0);
        }
        if constexpr (qk_norm) {  // Q is fp32 like k_beta: the same unpack formats serve both products
            for (uint32_t ki = 0; ki < ck; ki++) {
                matmul_tiles(Q, Kk, ki, ki, 1);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb.S, 0);
        if constexpr (qk_norm) {
            pack_tile(1, cb.s3, 0);
        }
        tile_regs_release();
        cb_push_back(cb.S, 1);
        if constexpr (!qk_norm) {  // bf16 Q: its own unpack format config
            reconfig_data_format(Kk, Q);
            matmul_init(Q, Kk, 1);
            tile_regs_acquire();
            for (uint32_t ki = 0; ki < ck; ki++) {
                matmul_tiles(Q, Kk, ki, ki, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb.s3, 0);
            tile_regs_release();
        }
        cb_push_back(cb.s3, 1);
        POP(Kk, ck);
    }

    // ---- S7: L_mask = exp(D*tril)*tril; k_dec_t = transpose(k*decayfac); dl = diag(dl column) ----
    {
        GDN_ZONE("c1 S7 Lm+kdt");
        WAIT(cb.final_s, 1);
        ew1(cb.final_s, cb.tril, cb.lmask, 2);
        POP(cb.final_s, 1);
        WAIT(cb.sck, ck);
        cb_reserve_back(cb.kdec_t, ck);
        reconfig_data_format_srca(cb.sck);
        transpose_init(cb.sck);
        tile_regs_acquire();
        for (uint32_t ki = 0; ki < ck; ki++) {
            transpose_tile(cb.sck, ki, ki);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t ki = 0; ki < ck; ki++) {
            pack_tile(ki, cb.kdec_t, ki);
        }
        tile_regs_release();
        cb_push_back(cb.kdec_t, ck);
        POP(cb.sck, ck);
        WAIT(cb.s2, 1);
        bcast_cols_mul_n(cb.eye, cb.s2, cb.dl, 1);
        POP(cb.s2, 1);
    }

    // ---- S8: kk*L_mask and intra = qk*L_mask (one acquire); q_decay ----
    {
        GDN_ZONE("c1 S8 kkL+qd");
        WAIT(cb.lmask, 1);
        WAIT(cb.S, 1);
        WAIT(cb.s3, 1);
        cb_reserve_back(cb.scr1, 1);
        cb_reserve_back(cb.intra, 1);
        reconfig_data_format(cb.S, cb.lmask);
        mul_init(cb.S, cb.lmask);
        tile_regs_acquire();
        mul_tiles(cb.S, cb.lmask, 0, 0, 0);
        mul_tiles(cb.s3, cb.lmask, 0, 0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb.scr1, 0);
        pack_tile(1, cb.intra, 0);
        tile_regs_release();
        cb_push_back(cb.scr1, 1);
        cb_push_back(cb.intra, 1);
        POP(cb.S, 1);
        POP(cb.s3, 1);
        POP(cb.lmask, 1);
        bcast_cols_mul_n(Q, cb.decay_exp, cb.qdecay, ck);  // q_decay (output)
        POP(Q, ck);
    }

    // ---- S9: kd = k_beta*decay_exp [Horner: + diag(kk*L)] ----
    {
        GDN_ZONE("c1 S9 kd");
#if !defined(GDN_TINV_SFPU)
        WAIT(cb.scr1, 1);
        ew1(cb.scr1, cb.eye, cb.scr2, 2);
#endif
        bcast_cols_mul_n(cb.kbeta, cb.decay_exp, cb.w, ck);  // kd (output)
        POP(cb.kbeta, ck);
        POP(cb.decay_exp, 1);
    }

    // ---- S10 (Horner only): negN = diag - kk*L = -(strictly_lower(kk*L)). The SFPU solve reads kk*L itself
    //      and forms negN's elements in flight (sfpu_tinv<true>), bit-identically. ----
#if !defined(GDN_TINV_SFPU)
    {
        GDN_ZONE("c1 S10 negN");
        WAIT(cb.scr2, 1);
        ew1(cb.scr2, cb.scr1, cb.scr3, 1);
        WAIT(cb.scr3, 1);
        POP(cb.scr1, 1);
        POP(cb.scr2, 1);
    }
#endif

    // ---- S11: T_inv = (I - negN)^-1 ----
    {
        GDN_ZONE("c1 S11 T_inv");
#if defined(GDN_TINV_SFPU)
        WAIT(cb.scr1, 1);
        sfpu_tinv<true>(cb.scr1, cb.eye, cb.Tinv);
        WAIT(cb.Tinv, 1);
        POP(cb.scr1, 1);
#else
        invert_block(cb.scr3, 0, cb.Tinv, cb.scr1, cb.scr2, cb.eye, cb.mask, cb.S, cb.final_s, cb.s2, cb.s3);
        WAIT(cb.Tinv, 1);
        POP(cb.scr3, 1);
#endif
    }
}

// PHASE A (prep): one state-independent (head, chunk) work-item. No recurrent state here; the
// sequential state scan lives in scan_step. Outputs (per chunk) v_beta, kd(->cb.w), T_inv,
// k_dec_t, q_decay, intra, dl are pushed to their CBs and streamed to DRAM by the prep writer.
// Ct/Kt/Vt/qk_norm are TEMPLATE parameters (not runtime args): the shape branches below must compile
// out exactly as the monolithic kernel's `if constexpr` did, or the Ct==2 prep program overflows the
// 70 KB kernel-config buffer (found on QB2 at chunk_size=64: 73344 > 70656 bytes).
// gb_flat is a template parameter for the same reason: its select block compiles out when off.
// Ct == 1 (Kt, Vt <= 4) dispatches to prep_chunk_c1 (above); prep_chunk_generic is every other shape.
template <uint32_t Ct, uint32_t Kt, uint32_t Vt, bool qk_norm, bool gb_flat = false>
inline void prep_chunk_generic(const GdnPrepCbs& cb, uint32_t scale_bits, uint32_t eps_bits);

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, bool qk_norm, bool gb_flat = false>
inline void prep_chunk(const GdnPrepCbs& cb, uint32_t scale_bits, uint32_t eps_bits) {
#if defined(GDN_DECAY_SFPU)
    // The SFPU decay chain lives in prep_chunk_generic (its P2 block); prep_chunk_c1 has its own FPU decay chain.
    constexpr bool kUseC1 = false;
#else
    constexpr bool kUseC1 = (Ct == 1 && Kt <= 4 && Vt <= 4);
#endif
    if constexpr (kUseC1) {
        prep_chunk_c1<Kt, Vt, qk_norm, gb_flat>(cb, scale_bits, eps_bits);
    } else {
        prep_chunk_generic<Ct, Kt, Vt, qk_norm, gb_flat>(cb, scale_bits, eps_bits);
    }
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, bool qk_norm, bool gb_flat>
inline void prep_chunk_generic(const GdnPrepCbs& cb, uint32_t scale_bits, uint32_t eps_bits) {
    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t C = Ct * 32;

    WAIT(cb.q, ck);
    WAIT(cb.k, ck);
    WAIT(cb.v, cv);
    WAIT(cb.g, Ct);
    WAIT(cb.beta, Ct);

    // ---- gb_flat (Option B, fused path only): the reader delivered RAW tiles of the model's
    // [B,T,HV] tensor into cb.g/cb.beta — every head sharing this (batch,chunk) got the IDENTICAL
    // Ct tiles, with only column h relevant to this work item. Select head h's column into column 0
    // with an exact one-hot matmul against the selector (out[:,0] = raw[:,h]); bit-exact (0/1
    // matmul, fp32 dest). ZERO CB GROWTH (a larger CB region in the prefill trace clobbered L1
    // buffers allocated after capture): the selector is tile 3 of the mask CB (the u slot holds
    // max(cv,3)+1 >= 5 tiles with the credit tile last; the reader loads the selector ONCE because a
    // fused producer core serves one head), and the selected g/beta go to the decayfac/decay_exp
    // slots — both are empty here (popped at the end of the previous item) and are first written
    // only after G/B are popped below (G after mm(ones,G); B after P1), so the aliasing is safe.
    // Downstream code reads G/B.
    uint32_t G = cb.g, B = cb.beta;
    {
        GDN_ZONE("gb select");
        if constexpr (gb_flat) {
            sel_col(cb.g, cb.mask, 3, cb.decayfac, Ct);  // g_sel -> decayfac slot
            WAIT(cb.decayfac, Ct);
            POP(cb.g, Ct);
            sel_col(cb.beta, cb.mask, 3, cb.decay_exp, Ct);  // beta_sel -> decay_exp slot
            WAIT(cb.decay_exp, Ct);
            POP(cb.beta, Ct);
            G = cb.decayfac;
            B = cb.decay_exp;
        }
    }

    // In-kernel L2-norm of q,k over K (fold q's scale). Consumes the raw reader q/k
    // and produces normalized q->cb.supd, k->cb.stmp (both free in Ct==1). The rest of the chunk
    // then reads Q/Kk instead of cb.q/cb.k. sck/scr2/scr3 are free here (used only later). ----
    uint32_t Q = cb.q, Kk = cb.k;
    {
        GDN_ZONE("qk norm");
        if constexpr (qk_norm) {
            // q: q^2 -> rowsum_K -> rsqrt(+eps)*scale -> q_normed (cb.supd)
            ew(cb.q, cb.q, cb.sck, ck, 2);
            WAIT(cb.sck, ck);
            rowsum_k(cb.sck, cb.scr2, Ct, Kt, cb.ones);
            WAIT(cb.scr2, Ct);
            POP(cb.sck, ck);
            inv_rms(cb.scr2, cb.scr3, Ct, eps_bits, scale_bits, /*do_scale=*/true);
            WAIT(cb.scr3, Ct);
            POP(cb.scr2, Ct);
            bcast_cols_mul(cb.q, cb.scr3, cb.supd, Ct, Kt);
            WAIT(cb.supd, ck);
            POP(cb.scr3, Ct);
            POP(cb.q, ck);
            // k: same, no scale -> k_normed (cb.stmp)
            ew(cb.k, cb.k, cb.sck, ck, 2);
            WAIT(cb.sck, ck);
            rowsum_k(cb.sck, cb.scr2, Ct, Kt, cb.ones);
            WAIT(cb.scr2, Ct);
            POP(cb.sck, ck);
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
    }

    // ---- P1: v_beta, k_beta ----
    {
        GDN_ZONE("beta mults");
        bcast_cols_mul(cb.v, B, cb.vbeta, Ct, Vt);
        WAIT(cb.vbeta, cv);
        bcast_cols_mul(Kk, B, cb.kbeta, Ct, Kt);
        WAIT(cb.kbeta, ck);
        POP(B, Ct);
        POP(cb.v, cv);
    }

    // ---- P2: decay = tril@g, decay_exp, decay_row ----
#if defined(GDN_DECAY_SFPU)
    // decay_sfpu (Ct == 1): the whole 1-tile decay chain in two DST round trips instead of ~13 single-tile ops that
    // each pay an unpack->math->pack round trip, in fp32 SFPU arithmetic (more accurate than the tf32 FPU operands).
    //   pass 1: decay = tril@G, gsum = ones@G (matmul); decayfac = exp(gsum - decay), decay_exp = exp(decay),
    //           exp(gsum) (SFPU, fp32 in DST).
    //   pass 2: L = tril(exp(tril(decay_i - decay_j))) with decay_j from an in-DST transpose, and dl*I.
    static_assert(Ct == 1, "GDN_DECAY_SFPU: Ct == 1 only");
    {
        const uint32_t cb_eg = cb.scr2;  // exp(g_sum), consumed by pass 2
        cb_reserve_back(cb.decay, 1);
        cb_reserve_back(cb.decay_exp, 1);
        cb_reserve_back(cb_eg, 1);
        pack_reconfig_data_format(cb.decay);
        reconfig_data_format(G, cb.tril);  // matmul_tiles(tril|ones, G): in0 -> srcB, G -> srcA
        tile_regs_acquire();
        matmul_init(cb.tril, G, 0);
        matmul_tiles(cb.tril, G, 0, 0, 0);  // DST0 = decay (column form)
        matmul_tiles(cb.tril, G, 0, 0, 2);  // DST2 = decay (second copy, becomes decay_exp)
        matmul_init(cb.ones, G, 0);
        matmul_tiles(cb.ones, G, 0, 0, 1);  // DST1 = g_sum (column form)
        POP(G, Ct);
        sub_binary_tile_init();
        sub_binary_tile(1, 0, 3);  // DST3 = g_sum - decay
        exp_tile_init();
        // Column-form tiles: only column 0 is ever read (bcast_cols / dl at (0,0)), so exponentiate the left
        // faces (0, 2) only. Faces 1 and 3 keep 0 instead of exp(0) = 1; nothing reads them.
        exp_tile(3, VectorMode::C);  // decayfac
        exp_tile(2, VectorMode::C);  // decay_exp
        exp_tile(1, VectorMode::C);  // exp(g_sum)
        tile_regs_commit();
        cb_reserve_back(cb.decayfac, 1);  // gb_flat: G lived in this slot; popped above
        tile_regs_wait();
        pack_tile(0, cb.decay, 0);
        pack_tile(2, cb.decay_exp, 0);
        pack_tile(3, cb.decayfac, 0);
        pack_tile(1, cb_eg, 0);
        tile_regs_release();
        cb_push_back(cb.decay, 1);
        cb_push_back(cb.decay_exp, 1);
        cb_push_back(cb.decayfac, 1);
        cb_push_back(cb_eg, 1);
    }
    WAIT(cb.decay_exp, Ct);
    WAIT(cb.decayfac, Ct);
    {
        WAIT(cb.decay, 1);
        cb_reserve_back(cb.lmask, 1);
        WAIT(cb.scr2, 1);
        cb_reserve_back(cb.dl, 1);
        pack_reconfig_data_format(cb.lmask);
        reconfig_data_format_srca(cb.decay);
        tile_regs_acquire();
        copy_init(cb.decay);
        copy_tile(cb.decay, 0, 0);  // DST0 = decay (column form)
        copy_tile(cb.decay, 0, 1);
        transpose_dest_init<true>(cb.decay);
        transpose_dest<true>(1);  // DST1 = decay (row form)
        fill_tile_init();
        fill_tile(2, 0.0f);
        sfpu_bcast_col_init();
        sfpu_add_bcast_col(2, 0);  // DST2[i][j] = decay_i
        sfpu_bcast_row_init();
        sfpu_sub_bcast_row(2, 1);  // DST2[i][j] = decay_i - decay_j
        copy_init(cb.tril);
        copy_tile(cb.tril, 0, 3);  // DST3 = tril
        mul_binary_tile_init();
        mul_binary_tile(2, 3, 2);  // zero the upper triangle before exp (it holds sums of -g >= 0)
        exp_tile_init();
        exp_tile(2);
        mul_binary_tile_init();
        mul_binary_tile(2, 3, 2);  // L_mask
        copy_init(cb.eye);
        copy_tile(cb.eye, 0, 0);   // DST0 = I
        copy_tile(cb.scr2, 0, 1);  // DST1 = exp(g_sum) (column form)
        sfpu_bcast_col_init();
        sfpu_mul_bcast_col(0, 1);  // dl*I
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(2, cb.lmask, 0);
        pack_tile(0, cb.dl, 0);
        tile_regs_release();
        cb_push_back(cb.lmask, 1);
        cb_push_back(cb.dl, 1);
        POP(cb.scr2, 1);
        POP(cb.decay, 1);
    }
    WAIT(cb.lmask, cc);
#else
    {
        GDN_ZONE("decay+exp");
        mm(cb.tril, G, cb.decay, Ct, Ct, 1, false);
        WAIT(cb.decay, Ct);
        expc(cb.decay, cb.decay_exp, Ct);
        WAIT(cb.decay_exp, Ct);
        transpose_col(cb.decay, cb.scr1, Ct);  // decay_row in scr1
        WAIT(cb.scr1, Ct);
    }

    // ---- L_mask = tril(exp(decay_i - decay_j)) ----
    {
        GDN_ZONE("L_mask");
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
    }

    // ---- decayfac = exp(g_sum - decay) ----
    // (dl = exp(g_sum) is recomputed at the scan from decayfac[0]*decay_exp[0] so its CB
    //  slot can be reused as the third ping-pong state buffer cb_s3.)
    {
        GDN_ZONE("decayfac");
        mm(cb.ones, G, cb.scr1, Ct, Ct, 1, false);  // g_sum in every row (col form)
        WAIT(cb.scr1, Ct);
        POP(G, Ct);
        ew(cb.scr1, cb.decay, cb.scr2, Ct, 1);  // g_sum - decay
        WAIT(cb.scr2, Ct);
        POP(cb.scr1, Ct);
        POP(cb.decay, Ct);
        expc(cb.scr2, cb.decayfac, Ct);
        WAIT(cb.decayfac, Ct);
        POP(cb.scr2, Ct);
    }
#endif  // GDN_DECAY_SFPU

    // ---- N = strictly_lower(k_beta@k^T * L_mask); T_inv = (I + strictly_lower)^-1 ----
    // The WY inverse, mirroring FLA's solve_tril: block down to 16x16 (invert_block splits each
    // 32x32 tile into 16-quadrants), invert the small diagonal blocks with bounded Horners, and
    // merge off-diagonal blocks exactly. This keeps every intermediate bounded, unlike a single
    // 32x32/full-matrix Horner whose deep power series loses fp32 precision on harder chunks.
    {
        GDN_ZONE("kk+negN");
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
    }

    // invert_block's private scratch A..D = cb.S/cb.final_s/cb.s2/cb.s3 — all fp32 and NOT drained
    // by the prep writer (unlike the output CBs cb.w/cb.qdecay/cb.intra, whose scratch pushes the
    // writer would wrongly consume). None alias src (cb.scr3), out, or the Ct==2 persistents
    // (cb.supd/cb.stmp).
    {
        GDN_ZONE("T_inv");
        if constexpr (Ct == 1) {
#if defined(GDN_TINV_SFPU)
        // One SFPU forward-substitution solve on negN in place; the Horner quadrants below are the reference.
        sfpu_tinv(cb.scr3, cb.eye, cb.Tinv);
        WAIT(cb.Tinv, cc);
        POP(cb.scr3, cc);
#else
        // Single 32x32 block: T_inv is just its inverse.
        invert_block(cb.scr3, 0, cb.Tinv, cb.scr1, cb.scr2, cb.eye, cb.mask, cb.S, cb.final_s, cb.s2, cb.s3);
        WAIT(cb.Tinv, cc);
        POP(cb.scr3, cc);
#endif
        } else if constexpr (Ct == 2) {
            // 2x2 tile-block lower-triangular. negN tiles: 0=(0,0), 2=(1,0), 3=(1,1); (0,1)=0.
            // Diagonal inverses Mi11, Mi22, then off-diagonal Mi21 = -Mi22 @ A21 @ Mi11.
            // (A21 = -negN21, so -Mi22@A21@Mi11 = Mi22 @ negN21 @ Mi11.)
            // Mi11 -> cb.supd, Mi22 -> cb.stmp, Mi21 -> cb.ointer (all free in prep).
            // Mi11 -> cb.supd (negN tile 0), Mi22 -> cb.stmp (negN tile 3). ONE inlined invert_block body
            // serves both through a 2-iteration loop the compiler must not unroll: inlining it twice
            // put the Ct==2 prep program over the kernel-config buffer (70752 > 70656 B on QB2), and the
            // LLK's inline asm forbids the out-of-line (noinline / -Os) alternatives. Same ops, same
            // order, same pack boundaries -> bit-exact with the unrolled form.
            {
                const uint32_t neg_tile[2] = {0, 3};
                const uint32_t inv_out[2] = {cb.supd, cb.stmp};
                // clang-format off
#pragma GCC unroll 1
            for (uint32_t i = 0; i < 2; i++) {
                invert_block(
                    cb.scr3,
                    neg_tile[i],
                    inv_out[i],
                    cb.scr1,
                    cb.scr2,
                    cb.eye,
                    cb.mask,
                    cb.S,
                    cb.final_s,
                    cb.s2,
                    cb.s3);
            }
            // clang-format on
            }
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
    }

    // ---- un-premultiplied WY hand-off: output v_beta (cb.vbeta), kd=k_beta*decay_exp (cb.w),
    // T_inv (cb.Tinv). The scan computes v_new = T_inv @ (v_beta - kd@S), applying the inverse
    // AFTER the subtraction so its fp error is not amplified by the u - w@S cancellation.
    {
        GDN_ZONE("kd");
        bcast_cols_mul(cb.kbeta, cb.decay_exp, cb.w, Ct, Kt);  // kd -> cb.w (output)
        WAIT(cb.w, ck);
        POP(cb.kbeta, ck);
        // cb.vbeta (v_beta) and cb.Tinv (T_inv) remain pushed for the writer; NOT popped here.
    }

    // ---- intra = (q@k^T) * L_mask ; q_decay = q*decay_exp ; k_dec_t ----
    {
        GDN_ZONE("intra");
        mm(Q, Kk, cb.scr1, Ct, Kt, Ct, true);  // qk = q @ k^T (Q/Kk = normalized q,k)
        WAIT(cb.scr1, cc);
        ew(cb.scr1, cb.lmask, cb.intra, cc, 2);
        WAIT(cb.intra, cc);
        POP(cb.scr1, cc);
        POP(cb.lmask, cc);
    }
    {
        GDN_ZONE("q_decay");
        bcast_cols_mul(Q, cb.decay_exp, cb.qdecay, Ct, Kt);
        WAIT(cb.qdecay, ck);
        POP(Q, ck);
        // decay_exp kept alive: reused at the scan to recompute dl = exp(g_sum).
    }
    {
        GDN_ZONE("k_dec_t");
        bcast_cols_mul(Kk, cb.decayfac, cb.sck, Ct, Kt);  // k * exp(decay_last-decay)
        WAIT(cb.sck, ck);
        POP(Kk, ck);
        // decayfac kept alive: reused at the scan to recompute dl = exp(g_sum).
        // k_dec_t = transpose(k_dec) [K,C]: transpose each [Ct,Kt] tile block into [Kt,Ct].
        cb_reserve_back(cb.kdec_t, Kt * Ct);
        pack_reconfig_data_format(cb.kdec_t);
        reconfig_data_format_srca(cb.sck);  // unary: in->srcA
        transpose_init(cb.sck);
        for (uint32_t ki = 0; ki < Kt; ki++) {
            for (uint32_t ci = 0; ci < Ct; ci++) {
                tile_regs_acquire();
                transpose_tile(cb.sck, ci * Kt + ki, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb.kdec_t, ki * Ct + ci);
                tile_regs_release();
            }
        }
        cb_push_back(cb.kdec_t, Kt * Ct);
        POP(cb.sck, ck);
    }

    // ---- dl*I: dl = exp(g_sum) = decayfac[i]*decay_exp[i] (the same value in every row of column 0),
    // broadcast down the identity -> one tile with dl on the diagonal. The scan decays the state as
    // the matmul (dl*I) @ S_tile so the update S <- dl*S + k_dec_t@v_new accumulates in one DST pass.
#if defined(GDN_DECAY_SFPU)
    POP(cb.decayfac, Ct);  // dl*I was produced by the SFPU decay chain
    POP(cb.decay_exp, Ct);
#else
    {
        GDN_ZONE("dl");
        ew(cb.decayfac, cb.decay_exp, cb.scr1, 1, 2);
        WAIT(cb.scr1, 1);
        bcast_cols_mul(cb.eye, cb.scr1, cb.dl, 1, 1);
        WAIT(cb.dl, 1);
        POP(cb.scr1, 1);
        POP(cb.decayfac, Ct);
        POP(cb.decay_exp, Ct);
    }
#endif
    // u, w, k_dec_t, q_decay, intra, dl remain pushed in their CBs -> prep writer -> DRAM.
    // (They are NOT popped here; the writer drains them per chunk.)
}
// (R10B: this line is load-bearing -- it nudges scan_step's zone pragmas off a 16-bit
// device-profiler hash collision that the prep sub-zones' line shift landed them on above.)

// ---- Ct == 1 scan step (P4_FLARCV) ------------------------------------------------------------------
// Same products and per-output accumulation order as scan_step_generic<1, ...>, so bit-identical:
//  * every block is a matmul_block row (ct_dim = Vt, rt_dim = 1): the in0 tile unpacks once per inner
//    index and serves the row's Vt output tiles (kdS, v_new, o, S_new: ~94 instead of ~144 tile unpacks);
//  * S_new is computed before o: o is off the S chain, so its matmuls hide the S_new pack -> next-step
//    kdS unpack round trip.
// o = q_decay @ S + intra @ v_new -> cb_out (drained by the writer) for the chunk whose q_decay / intra /
// v_new are at the front of their CBs and whose state is S; pops all four.
template <uint32_t Kt, uint32_t Vt>
inline void scan_o_c1(const GdnScanCbs& cb, uint32_t S) {
    constexpr uint32_t ck = Kt;
    constexpr uint32_t cv = Vt;
    constexpr uint32_t kv = Kt * Vt;
    GDN_ZONE("st_o");
    WAIT(cb.qdecay, ck);
    WAIT(cb.intra, 1);
    WAIT(cb.vnew, cv);
    WAIT(S, kv);
    cb_reserve_back(cb.out, cv);
    matmul_block_init(cb.qdecay, S, 0, cv, 1, 1);
    tile_regs_acquire();
    for (uint32_t ki = 0; ki < Kt; ki++) {
        matmul_block(cb.qdecay, S, ki, ki * cv, 0, 0, cv, 1, 1);
    }
    matmul_block(cb.intra, cb.vnew, 0, 0, 0, 0, cv, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < cv; j++) {
        pack_tile(j, cb.out, j);
    }
    tile_regs_release();
    cb_push_back(cb.out, cv);
    POP(cb.qdecay, ck);
    POP(cb.intra, 1);
    POP(cb.vnew, cv);
    POP(S, kv);
}

// kPipeO (fused receivers, GDN_SCAN_PIPE_O): o of the PREVIOUS chunk (prev_S; has_prev) runs right after this
// chunk's kdS, filling the kdS pack -> diff unpack round trip; this chunk's o, v_new and S stay in their CBs
// for the next step (or the caller's epilogue). Otherwise o runs at the end of the step, after S_new.
template <uint32_t Kt, uint32_t Vt, bool kPipeO = false>
inline void scan_step_c1(
    const GdnScanCbs& cb, uint32_t cur_S, uint32_t dst, uint32_t prev_S = 0, bool has_prev = false) {
    constexpr uint32_t ck = Kt;
    constexpr uint32_t cv = Vt;
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t kc = Kt;
    // Every block has ct_dim = cv, rt_dim = 1 (the MOP depends only on the shape; all operands fp32 32x32).
    auto mm_row_init = [&](uint32_t in0, uint32_t in1) { matmul_block_init(in0, in1, 0, cv, 1, 1); };

    // kdS = kd @ S -> scr1. S arrives one K tile-row at a time (the previous step pushes S_new per row),
    // so the k-th inner product waits only for row k. kdS is pushed per tile for the per-tile diff.
    {
        GDN_ZONE("st_kdS");
        WAIT(cb.kd, ck);
        cb_reserve_back(cb.scr1, cv);
        mm_row_init(cb.kd, cur_S);
        tile_regs_acquire();
        for (uint32_t ki = 0; ki < Kt; ki++) {
            WAIT(cur_S, (ki + 1) * cv);
            matmul_block(cb.kd, cur_S, ki, ki * cv, 0, 0, cv, 1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < cv; j++) {
            pack_tile(j, cb.scr1, 0);
            cb_push_back(cb.scr1, 1);
        }
        tile_regs_release();
        POP(cb.kd, ck);
    }
    if constexpr (kPipeO) {
        if (has_prev) {
            scan_o_c1<Kt, Vt>(cb, prev_S);
        }
    }
    // diff = v_beta - kdS -> ointer, and v_new = T_inv @ diff -> vnew: one DST acquire each (the math thread
    // never waits on the other DST half while the packer drains the previous block), but the unpacker waits
    // and the packer pushes per tile: tile j of each only needs tile j of its input.
    {
        GDN_ZONE("st_diff");
        WAIT(cb.vbeta, cv);
        cb_reserve_back(cb.ointer, cv);
        sub_init(cb.vbeta, cb.scr1);
        tile_regs_acquire();
        for (uint32_t j = 0; j < cv; j++) {
            WAIT(cb.scr1, j + 1);
            sub_tiles(cb.vbeta, cb.scr1, j, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < cv; j++) {
            pack_tile(j, cb.ointer, 0);
            cb_push_back(cb.ointer, 1);
        }
        tile_regs_release();
        POP(cb.vbeta, cv);
        POP(cb.scr1, cv);
    }
    {
        GDN_ZONE("st_vnew");
        WAIT(cb.Tinv, 1);
        cb_reserve_back(cb.vnew, cv);
        matmul_init(cb.Tinv, cb.ointer, 0);
        tile_regs_acquire();
        for (uint32_t j = 0; j < cv; j++) {
            WAIT(cb.ointer, j + 1);
            matmul_tiles(cb.Tinv, cb.ointer, 0, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < cv; j++) {
            pack_tile(j, cb.vnew, 0);
            cb_push_back(cb.vnew, 1);
        }
        tile_regs_release();
        POP(cb.Tinv, 1);
        POP(cb.ointer, cv);
    }
    // S_new = (dl*I) @ S + k_dec_t @ v_new -> dst, one DST row block (cv tiles) per K tile-row, each row
    // pushed as soon as it is packed. Row 0's decay product is issued before the wait for v_new.
    {
        GDN_ZONE("st_snew");
        WAIT(cb.kdec_t, kc);
        WAIT(cb.dl, 1);
        cb_reserve_back(dst, kv);
        mm_row_init(cb.dl, cur_S);
        for (uint32_t mi = 0; mi < Kt; mi++) {
            tile_regs_acquire();
            matmul_block(cb.dl, cur_S, 0, mi * cv, 0, 0, cv, 1, 1);
            WAIT(cb.vnew, cv);
            matmul_block(cb.kdec_t, cb.vnew, mi, 0, 0, 0, cv, 1, 1);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < cv; j++) {
                pack_tile(j, dst, j);
            }
            tile_regs_release();
            cb_push_back(dst, cv);
        }
        POP(cb.kdec_t, kc);
        POP(cb.dl, 1);
    }
    if constexpr (!kPipeO) {
        scan_o_c1<Kt, Vt>(cb, cur_S);
    }
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
inline void scan_step_generic(const GdnScanCbs& cb, uint32_t cur_S, uint32_t dst);

// PHASE B (scan): one chunk of the sequential recurrence. cur_S = the state input CB for this
// chunk (reader-fed cb_S at chunk 0, then the compute-only ping-pong), dst = where the updated
// state goes (the other ping-pong CB, or the final-state CB on the last chunk).
// Ct == 1 with Vt <= 4 (one fp32 DST half per output row) runs scan_step_c1.
template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
inline void scan_step(const GdnScanCbs& cb, uint32_t cur_S, uint32_t dst) {
    if constexpr (Ct == 1 && Vt <= 4) {
        scan_step_c1<Kt, Vt>(cb, cur_S, dst);
    } else {
        scan_step_generic<Ct, Kt, Vt>(cb, cur_S, dst);
    }
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
inline void scan_step_generic(const GdnScanCbs& cb, uint32_t cur_S, uint32_t dst) {
    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t kc = Kt * Ct;
    // Every CB the scan touches is fp32 (hand-off intermediates, state, scratch, o), so the
    // unpacker/packer formats set once at kernel start (compute_kernel_hw_startup on fp32 CBs) hold
    // for the whole step: skip the per-call reconfigs (the producer does the same under
    // GDN_HOIST_RECONFIG). Formats are identical either way => bit-exact.
    constexpr bool H = true;

    // v_new = T_inv @ (v_beta - kd@S)  -- apply the inverse after the subtraction so the WY
    // inverse's fp error is not amplified by the cancellation (vs the u - w@S form).
    {
        GDN_ZONE("st_kdS");
        WAIT(cb.kd, ck);
        WAIT(cur_S, kv);
        mm(cb.kd, cur_S, cb.scr1, Ct, Kt, Vt, false, H);  // kdS = kd @ S -> scr1
        WAIT(cb.scr1, cv);
        POP(cb.kd, ck);
    }
    {
        GDN_ZONE("st_diff");
        WAIT(cb.vbeta, cv);
        ew(cb.vbeta, cb.scr1, cb.ointer, cv, 1, H);  // diff = v_beta - kdS -> ointer
        WAIT(cb.ointer, cv);
        POP(cb.vbeta, cv);
        POP(cb.scr1, cv);
    }
    {
        GDN_ZONE("st_vnew");
        WAIT(cb.Tinv, cc);
        mm(cb.Tinv, cb.ointer, cb.vnew, Ct, Ct, Vt, false, H);  // v_new = T_inv @ diff -> vnew
        WAIT(cb.vnew, cv);
        POP(cb.Tinv, cc);
        POP(cb.ointer, cv);
    }

    // The two outputs are each a sum of two matmuls; both sums stay in DST: the second matmul
    // accumulates onto the first (MVMUL adds into DST, which only the packer zeroes at release), so
    // there is no packed intermediate, no re-unpack and no eltwise block. matmul_tiles binds its
    // operand CBs per call, so one matmul_init serves both products of a block.
    // o = q_decay @ S + intra @ v_new -> cb_out (drained by the writer)
    {
        GDN_ZONE("st_o");
        WAIT(cb.qdecay, ck);
        WAIT(cb.intra, cc);
        cb_reserve_back(cb.out, cv);
        matmul_init(cb.qdecay, cur_S, 0);
        for (uint32_t t0 = 0; t0 < cv; t0 += kDstTiles) {
            const uint32_t nb = (cv - t0 < kDstTiles) ? (cv - t0) : kDstTiles;
            tile_regs_acquire();
            for (uint32_t j = 0; j < nb; j++) {
                const uint32_t t = t0 + j;
                const uint32_t mi = t / Vt;
                const uint32_t ni = t - mi * Vt;
                for (uint32_t ki = 0; ki < Kt; ki++) {
                    matmul_tiles(cb.qdecay, cur_S, mi * Kt + ki, ki * Vt + ni, j);
                }
                for (uint32_t kc_ = 0; kc_ < Ct; kc_++) {
                    matmul_tiles(cb.intra, cb.vnew, mi * Ct + kc_, kc_ * Vt + ni, j);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nb; j++) {
                pack_tile(j, cb.out, t0 + j);
            }
            tile_regs_release();
        }
        cb_push_back(cb.out, cv);
        POP(cb.qdecay, ck);
        POP(cb.intra, cc);
    }
    // S_new = (dl*I) @ S + k_dec_t @ v_new -> dst (the next chunk's cur_S, or the final state).
    // The decay is block-diagonal, so every state tile (i,j) uses the single dl*I tile as in0.
    {
        GDN_ZONE("st_snew");
        WAIT(cb.kdec_t, kc);
        WAIT(cb.dl, 1);
        cb_reserve_back(dst, kv);
        matmul_init(cb.dl, cur_S, 0);
        for (uint32_t t0 = 0; t0 < kv; t0 += kDstTiles) {
            const uint32_t nb = (kv - t0 < kDstTiles) ? (kv - t0) : kDstTiles;
            tile_regs_acquire();
            for (uint32_t j = 0; j < nb; j++) {
                matmul_tiles(cb.dl, cur_S, 0, t0 + j, j);
            }
            for (uint32_t j = 0; j < nb; j++) {
                const uint32_t t = t0 + j;
                const uint32_t mi = t / Vt;
                const uint32_t ni = t - mi * Vt;
                for (uint32_t kc_ = 0; kc_ < Ct; kc_++) {
                    matmul_tiles(cb.kdec_t, cb.vnew, mi * Ct + kc_, kc_ * Vt + ni, j);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nb; j++) {
                pack_tile(j, dst, t0 + j);
            }
            tile_regs_release();
        }
        cb_push_back(dst, kv);
        POP(cb.kdec_t, kc);
        POP(cb.dl, 1);
    }
    // v_new and the input state fed both blocks; release them once everything is packed.
    POP(cb.vnew, cv);
    POP(cur_S, kv);
}
