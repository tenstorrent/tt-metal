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
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

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

#if defined(GDN_TINV_SFPU)
#include "api/compute/triangle_solve.h"
#endif

inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
// Ct is a template parameter of prep_chunk, so every per-tile loop in the helpers below unrolls fully; at Ct == 2
// that puts the prep program over the 70,656 B kernel-config buffer. The Ct == 2 build passes the tile count
// through this identity-asm so the compiler treats it as a runtime value and compiles each loop body once
// (the Ct == 1 build keeps the constant: its loops are single iterations anyway).
template <uint32_t Ct>
inline uint32_t tiles_dim(uint32_t v) {
    if constexpr (Ct > 1) {
        asm volatile("" : "+r"(v));
    }
    return v;
}
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

// Elementwise binary op selector for ew() / ewt().
enum class EwOp : uint8_t { Add, Sub, Mul };

// out = A (op) B elementwise, n tiles.
inline void ew(uint32_t a, uint32_t b, uint32_t o, uint32_t n, EwOp op, bool skip_reconfig = false) {
    cb_reserve_back(o, n);
    if (!skip_reconfig) {
        pack_reconfig_data_format(o);
        reconfig_data_format(a, b);  // binary(a,b): a->srcA, b->srcB
    }
    if (op == EwOp::Add) {
        add_init(a, b);
    } else if (op == EwOp::Sub) {
        sub_init(a, b);
    } else {
        mul_init(a, b);
    }
    if constexpr (kDstTiles == 1) {
        for (uint32_t i = 0; i < n; i++) {
            tile_regs_acquire();
            if (op == EwOp::Add) {
                add_tiles(a, b, i, i, 0);
            } else if (op == EwOp::Sub) {
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
            if (op == EwOp::Add) {
                add_tiles(a, b, i0 + j, i0 + j, j);
            } else if (op == EwOp::Sub) {
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

// The SFPU DST<->DST ops are inlined LLK bodies of several hundred bytes each and the fused windows below call
// them at many sites; kept out of line (one copy each) so the Ct == 2 prep program fits the kernel-config buffer.
[[gnu::noinline]] inline void sfpu_sub_dst(uint32_t a, uint32_t b, uint32_t o) { sub_binary_tile(a, b, o); }
[[gnu::noinline]] inline void sfpu_mul_dst(uint32_t a, uint32_t b, uint32_t o) { mul_binary_tile(a, b, o); }
[[gnu::noinline]] inline void sfpu_exp_dst(uint32_t d) { exp_tile(d); }

// ---- fused single-DST-pass prep blocks (C3). Each replaces a chain of one-op helper blocks whose
// intermediates were packed to scratch and re-unpacked; here the FPU op lands in DST and the rest of
// the chain runs on the SFPU (DST <-> DST) before one pack. fp32 CBs only.

// decay = tril @ g (col form), decay_exp = exp(decay), decayfac = exp(g_sum - decay) with g_sum = ones @ g, and
// dl = exp(g_sum) = decayfac_i * decay_exp_i, all from ONE DST pass per tile: the matmuls land in DST0 (decay,
// packed as is), DST3 (decay again, for the exp) and DST1 (g_sum); then SFPU: DST1 -= DST0, exp on DST1 and
// DST3, and DST2 = DST1 * DST3 -- dl as the exact fp32 product of the two exps, not the broadcast product whose
// rounded factors cost ~1e-3 on the value that decays the state every chunk. Packs: decay -> o_decay[i],
// decay_exp -> o_exp[i], decayfac -> o_fac[i], dl -> o_fac[Ct] (from tile 0's window; every row holds g_sum).
inline void decay_all(
    uint32_t tril, uint32_t ones, uint32_t g, uint32_t o_decay, uint32_t o_exp, uint32_t o_fac, uint32_t Ct) {
    cb_reserve_back(o_decay, Ct);
    cb_reserve_back(o_exp, Ct);
    cb_reserve_back(o_fac, Ct + 1);
    pack_reconfig_data_format(o_decay);  // all fp32
    reconfig_data_format(g, tril);       // matmul(tril|ones, g): in0->srcB, g->srcA
    matmul_init(tril, g, 0);
    for (uint32_t i = 0; i < Ct; i++) {
        tile_regs_acquire();
        for (uint32_t j = 0; j < Ct; j++) {
            matmul_tiles(tril, g, i * Ct + j, j, 0);  // DST0 = decay_i
            matmul_tiles(tril, g, i * Ct + j, j, 3);  // DST3 = decay_i (exp'd below)
            matmul_tiles(ones, g, i * Ct + j, j, 1);  // DST1 = g_sum
        }
        sub_binary_tile_init();
        sfpu_sub_dst(1, 0, 1);  // g_sum - decay_i
        exp_tile_init();
        sfpu_exp_dst(1);  // decayfac_i
        sfpu_exp_dst(3);  // decay_exp_i
        if (i == 0) {
            mul_binary_tile_init();
            sfpu_mul_dst(1, 3, 2);  // dl = decayfac_i * decay_exp_i = exp(g_sum)
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o_decay, i);
        pack_tile(3, o_exp, i);
        pack_tile(1, o_fac, i);
        if (i == 0) {
            pack_tile(2, o_fac, Ct);
        }
        tile_regs_release();
    }
    cb_push_back(o_decay, Ct);
    cb_push_back(o_exp, Ct);
    cb_push_back(o_fac, Ct + 1);
}

// L_mask[i,j] = tril * exp(decay_i - decay_j), one DST pass per output tile: decay_i (col broadcast of
// `decay`) in DST0, decay_j (row broadcast of `decay_row`) in DST1, then SFPU: DST0 -= DST1, the tril
// tile is copied to DST1 and applied before the exp (the upper triangle would overflow) and again after.
inline void lmask_fused(uint32_t ones, uint32_t decay, uint32_t decay_row, uint32_t tril, uint32_t o, uint32_t Ct) {
    cb_reserve_back(o, Ct * Ct);
    pack_reconfig_data_format(o);
    for (uint32_t mi = 0; mi < Ct; mi++) {
        for (uint32_t ni = 0; ni < Ct; ni++) {
            tile_regs_acquire();
            reconfig_data_format(ones, decay);  // bcast(a, b): a->srcA, b->srcB
            mul_bcast_cols_init(ones, decay);
            mul_tiles_bcast_cols(ones, decay, 0, mi, 0);  // DST0[i,j] = decay_i
            mul_bcast_rows_init(ones, decay_row);
            mul_tiles_bcast_rows(ones, decay_row, 0, ni, 1);  // DST1[i,j] = decay_j
            sub_binary_tile_init();
            sfpu_sub_dst(0, 1, 0);  // DST0 = decay_i - decay_j
            copy_init(tril);
            copy_tile(tril, mi * Ct + ni, 1);  // DST1 = tril tile
            mul_binary_tile_init();
            sfpu_mul_dst(0, 1, 0);  // zero the upper triangle before the exp
            exp_tile_init();
            sfpu_exp_dst(0);
            sfpu_mul_dst(0, 1, 0);  // * tril -> L_mask
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Ct + ni);
            tile_regs_release();
        }
    }
    cb_push_back(o, Ct * Ct);
}

// dl*I: the identity tile scaled by column 0 of tile `col_tile` of `col` (dl in every row).
inline void dl_tile(uint32_t eye, uint32_t col, uint32_t col_tile, uint32_t o) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(eye, col);  // bcast(a,col): a->srcA, col->srcB
    mul_bcast_cols_init(eye, col);
    tile_regs_acquire();
    mul_tiles_bcast_cols(eye, col, 0, col_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

// out[Mt,Nt] = -(A[Mt,Nt] * col[Mt,1]): the broadcast product lands in DST, the sign flip is an exact SFPU op on
// it before the pack (the same values as negating the column first, without staging the negated column).
inline void bcast_cols_mul_neg(uint32_t a, uint32_t col, uint32_t o, uint32_t Mt, uint32_t Nt) {
    cb_reserve_back(o, Mt * Nt);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, col);  // bcast(a,col): a->srcA, col->srcB
    mul_bcast_cols_init(a, col);
    negative_tile_init();
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(a, col, mi * Nt + ni, mi, 0);
            negative_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_push_back(o, Mt * Nt);
}

// negN = -strictly_lower(kk * L_mask), kk = k_beta @ k^T, one DST pass per output tile: kk lands in
// DST0 (Kt products, in1 transposed), L_mask is copied to DST1 and multiplied in; the constant (I - 1)
// -- 0 on the diagonal, -1 elsewhere -- is formed in DST2 from the eye tile and multiplied in, which drops the
// diagonal and flips the sign of the strictly-lower part exactly (the upper triangle is already zero from
// L_mask).
inline void negn_fused(uint32_t kbeta, uint32_t k, uint32_t lmask, uint32_t eye, uint32_t o, uint32_t Ct, uint32_t Kt) {
    cb_reserve_back(o, Ct * Ct);
    pack_reconfig_data_format(o);
    for (uint32_t mi = 0; mi < Ct; mi++) {
        for (uint32_t ni = 0; ni < Ct; ni++) {
            tile_regs_acquire();
            reconfig_data_format(k, kbeta);  // matmul(kbeta, k): kbeta->srcB, k->srcA
            matmul_init(kbeta, k, 1);
            for (uint32_t ki = 0; ki < Kt; ki++) {
                matmul_tiles(kbeta, k, mi * Kt + ki, ni * Kt + ki, 0);  // DST0 = kk tile
            }
            reconfig_data_format_srca(lmask);
            copy_init(lmask);
            copy_tile(lmask, mi * Ct + ni, 1);
            mul_binary_tile_init();
            sfpu_mul_dst(0, 1, 0);  // kk * L_mask
            copy_init(eye);
            copy_tile(eye, mi * Ct + ni, 2);
            binop_with_scalar_tile_init();
            add_unary_tile(2, 0xBF800000u);  // I - 1  (-1.0f)
            sfpu_mul_dst(0, 2, 0);           // negN
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Ct + ni);
            tile_regs_release();
        }
    }
    cb_push_back(o, Ct * Ct);
}

// intra = (q @ k^T) * L_mask, one DST pass per output tile: qk in DST0 (Kt products), L_mask copied to
// DST1 and multiplied in on the SFPU.
inline void intra_fused(uint32_t q, uint32_t k, uint32_t lmask, uint32_t o, uint32_t Ct, uint32_t Kt) {
    cb_reserve_back(o, Ct * Ct);
    pack_reconfig_data_format(o);
    for (uint32_t mi = 0; mi < Ct; mi++) {
        for (uint32_t ni = 0; ni < Ct; ni++) {
            tile_regs_acquire();
            reconfig_data_format(k, q);  // matmul(q, k): q->srcB, k->srcA (bf16 when the reader's raw q/k are used)
            matmul_init(q, k, 1);
            for (uint32_t ki = 0; ki < Kt; ki++) {
                matmul_tiles(q, k, mi * Kt + ki, ni * Kt + ki, 0);
            }
            reconfig_data_format_srca(lmask);
            copy_init(lmask);
            copy_tile(lmask, mi * Ct + ni, 1);
            mul_binary_tile_init();
            sfpu_mul_dst(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Ct + ni);
            tile_regs_release();
        }
    }
    cb_push_back(o, Ct * Ct);
}

// ---- fused qk-norm (C5). rowsum(x^2) is the diagonal of x @ x^T, so the per-row inverse rms comes out of one
// matmul window as a diagonal tile, and the normalization itself is one block-diagonal matmul: 2 blocks per
// operand instead of 4 (x^2, rowsum, rsqrt, broadcast), and no packed intermediates.

// D[mi] = diag(rsqrt(rowsum(x[mi,:]^2) + eps) [* scale]) for each row-tile of x [Ct, Kt], one DST pass per tile:
// the Kt products x[mi,ki] @ x[mi,ki]^T accumulate the row sums of squares on the diagonal of DST0 (the
// off-diagonal x_i.x_j are discarded); the identity is copied to DST1 and applied before and after the SFPU chain
// (+eps, rsqrt, *scale), so the off-diagonal rsqrt(eps) never leaves DST.
inline void inv_rms_diag(
    uint32_t x,
    uint32_t eye,
    uint32_t o,
    uint32_t Ct,
    uint32_t Kt,
    uint32_t eps_bits,
    uint32_t scale_bits,
    bool do_scale) {
    cb_reserve_back(o, Ct);
    pack_reconfig_data_format(o);
    for (uint32_t mi = 0; mi < Ct; mi++) {
        tile_regs_acquire();
        reconfig_data_format(x, x);  // matmul(x, x^T): x on both srcA and srcB
        matmul_init(x, x, 1);
        for (uint32_t ki = 0; ki < Kt; ki++) {
            matmul_tiles(x, x, mi * Kt + ki, mi * Kt + ki, 0);  // DST0 = x_mi @ x_mi^T
        }
        reconfig_data_format_srca(eye);
        copy_init(eye);
        copy_tile(eye, 0, 1);  // the identity block (cb.eye tile 0)
        mul_binary_tile_init();
        sfpu_mul_dst(0, 1, 0);  // diag(rowsum)
        binop_with_scalar_tile_init();
        add_unary_tile(0, eps_bits);  // + eps (off-diagonal: eps)
        rsqrt_tile_init();
        rsqrt_tile(0);  // off-diagonal: rsqrt(eps), finite
        if (do_scale) {
            binop_with_scalar_tile_init();
            mul_unary_tile(0, scale_bits);  // * scale (q only)
        }
        mul_binary_tile_init();
        sfpu_mul_dst(0, 1, 0);  // off-diagonal -> 0
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, mi);
        tile_regs_release();
    }
    cb_push_back(o, Ct);
}

// out[mi, ki] = D[mi] @ x[mi, ki]: block-diagonal left multiply (D holds Ct diagonal tiles), one product per tile.
inline void mm_diag(uint32_t d, uint32_t x, uint32_t o, uint32_t Ct, uint32_t Kt) {
    cb_reserve_back(o, Ct * Kt);
    pack_reconfig_data_format(o);
    reconfig_data_format(x, d);  // matmul(d, x): d->srcB, x->srcA
    matmul_init(d, x, 0);
    for (uint32_t mi = 0; mi < Ct; mi++) {
        for (uint32_t ki = 0; ki < Kt; ki++) {
            tile_regs_acquire();
            matmul_tiles(d, x, mi, mi * Kt + ki, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Kt + ki);
            tile_regs_release();
        }
    }
    cb_push_back(o, Ct * Kt);
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

// out[0] = a[ai] (op) b[bi], single tile; Add or Mul. (Like ew but with free tile indices.)
inline void ewt(uint32_t a, uint32_t ai, uint32_t b, uint32_t bi, uint32_t o, EwOp op, bool skip_reconfig = false) {
    cb_reserve_back(o, 1);
    if (!skip_reconfig) {  // see mm() for the skip_reconfig contract
        pack_reconfig_data_format(o);
        reconfig_data_format(a, b);
    }
    if (op == EwOp::Add) {
        add_init(a, b);
    } else {
        mul_init(a, b);
    }
    tile_regs_acquire();
    if (op == EwOp::Add) {
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
    ewt(tmpN, 0, cb_mask, 0, A, EwOp::Mul, kGdnHoistReconfig);
    CircularBuffer(A).wait_front(1);  // N00
    invert16(A, B, tmpT, cb_eye);
    CircularBuffer(B).wait_front(1);
    CircularBuffer(A).pop_front(1);  // Bi00 -> B
    // Bi11 = (I-N11)^-1  (N11 = bottom-right quadrant)
    ewt(tmpN, 0, cb_mask, 1, A, EwOp::Mul, kGdnHoistReconfig);
    CircularBuffer(A).wait_front(1);  // N11
    invert16(A, C, tmpT, cb_eye);
    CircularBuffer(C).wait_front(1);
    CircularBuffer(A).pop_front(1);  // Bi11 -> C
    // off = Bi11 @ N10 @ Bi00  (N10 = bottom-left quadrant; result lives only there)
    ewt(tmpN, 0, cb_mask, 2, A, EwOp::Mul, kGdnHoistReconfig);
    CircularBuffer(A).wait_front(1);  // N10
    CircularBuffer(tmpN).pop_front(1);
    mm(C, A, tmpT, 1, 1, 1, false, kGdnHoistReconfig);
    CircularBuffer(tmpT).wait_front(1);
    CircularBuffer(A).pop_front(1);  // Bi11@N10
    mm(tmpT, B, A, 1, 1, 1, false, kGdnHoistReconfig);
    CircularBuffer(A).wait_front(1);
    CircularBuffer(tmpT).pop_front(1);  // @Bi00 -> A(off)
    // out = Qtl*Bi00 + Qbr*Bi11 + off
    ewt(B, 0, cb_mask, 0, D, EwOp::Mul, kGdnHoistReconfig);
    CircularBuffer(D).wait_front(1);
    CircularBuffer(B).pop_front(1);  // Bi00_tl -> D
    ewt(C, 0, cb_mask, 1, B, EwOp::Mul, kGdnHoistReconfig);
    CircularBuffer(B).wait_front(1);
    CircularBuffer(C).pop_front(1);  // Bi11_br -> B
    ewt(D, 0, B, 0, C, EwOp::Add, true);
    CircularBuffer(C).wait_front(1);
    CircularBuffer(D).pop_front(1);
    CircularBuffer(B).pop_front(1);
    ewt(C, 0, A, 0, out, EwOp::Add, true);
    CircularBuffer(C).pop_front(1);
    CircularBuffer(A).pop_front(1);  // + off -> out
}

#if defined(GDN_TINV_SFPU)
// T_inv = (I - negN)^-1 of one 32x32 tile by forward substitution.
//   negN   : fp32 CB whose front tile is -strictly_lower(N) (unit diagonal implicit), front-waited by the caller.
//   cb_eye : CB holding the identity tile.
//   out    : fp32 CB that receives T_inv (one tile pushed).
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
    triangle_solve_tile_init();
    triangle_solve_tile<DataFormat::Float32, /*L_NEGATED=*/true>(l, 0, /*idst_in=*/0, /*idst_out=*/1);
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
    uint32_t vbeta, nkd, qdecay, intra;
    uint32_t vnew, ointer, kdec_t;
    uint32_t eye;  // one 32x32 identity tile (fp32), written by the reader at kernel start
};

// PHASE A (prep): one state-independent (head, chunk) work-item. No recurrent state here; the
// sequential state scan lives in scan_step. Outputs (per chunk) v_beta, nkd(->cb.w), T_inv,
// k_dec_t, q_decay, intra, dl are pushed to their CBs and streamed to DRAM by the prep writer.
// Ct/Kt/Vt/qk_norm are TEMPLATE parameters (not runtime args): the shape branches below must compile
// out exactly as the monolithic kernel's `if constexpr` did, or the Ct==2 prep program overflows the
// 70 KB kernel-config buffer (found on QB2 at chunk_size=64: 73344 > 70656 bytes).
template <uint32_t Ct, uint32_t Kt, uint32_t Vt, bool qk_norm>
inline void prep_chunk(const GdnPrepCbs& cb, uint32_t scale_bits, uint32_t eps_bits) {
    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t C = Ct * 32;
    // Runtime copies of the tile counts for the helper calls (identical code at Ct == 1; keeps the Ct == 2
    // build's per-tile loops rolled -- see tiles_dim).
    const uint32_t ct = tiles_dim<Ct>(Ct);
    const uint32_t cc_rt = ct * ct;
    const uint32_t ck_rt = ct * Kt;

    WAIT(cb.q, ck);
    WAIT(cb.k, ck);
    WAIT(cb.v, cv);
    WAIT(cb.g, Ct);
    WAIT(cb.beta, Ct);

    // In-kernel L2-norm of q,k over K (fold q's scale). Consumes the raw reader q/k
    // and produces normalized q->cb.supd, k->cb.stmp (both free in Ct==1). The rest of the chunk
    // then reads Q/Kk instead of cb.q/cb.k. scr1/scr2/scr3 are free here (used only later). ----
    uint32_t Q = cb.q, Kk = cb.k;
    {
        GDN_ZONE("pp_norm");
        if constexpr (qk_norm) {
            // q: D = diag(rsqrt(rowsum(q^2) + eps) * scale) from q @ q^T in one DST pass, then q_normed = D @ q
            // (cb.supd)
            inv_rms_diag(cb.q, cb.eye, cb.scr3, ct, Kt, eps_bits, scale_bits, /*do_scale=*/true);
            WAIT(cb.scr3, Ct);
            mm_diag(cb.scr3, cb.q, cb.supd, ct, Kt);
            WAIT(cb.supd, ck);
            POP(cb.scr3, Ct);
            POP(cb.q, ck);
            // k: same, no scale -> k_normed (cb.stmp)
            inv_rms_diag(cb.k, cb.eye, cb.scr3, ct, Kt, eps_bits, scale_bits, /*do_scale=*/false);
            WAIT(cb.scr3, Ct);
            mm_diag(cb.scr3, cb.k, cb.stmp, ct, Kt);
            WAIT(cb.stmp, ck);
            POP(cb.scr3, Ct);
            POP(cb.k, ck);
            Q = cb.supd;
            Kk = cb.stmp;
        }
    }

    {
        GDN_ZONE("pp_p1");
        // ---- P1: v_beta, k_beta ----
        // No wait on the outputs here: v_beta goes to the writer, k_beta is waited for by its first consumer
        // (pp_negn); the pops are ordered after this thread's unpacks by the CB protocol. Every block below follows
        // the same rule -- a WAIT right after a producing block drains the unpack->math->pack pipeline (~0.2-0.3 us
        // on a one-tile block), so it is placed only where the next block reads the result.
        bcast_cols_mul(cb.v, cb.beta, cb.vbeta, ct, Vt);
        bcast_cols_mul(Kk, cb.beta, cb.kbeta, ct, Kt);
        POP(cb.beta, Ct);
        POP(cb.v, cv);
    }

    {
        GDN_ZONE("pp_decay");
        // ---- P2: decay = tril@g, decay_exp, decayfac = exp(g_sum - decay), dl = exp(g_sum): one DST pass;
        // then decay_row ----
        decay_all(cb.tril, cb.ones, cb.g, cb.decay, cb.decay_exp, cb.decayfac, ct);
        WAIT(cb.decay, Ct);  // decay_exp / decayfac are waited for at pp_kd / pp_kdec
        POP(cb.g, Ct);
        transpose_col(cb.decay, cb.scr1, ct);  // decay_row in scr1
        WAIT(cb.scr1, Ct);
    }

    {
        GDN_ZONE("pp_lmask");
        // ---- L_mask = tril(exp(decay_i - decay_j)): one DST pass per tile (was five packed blocks) ----
        lmask_fused(cb.ones, cb.decay, cb.scr1, cb.tril, cb.lmask, ct);
        WAIT(cb.lmask, cc);
        POP(cb.scr1, Ct);  // decay_row done
        POP(cb.decay, Ct);
    }

    {
        GDN_ZONE("pp_negn");
        // ---- N = strictly_lower(k_beta@k^T * L_mask); T_inv = (I + strictly_lower)^-1 ----
        // The WY inverse, mirroring FLA's solve_tril: block down to 16x16 (invert_block splits each
        // 32x32 tile into 16-quadrants), invert the small diagonal blocks with bounded Horners, and
        // merge off-diagonal blocks exactly. This keeps every intermediate bounded, unlike a single
        // 32x32/full-matrix Horner whose deep power series loses fp32 precision on harder chunks.
        // negN = -(strictly_lower(kk * L_mask)) = -A_strict, kept in cb.scr3: one DST pass per tile (kk from the
        // matmul, L_mask and the (I - 1) mask applied on the SFPU) instead of four packed blocks.
        WAIT(cb.kbeta, ck);
        negn_fused(cb.kbeta, Kk, cb.lmask, cb.eye, cb.scr3, ct, Kt);
        WAIT(cb.scr3, cc);
    }

    {
        GDN_ZONE("pp_tinv");
        // invert_block's private scratch A..D = cb.S/cb.final_s/cb.s2/cb.s3 — all fp32 and NOT drained
        // by the prep writer (unlike the output CBs cb.w/cb.qdecay/cb.intra, whose scratch pushes the
        // writer would wrongly consume). None alias src (cb.scr3), out, or the Ct==2 persistents
        // (cb.supd/cb.stmp).
        if constexpr (Ct == 1) {
#if defined(GDN_TINV_SFPU)
            // One SFPU forward-substitution solve on negN in place.
            sfpu_tinv(cb.scr3, cb.eye, cb.Tinv);
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
            ew(cb.eye, cb.scr3, cb.Tinv, cc, EwOp::Add);
            WAIT(cb.Tinv, cc);
            for (uint32_t m = 2; m < C; m++) {
                mm(cb.scr3, cb.Tinv, cb.scr1, Ct, Ct, Ct, false);
                WAIT(cb.scr1, cc);
                POP(cb.Tinv, cc);
                ew(cb.eye, cb.scr1, cb.Tinv, cc, EwOp::Add);
                WAIT(cb.Tinv, cc);
                POP(cb.scr1, cc);
            }
            POP(cb.scr3, cc);
        }
    }

    {
        GDN_ZONE("pp_kd");
        // ---- un-premultiplied WY hand-off: output v_beta (cb.vbeta), nkd = -(k_beta*decay_exp) (cb.w),
        // T_inv (cb.Tinv). The scan computes v_new = T_inv @ (v_beta + nkd@S), applying the inverse
        // AFTER the subtraction so its fp error is not amplified by the u - w@S cancellation. The
        // operand is handed off NEGATED so the scan forms v_beta + nkd@S as one DST accumulation
        // (nkd @ S, then I @ v_beta accumulated onto it): the negation is an exact SFPU sign flip of
        // the broadcast product before it is packed.
        WAIT(cb.decay_exp, Ct);
        bcast_cols_mul_neg(cb.kbeta, cb.decay_exp, cb.w, ct, Kt);  // nkd -> cb.w (output, no wait)
        POP(cb.kbeta, ck);
    }
    // cb.vbeta (v_beta) and cb.Tinv (T_inv) remain pushed for the writer; NOT popped here.

    {
        GDN_ZONE("pp_intra");
        // ---- intra = (q@k^T) * L_mask ; q_decay = q*decay_exp ; k_dec_t ----
        intra_fused(Q, Kk, cb.lmask, cb.intra, ct, Kt);  // intra = (q @ k^T) * L_mask, one DST pass per tile
        POP(cb.lmask, cc);
    }
    {
        GDN_ZONE("pp_qdecay");
        bcast_cols_mul(Q, cb.decay_exp, cb.qdecay, ct, Kt);
        POP(Q, ck);
    }
    // decay_exp kept alive: reused at the scan to recompute dl = exp(g_sum).
    {
        GDN_ZONE("pp_kdec");
        WAIT(cb.decayfac, Ct + 1);
        bcast_cols_mul(Kk, cb.decayfac, cb.scr1, ct, Kt);  // k * exp(decay_last-decay)
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
    }

    {
        GDN_ZONE("pp_dl");
        // ---- dl*I: dl = exp(g_sum) (decayfac's extra tile Ct, the same value in every row of column 0)
        // broadcast down the identity -> one tile with dl on the diagonal. The scan decays the state as
        // the matmul (dl*I) @ S_tile so the update S <- dl*S + k_dec_t@v_new accumulates in one DST pass.
        dl_tile(cb.eye, cb.decayfac, Ct, cb.dl);
        POP(cb.decayfac, Ct + 1);
        POP(cb.decay_exp, Ct);
    }
    // u, w, k_dec_t, q_decay, intra, dl remain pushed in their CBs -> prep writer -> DRAM.
    // (They are NOT popped here; the writer drains them per chunk.)
}

// PHASE B (scan): one chunk of the sequential recurrence. cur_S = the state input CB for this
// chunk (reader-fed cb_S at chunk 0, then the compute-only ping-pong), dst = where the updated
// state goes (the other ping-pong CB, or the final-state CB on the last chunk).
template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
inline void scan_step(const GdnScanCbs& cb, uint32_t cur_S, uint32_t dst) {
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

    // v_new = T_inv @ (v_beta + nkd@S), nkd = -(k_beta*decay_exp) from prep -- apply the inverse
    // after the subtraction so the WY inverse's fp error is not amplified by the cancellation (vs the
    // u - w@S form). diff is formed in ONE DST pass of matmul accumulation: nkd @ S, then I @ v_beta
    // onto the same DST tile (MVMUL adds into DST; only the packer zeroes it at release). This replaces
    // the packed kd@S, its re-unpack and the eltwise block of the two-block form; the identity matmul
    // keeps the window inside one op class (no eltwise init or reconfig between the products). One
    // rounding moves: v_beta enters the sum through srcA like every other matmul operand instead of
    // through the eltwise add. diff itself must be packed: it is the in1 operand of T_inv @ diff, and
    // the WAIT on cb.ointer below is what keeps that block's unpacker behind this block's packer.
    {
        GDN_ZONE("st_diff");
        WAIT(cb.nkd, ck);
        WAIT(cb.vbeta, cv);
        WAIT(cur_S, kv);
        cb_reserve_back(cb.ointer, cv);
        matmul_init(cb.nkd, cur_S, 0);  // one init serves both products: matmul_tiles binds its CBs per call
        for (uint32_t t0 = 0; t0 < cv; t0 += kDstTiles) {
            const uint32_t nb = (cv - t0 < kDstTiles) ? (cv - t0) : kDstTiles;
            tile_regs_acquire();
            for (uint32_t j = 0; j < nb; j++) {
                const uint32_t t = t0 + j;
                const uint32_t mi = t / Vt;
                const uint32_t ni = t - mi * Vt;
                for (uint32_t ki = 0; ki < Kt; ki++) {
                    matmul_tiles(cb.nkd, cur_S, mi * Kt + ki, ki * Vt + ni, j);  // DST[j]  = nkd @ S
                }
                matmul_tiles(cb.eye, cb.vbeta, 0, t, j);  // DST[j] += I @ v_beta
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nb; j++) {
                pack_tile(j, cb.ointer, t0 + j);
            }
            tile_regs_release();
        }
        cb_push_back(cb.ointer, cv);
        POP(cb.nkd, ck);
        WAIT(cb.ointer, cv);  // the next block's unpacker must not run ahead of this block's packer
        POP(cb.vbeta, cv);
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
