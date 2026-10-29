// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compute: one core = (user u, value head h), T candidate tokens in rows r0 = u*T .. r0+T-1 (within the tile row).
// A. per chunk c (12): W_c = WO_W @ win_c + WO_N @ qkv_c (bf16 one-hots: an exact row gather; packed to bf16 for the
//    shifts AND to `wout` for the writer's window write); shift_j = SH_j @ W_c (row r0+t = W[t+j], exact); then the
//    conv in gdn_decode_step_conv.cpp's op order: DST_j = shift_j * tap_j (row-broadcast, bf16 x bf16 -> exact fp32
//    products), DST_0 += DST_1..3 (SFPU fp32), silu (SFPU), packed fp32 into qc | kc | vc (rows r0..r0+T-1, other
//    rows exactly 0). The plain kernel's placement matmul is not needed: rows already sit at r0+t, and its only
//    numeric effect (fp32 -> SrcB rounding) is idempotent with the one every consumer of qc/kc/vc applies.
// B. row-batched pre once: qn = l2norm(qc)*scale, kn = l2norm(kc) (mask_T), vm = vc*mask_T, kt = kn^T, T gate pairs
//    (a, b extracted from the a|b tile(s) by exact one-hot matmuls; AB2 = 1 when 2*Nv > 32 and b may sit in a second
//    tile, ab_in[b_idx]), zs = silu(z). GDN_FAST: the gate pairs arrive from the reader as ready scalar tiles.
// C. T serial delta-rule steps, state resident in L1 (state_in -> hn -> hn ...), post-token state copied to `hnew`
//    for the writer, the token's output row accumulated through the e_t row mask.
// D. row-batched post: out = rmsnorm(acc) * w * zs (rows outside the user's stay exactly 0 in the `out` tiles; the
//    writer emits only the user's T rows, and the last user's cores zero the padding rows of the tile row).
// Per-token arithmetic and helpers are gdn_decode_step_conv.cpp's; a T-step run equals T chained T = 1 runs.
// GDN_FAST / GDN_FP32_STATE select gdn_decode_step_conv_fast.cpp's arithmetic instead (gates on the reader, hnew
// packed from the state add's DST tile; fp32_state: SFPU decay / add on UnpackToDest operands, see the factory).
// VB > 1 (value blocks, GDN_VSPLIT): this core owns the head's state column tiles [v0, v0 + Vs), v0 = s*Vs. A converts
// only the q / k chunks and its own Vs v chunks (chunk index ci < Chs -> head chunk ci, or 2*Kt + v0 + (ci - 2*Kt));
// B / C run on Vs value tiles and KVs = Kt*Vs state tiles (tile (i, j) at i*Vs + j) with the same per-tile ops in
// the same order, so every value is the VB = 1 kernel's. The last token's output accumulator is packed into acc_mine
// (exact fp32) for the writer, which assembles the head's full pre-norm row in acc_x; D runs on acc_x exactly as on
// acc_fin and the core packs only its Vs output tiles.
// DFB block sizes (VB > 1): an indexed reserve_back(n) / pack_tile(i) / reduce over a DFB assumes the n-tile block
// does not straddle the ring end (the packer writes linearly past it while the per-tile pops wrap), so every block
// pushed into a DFB must keep its write pointer aligned to the DFB's other block sizes. `tmp` (max(Kt, Vt) tiles) only
// takes Kt / Vt blocks (the q / k squares, the norm's squares), as at VB = 1; the token loop's Vs-tile vm - vread goes
// through its own Vs-tile `tmpv`. Sharing `tmp` left its pointer at Vs * T (mod Vt) after the token loop, so the
// rmsnorm's squares were misread whenever Vs * T % Vt != 0 (the T = 1 seed at VB = 2 / 4, T = 2 at VB = 4): a wrong
// output scale, and the packer wrote past tmp into the next DFB.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/compute/gdn_step_helpers.hpp"
#include "api/debug/kernel_profiler.hpp"  // zones compile to nothing unless the build defines PROFILE_KERNEL

#ifndef GDN_FAST
#define GDN_FAST 0
#endif
#ifndef GDN_FP32_STATE
#define GDN_FP32_STATE 0
#endif
#ifndef GDN_VSPLIT
#define GDN_VSPLIT 0
#endif
#ifndef GDN_FUSED_DECAY
#define GDN_FUSED_DECAY 0  // kernel_opts bit 0 (fp32_state only): see outer_add_decay_sfpu
#endif
#if GDN_FUSED_DECAY && !GDN_FP32_STATE
#error "GDN_FUSED_DECAY needs GDN_FP32_STATE"
#endif
#ifndef GDN_GATE_SCALARS
#define GDN_GATE_SCALARS 0  // kernel_opts bit 1 (fp32_state only): beta / dec from the writer's gsc bits
#endif
#if GDN_GATE_SCALARS && !GDN_FP32_STATE
#error "GDN_GATE_SCALARS needs GDN_FP32_STATE"
#endif
#ifndef GDN_QKSHARE
#define GDN_QKSHARE 0  // kernel_opts bit 4 (VB > 1): this block convolves 2*Kt / VB q/k chunks; qc / kc from the writer
#endif
#if GDN_QKSHARE && !GDN_VSPLIT
#error "GDN_QKSHARE needs GDN_VSPLIT"
#endif
#ifndef GDN_A3_HALF
#define GDN_A3_HALF 0  // kernel_opts bit 5: the conv's SFPU tail on the tile half holding the user's rows only
#endif
#ifndef GDN_GSC_SPLIT
#define GDN_GSC_SPLIT 0  // kernel_opts bit 6 (GATE_SCALARS + VB > 1): gsc holds one 16-word slot per token
#endif
#if GDN_GSC_SPLIT && !(GDN_GATE_SCALARS && GDN_VSPLIT)
#error "GDN_GSC_SPLIT needs GDN_GATE_SCALARS and GDN_VSPLIT"
#endif
#ifndef GDN_VS_TAIL
#define GDN_VS_TAIL 0  // kernel_opts bit 7 (VB > 1): only this block's Vs normalized / gated output tiles
#endif
#if GDN_VS_TAIL && !GDN_VSPLIT
#error "GDN_VS_TAIL needs GDN_VSPLIT"
#endif
#ifndef GDN_HOIST_INV
#define GDN_HOIST_INV 0  // kernel_opts bit 3: the token-independent rsqrt(stats + eps) * scale of qn, once
#endif
#if GDN_GATE_SCALARS
// GATE_SCALARS: the writer hands over the T beta / decay fp32 bit patterns (gsc word t = beta[t], word T + t = dec[t])
// and the compute puts them into DST with fill_tile_bitcast -- the very fp32 values the reader's broadcast tiles held
// (an UnpackToDest copy of an all-equal fp32 tile and a bitcast fill give the same DST bits) -- so beta_t is packed
// here (fp32, exact) for the unchanged eb = e_t * beta_t, and the decay multiply reads dec from a filled DST tile.
#include "api/compute/eltwise_unary/fill.h"
#endif

using namespace gdn_step;

namespace {

// stats = row sums of `tmp` (n tiles); shared by the k l2norm, the q stats (kept through the token loop) and the
// rmsnorm (one DFB, three call sites)
inline void reduce_rows(uint32_t n) {
    compute_kernel_lib::
        reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, dfb::tmp, dfb::scaler, dfb::stats>(
            compute_kernel_lib::ReduceInputBlockShape::of(1, n));
}

constexpr uint32_t kOneBits = 0x3F800000u;  // 1.0f

#if !GDN_FAST
constexpr uint32_t kTwentyBits = 0x41A00000u;  // 20.0f (softplus threshold, as ttnn.softplus(1.0, 20.0))

// beta_t = sigmoid(b_s), dec = exp(neg_exp_A * softplus(a_s + dt_bias))  (all-equal scalar tiles), one DST acquire:
// DST0 <- b_s -> sigmoid; DST1 <- a_s, DST2 <- dtb, add, softplus, DST3 <- nea, mul, exp  -- gdn_decode_step_conv.cpp's
// gate_beta / gate_decay op order (DST index is not value-bearing)
inline void gates_beta_decay(DataflowBuffer& beta_t, DataflowBuffer& dec) {
    beta_t.reserve_back(1);
    dec.reserve_back(1);
    pack_reconfig_data_format(dfb::beta_t);  // beta_t and dec share the fp32 format
    reconfig_data_format_srca(dfb::b_s);     // a_s / b_s / dtb_s / nea_s are all fp32 scalar tiles
    copy_init(dfb::b_s);
    tile_regs_acquire();
    copy_tile(dfb::b_s, 0, 0);
    sigmoid_tile_init();
    sigmoid_tile(0);
    copy_init(dfb::a_s);
    copy_tile(dfb::a_s, 0, 1);
    copy_tile(dfb::dtb_s, 0, 2);
    add_binary_tile_init();
    add_binary_tile(1, 2, 1);
    softplus_tile_init();
    softplus_tile(1, kOneBits, kOneBits, kTwentyBits);
    copy_init(dfb::nea_s);
    copy_tile(dfb::nea_s, 0, 3);
    mul_binary_tile_init();
    mul_binary_tile(1, 3, 1);
    exp_tile_init<false>();
    exp_tile<false>(1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::beta_t, 0);
    pack_tile(1, dfb::dec, 0);
    tile_regs_release();
    beta_t.push_back(1);
    dec.push_back(1);
}
#endif  // !GDN_FAST

// zs[i] = silu(z[i])
#if GDN_VSPLIT
// noinline (VB > 1 only, called once per launch): inlined into the larger vsplit compute(), GCC + LTO can stop
// constant-folding the LLK ADDR_MOD setup behind silu_tile_init ("impossible constraint in 'asm'", as in
// gdn_decode_step_conv_fast_vsplit.cpp); VB = 1 keeps the original inline helper
__attribute__((noinline)) void silu_tiles(uint32_t a, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
#else
inline void silu_tiles(uint32_t a, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
#endif
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(a);
    copy_init(a);
    silu_tile_init();
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        copy_tile(a, i, 0);
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out[i] = a[first + i] (copy_tiles from tile `first`: the value block's output tiles of the full-row result)
inline void copy_tiles_from(uint32_t a, uint32_t first, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(a);
    copy_init(a);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        copy_tile(a, first + i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out[i] = (on[i] * w[i] per column) * zs[i]  (final product on the SFPU, fp32); packed in `out`'s format
inline void gated_out(uint32_t on, uint32_t w, uint32_t zs, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format(on, w);
    for (uint32_t i = 0; i < n; ++i) {
        mul_bcast_rows_init(on, w);
        tile_regs_acquire();
        mul_tiles_bcast_rows(on, w, i, i, 0);
        copy_init(zs);
        copy_tile(zs, i, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out[j] = (preload ? pre[j] : 0) + sum_i a[i] @ b[i*Nt + j], Nt <= 4 output tiles in one DST acquire (fp32 DST under
// the half-DST sync holds 4 tiles). With preload the accumulator tile is loaded into DST by a datacopy (exact for the
// packed fp32 values) and the token's row product accumulates onto its (zero) row exactly as the plain kernel's
// row_times_matrix does from a cleared DST; the other rows receive exact zeros (a's rows there are 0).
inline void row_times_matrix_acc(
    uint32_t pre,
    uint32_t a,
    uint32_t b,
    uint32_t out,
    DataflowBuffer& out_dfb,
    uint32_t Kt,
    uint32_t Nt,
    bool preload) {
    out_dfb.reserve_back(Nt);
    pack_reconfig_data_format(out);
    tile_regs_acquire();
    if (preload) {
        reconfig_data_format_srca(pre);
        copy_init(pre);
        for (uint32_t j = 0; j < Nt; ++j) {
            copy_tile(pre, j, j);
        }
    }
    reconfig_data_format<SrcOrder::Reverse>(a, b);
    matmul_init(a, b);
    for (uint32_t j = 0; j < Nt; ++j) {
        for (uint32_t i = 0; i < Kt; ++i) {
            matmul_tiles(a, b, i, i * Nt + j, j);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Nt; ++j) {
        pack_tile(j, out, j);
    }
    tile_regs_release();
    out_dfb.push_back(Nt);
}

// dst = l2norm(src) * post per row; rows whose `mask` column-0 entry is 0 -> exactly 0
template <uint32_t N>
inline void l2norm_rows(
    uint32_t src,
    uint32_t mask,
    uint32_t dst,
    DataflowBuffer& dst_dfb,
    DataflowBuffer& tmp,
    DataflowBuffer& stats,
    DataflowBuffer& scratch,
    DataflowBuffer& inv,
    uint32_t post_bits) {
    square_tiles(src, dfb::tmp, tmp, N);
    reduce_rows(N);
    stats.wait_front(1);
    inverse_l2(dfb::stats, dfb::eps_l2, mask, dfb::scratch, scratch, dfb::inv, inv, post_bits);
    stats.pop_front(1);
    inv.wait_front(1);
    scale_rows(src, dfb::inv, dst, dst_dfb, N);
    inv.pop_front(1);
}

// dst = rmsnorm(src) per row (zero rows stay exactly 0)
template <uint32_t N>
inline void rmsnorm_rows(
    uint32_t src,
    uint32_t dst,
    DataflowBuffer& dst_dfb,
    DataflowBuffer& tmp,
    DataflowBuffer& stats,
    DataflowBuffer& scratch,
    DataflowBuffer& inv,
    uint32_t inv_n_bits) {
    square_tiles(src, dfb::tmp, tmp, N);
    reduce_rows(N);
    stats.wait_front(1);
    inverse_rms(dfb::stats, dfb::eps_norm, dfb::scratch, scratch, dfb::inv, inv, inv_n_bits);
    stats.pop_front(1);
    inv.wait_front(1);
    scale_rows(src, dfb::inv, dst, dst_dfb, N);
    inv.pop_front(1);
}

#if GDN_HOIST_INV
// HOIST_INV: gdn_step::inverse_l2 split at its scratch tile. The first half -- sc = rsqrt(stats + eps) * post, the
// same op sequence and DST use -- depends only on the q row stats, eps and the scale, i.e. it is the same fp32 tile for
// every token, so it runs once before the token loop (into `qsc`); each token keeps only the second half, inv = sc *
// mask (e_t), on the same packed fp32 operand. Same bits as T calls of inverse_l2.
inline void l2_scale_once(
    uint32_t stats, uint32_t eps, uint32_t out, DataflowBuffer& out_dfb, uint32_t post_scale_bits) {
    out_dfb.reserve_back(1);
    pack_reconfig_data_format(out);
    reconfig_data_format(stats, eps);
    add_init(stats, eps);
    tile_regs_acquire();
    add_tiles(stats, eps, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, post_scale_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out, 0);
    tile_regs_release();
    out_dfb.push_back(1);
}

inline void l2_mask(uint32_t sc, uint32_t mask, uint32_t inv, DataflowBuffer& inv_dfb) {
    inv_dfb.reserve_back(1);
    pack_reconfig_data_format(inv);
    reconfig_data_format(sc, mask);
    mul_init(sc, mask);
    tile_regs_acquire();
    mul_tiles(sc, mask, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, inv, 0);
    tile_regs_release();
    inv_dfb.push_back(1);
}
#endif  // GDN_HOIST_INV

#if GDN_A3_HALF
// A3_HALF: run one SFPU face routine (the routine add_binary_tile / silu_tile run once per face) on the two faces of
// the tile half that holds the user's rows -- faces 0-1 (rows 0-15) or faces 2-3 (rows 16-31) -- exactly as
// _llk_math_eltwise_sfpu_apply_vector_mode_ does for VectorMode::R (faces 0-1), shifted by two faces for the bottom
// half. MATH thread only (the unpack / pack threads have nothing to do for an SFPU op).
template <typename F, typename... Args>
ALWI void sfpu_half(F f, uint32_t base_tile, bool bottom, Args... args) {
#ifdef TRISC_MATH
    _llk_math_eltwise_sfpu_start_(base_tile);
    if (bottom) {
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
#pragma GCC unroll 0  // one copy of the face routine per call site (code size: the 69 KB kernel config buffer)
    for (int face = 0; face < 2; face++) {
        f(args...);
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
#else
    (void)f;
    (void)base_tile;
    (void)bottom;
    ((void)args, ...);
#endif
}
#endif  // GDN_A3_HALF

#if GDN_VS_TAIL
// VS_TAIL: scale_rows / gated_out on tiles [first, first + n) of their row operands only (per-tile ops unchanged)
inline void scale_rows_from(uint32_t a, uint32_t first, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, b);
    mul_bcast_cols_init(a, b);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(a, b, first + i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

inline void gated_out_from(
    uint32_t on, uint32_t w, uint32_t zs, uint32_t first, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format(on, w);
    for (uint32_t i = 0; i < n; ++i) {
        mul_bcast_rows_init(on, w);
        tile_regs_acquire();
        mul_tiles_bcast_rows(on, w, i, first + i, 0);  // on holds this block's tiles only; w / zs the full row
        copy_init(zs);
        copy_tile(zs, first + i, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}
#endif  // GDN_VS_TAIL

#if !GDN_FAST
// gates of one token: a = ab[r0+t, h], b = ab[r0+t, Nv+h] extracted exactly with the bf16 one-hot selectors
// (g1[i] = rsel[t] @ ab_in[i]: every row = row r0+t of that a|b tile; a_s = g1[0] @ csel[0], b_s = g1[AB2] @ csel[1]:
// all-equal scalar tiles). AB2 = 0: one a|b tile (2*Nv <= 32), exactly the one-tile op order. AB2 = 1 (Nv = 24): b's
// row is gathered from ab_in[b_idx] (b_idx = 0 when this head's a and b share a tile) into g1[1]; single-tile
// one-hot matmuls are exact gathers, so a head whose pair shares a tile computes the same bits either way.
template <uint32_t AB2>
inline void gates_token(
    DataflowBuffer& rsel,
    DataflowBuffer& g1,
    DataflowBuffer& a_s,
    DataflowBuffer& b_s,
    DataflowBuffer& beta_t,
    DataflowBuffer& dec,
    uint32_t b_idx) {
    rsel.wait_front(1);
    g1.reserve_back(1 + AB2);
    pack_reconfig_data_format(dfb::g1);
    reconfig_data_format<SrcOrder::Reverse>(dfb::rsel, dfb::ab_in);
    matmul_init(dfb::rsel, dfb::ab_in);
    tile_regs_acquire();
    matmul_tiles(dfb::rsel, dfb::ab_in, 0, 0, 0);
    if constexpr (AB2 != 0) {
        matmul_tiles(dfb::rsel, dfb::ab_in, 0, b_idx, 1);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::g1, 0);
    if constexpr (AB2 != 0) {
        pack_tile(1, dfb::g1, 1);
    }
    tile_regs_release();
    g1.push_back(1 + AB2);
    g1.wait_front(1 + AB2);
    a_s.reserve_back(1);
    b_s.reserve_back(1);
    pack_reconfig_data_format(dfb::a_s);
    reconfig_data_format<SrcOrder::Reverse>(dfb::g1, dfb::csel);
    matmul_init(dfb::g1, dfb::csel);
    tile_regs_acquire();
    matmul_tiles(dfb::g1, dfb::csel, 0, 0, 0);
    matmul_tiles(dfb::g1, dfb::csel, AB2, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::a_s, 0);
    pack_tile(1, dfb::b_s, 0);
    tile_regs_release();
    a_s.push_back(1);
    b_s.push_back(1);
    g1.pop_front(1 + AB2);
    rsel.pop_front(1);
    a_s.wait_front(1);
    b_s.wait_front(1);
    gates_beta_decay(beta_t, dec);
    a_s.pop_front(1);
    b_s.pop_front(1);
}
#endif  // !GDN_FAST

#if !GDN_FP32_STATE
// out[i*Nt + j] = a[i] @ b[j] like gdn_step::outer_product, but up to 4 tiles per DST acquire (fp32 DST under the
// half-DST sync holds 4; bf16 DST holds 8) with one matmul init: each DST tile still receives exactly one K = 1 tile
// matmul, so the values are unchanged. The FP32_STATE path fuses the outer product into the state add instead.
inline void outer_product4(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t Kt, uint32_t Nt) {
    out_dfb.reserve_back(Kt * Nt);
    pack_reconfig_data_format(out);
    reconfig_data_format<SrcOrder::Reverse>(a, b);
    matmul_init(a, b);
    const uint32_t total = Kt * Nt;
    for (uint32_t k0 = 0; k0 < total; k0 += 4) {
        const uint32_t n = (total - k0) < 4 ? (total - k0) : 4;
        tile_regs_acquire();
        for (uint32_t q = 0; q < n; ++q) {
            const uint32_t k = k0 + q;
            matmul_tiles(a, b, k / Nt, k % Nt, q);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t q = 0; q < n; ++q) {
            pack_tile(q, out, k0 + q);
        }
        tile_regs_release();
    }
    out_dfb.push_back(total);
}
#endif  // !GDN_FP32_STATE

#if GDN_FAST && !GDN_FP32_STATE
// FAST: out[i] = a[i] + b[i], packed into two DFBs (hn for o = qn @ hn, hnew for the writer) from the same DST tile
// (gdn_decode_step_conv_fast.cpp's add_tiles_n_dual: the stored state is the full fp32 sum)
inline void add_tiles_n_dual(
    uint32_t a,
    uint32_t b,
    uint32_t out0,
    DataflowBuffer& out0_dfb,
    uint32_t out1,
    DataflowBuffer& out1_dfb,
    uint32_t n) {
    out0_dfb.reserve_back(n);
    out1_dfb.reserve_back(n);
    pack_reconfig_data_format(out0);
    reconfig_data_format(a, b);
    add_init(a, b);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        add_tiles(a, b, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out0, i);
        pack_tile(0, out1, i);
        tile_regs_release();
    }
    out0_dfb.push_back(n);
    out1_dfb.push_back(n);
}
#endif  // GDN_FAST && !GDN_FP32_STATE

#if GDN_FP32_STATE  // hd32 / hn32 DFBs are only allocated when the op attribute fp32_state is set
// FP32_STATE: hd = hd32 = S_cur * decay with an exact fp32 SFPU multiply (gdn_decode_step_conv_fast.cpp's
// decay_state_sfpu). cur32 (state_in at t = 0, else the previous token's hn32) and dec are UnpackToDest DFBs, so
// copy_tile lands the full fp32 values in DST. The same DST tile is packed into hd (matmul operand, UnpackToSrc) and
// hd32 (UnpackToDest, for the SFPU state add). Copy init sequence mirrors gate_decay / binary_ng sfpu no_bcast.
inline void decay_state_sfpu(
    uint32_t cur32, uint32_t dec_id, DataflowBuffer& hd_dfb, DataflowBuffer& hd32_dfb, uint32_t n, uint32_t dec_bits) {
#if !GDN_GATE_SCALARS
    (void)dec_bits;
#endif
    hd_dfb.reserve_back(n);
    hd32_dfb.reserve_back(n);
    pack_reconfig_data_format(dfb::hd);
    reconfig_data_format_srca(cur32);
    // The all-equal dec tile is loaded into DST[3] once per acquire and stays resident for up to 3 state tiles
    // (DST[0..2]); fp32 DST under the half-DST sync holds 4 tiles. Per element this is still S * dec in fp32 on the
    // SFPU (the multiply is commutative; the operands are the same exact fp32 values), so every bit is unchanged.
    constexpr uint32_t kChunk = 3;
    for (uint32_t i0 = 0; i0 < n; i0 += kChunk) {
        const uint32_t m = (n - i0) < kChunk ? (n - i0) : kChunk;
        tile_regs_acquire();
        copy_init(cur32);
        for (uint32_t jj = 0; jj < m; ++jj) {
            copy_tile(cur32, i0 + jj, jj);
        }
#if GDN_GATE_SCALARS
        (void)dec_id;
        fill_tile_init();
        fill_tile_bitcast(kChunk, dec_bits);  // the all-equal dec tile, from the writer's bits
#else
        copy_init(dec_id);  // FPU datacopy re-init after the SFPU ops of the previous iteration
        copy_tile(dec_id, 0, kChunk);
#endif
        mul_binary_tile_init();
        for (uint32_t jj = 0; jj < m; ++jj) {
            mul_binary_tile(jj, kChunk, jj);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t jj = 0; jj < m; ++jj) {
            pack_tile(jj, dfb::hd, i0 + jj);
            pack_tile(jj, dfb::hd32, i0 + jj);
        }
        tile_regs_release();
    }
    hd_dfb.push_back(n);
    hd32_dfb.push_back(n);
}

#if !GDN_FUSED_DECAY  // FUSED_DECAY has no hn32 DFB: outer_add_decay_sfpu replaces this helper
// FP32_STATE: hn = hnew = hn32 = hd32 + kt @ delta, the outer product and the exact fp32 SFPU state add fused in DST
// (was: outer_product -> `outer` DFB -> add_state_sfpu, 16 packs + 16 unpacks + 16 acquires per token more). Per
// acquire, 2 state tiles (fp32 DST under the half-DST sync holds 4 tiles): DST[jj] <- kt[i] @ delta[j0 + jj] (a K = 1
// tile matmul onto the freshly cleared DST tile, as outer_product did), DST[2 + jj] <- hd32 tile (UnpackToDest,
// exact), DST[jj] <- DST[jj] + DST[2 + jj] (SFPU fp32), then the sum is packed into hn, hnew and hn32. The values the
// add reads are the very fp32 values the old path round-tripped through the `outer` DFB (fp32 pack and
// UnpackToDest unpack are exact), and the add is commutative, so every output bit is unchanged. The `outer` DFB stays
// declared (the factory is not touched) but is not used.
inline void outer_add_state_sfpu(
    DataflowBuffer& hn_dfb, DataflowBuffer& hnew_dfb, DataflowBuffer& hn32_dfb, uint32_t Kt, uint32_t Vt) {
    constexpr uint32_t kPair = 2;
    hn_dfb.reserve_back(Kt * Vt);
    hnew_dfb.reserve_back(Kt * Vt);
    hn32_dfb.reserve_back(Kt * Vt);
    pack_reconfig_data_format(dfb::hn);  // hn, hnew, hn32 share the fp32 format
    for (uint32_t i = 0; i < Kt; ++i) {
        for (uint32_t j0 = 0; j0 < Vt; j0 += kPair) {
            const uint32_t n = (Vt - j0) < kPair ? (Vt - j0) : kPair;
            const uint32_t base = i * Vt + j0;
            tile_regs_acquire();
            reconfig_data_format<SrcOrder::Reverse>(dfb::kt, dfb::delta);
            matmul_init(dfb::kt, dfb::delta);
            for (uint32_t jj = 0; jj < n; ++jj) {
                matmul_tiles(dfb::kt, dfb::delta, i, j0 + jj, jj);
            }
            reconfig_data_format_srca(dfb::hd32);
            copy_init(dfb::hd32);
            for (uint32_t jj = 0; jj < n; ++jj) {
                copy_tile(dfb::hd32, base + jj, kPair + jj);
            }
            add_binary_tile_init();
            for (uint32_t jj = 0; jj < n; ++jj) {
                add_binary_tile(jj, kPair + jj, jj);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t jj = 0; jj < n; ++jj) {
                pack_tile(jj, dfb::hn, base + jj);
                pack_tile(jj, dfb::hnew, base + jj);
                pack_tile(jj, dfb::hn32, base + jj);
            }
            tile_regs_release();
        }
    }
    hn_dfb.push_back(Kt * Vt);
    hnew_dfb.push_back(Kt * Vt);
    hn32_dfb.push_back(Kt * Vt);
}
#endif  // !GDN_FUSED_DECAY

#if GDN_FUSED_DECAY
// FUSED_DECAY (kernel_opts bit 0): outer_add_state_sfpu without the hn32 round trip. S_next = hd32 + kt @ delta is
// formed in DST exactly as above and packed into hn (o operand) and hnew (writer); when the token has a successor
// (with_next), the successor's decayed state S_next * dec[t+1] is formed in the SAME acquire from the DST tiles that
// hold S_next -- decay_state_sfpu's exact fp32 SFPU multiply with the same operand order (state, dec), on the same
// fp32 values hn32 + UnpackToDest would have reloaded (fp32 pack / UnpackToDest copy are exact) -- and packed into hd
// and hd32 for the next token. DST per pair (fp32, half-DST sync: 4 tiles): [0..1] S_next, [2..3] the hd32 operands,
// then [2] <- dec[t+1] (UnpackToDest copy), [3] = [0] * [2], [2] = [1] * [2] (in place on the dec copy: the SFPU reads
// each element before it writes it). hd32 is consumed and refilled pair by pair (pop n, then reserve n: the next
// token's tiles take the slots just freed, so the KVs-entry ring needs no second block); hd was popped after vread.
inline void outer_add_decay_sfpu(
    DataflowBuffer& hn_dfb,
    DataflowBuffer& hnew_dfb,
    DataflowBuffer& hd_dfb,
    DataflowBuffer& hd32_dfb,
    uint32_t Kt,
    uint32_t Vt,
    bool with_next,
    uint32_t next_dec_bits) {
#if !GDN_GATE_SCALARS
    (void)next_dec_bits;
#endif
    constexpr uint32_t kPair = 2;
    hn_dfb.reserve_back(Kt * Vt);
    hnew_dfb.reserve_back(Kt * Vt);
    if (with_next) {
        hd_dfb.reserve_back(Kt * Vt);
    }
    pack_reconfig_data_format(dfb::hn);  // hn, hnew, hd, hd32 share the fp32 format
    for (uint32_t i = 0; i < Kt; ++i) {
        for (uint32_t j0 = 0; j0 < Vt; j0 += kPair) {
            const uint32_t n = (Vt - j0) < kPair ? (Vt - j0) : kPair;
            const uint32_t base = i * Vt + j0;
            tile_regs_acquire();
            reconfig_data_format<SrcOrder::Reverse>(dfb::kt, dfb::delta);
            matmul_init(dfb::kt, dfb::delta);
            for (uint32_t jj = 0; jj < n; ++jj) {
                matmul_tiles(dfb::kt, dfb::delta, i, j0 + jj, jj);
            }
            reconfig_data_format_srca(dfb::hd32);
            copy_init(dfb::hd32);
            for (uint32_t jj = 0; jj < n; ++jj) {
                copy_tile(dfb::hd32, jj, kPair + jj);  // this pair's tiles sit at the front (earlier pairs popped)
            }
            add_binary_tile_init();
            for (uint32_t jj = 0; jj < n; ++jj) {
                add_binary_tile(jj, kPair + jj, jj);
            }
            if (with_next) {
#if GDN_GATE_SCALARS
                fill_tile_init();
                fill_tile_bitcast(kPair, next_dec_bits);  // dec[t+1], from the writer's bits
#else
                copy_init(dfb::dec);
                copy_tile(dfb::dec, 0, kPair);  // dec[t+1] (dec[t] was popped after this token's own decay)
#endif
                mul_binary_tile_init();
                mul_binary_tile(0, 2, 3);
                if (n == kPair) {
                    mul_binary_tile(1, 2, 2);
                }
            }
            tile_regs_commit();
            hd32_dfb.pop_front(n);
            if (with_next) {
                hd32_dfb.reserve_back(n);
            }
            tile_regs_wait();
            for (uint32_t jj = 0; jj < n; ++jj) {
                pack_tile(jj, dfb::hn, base + jj);
                pack_tile(jj, dfb::hnew, base + jj);
            }
            if (with_next) {
                pack_tile(3, dfb::hd, base);
                pack_tile(3, dfb::hd32, 0);
                if (n == kPair) {
                    pack_tile(2, dfb::hd, base + 1);
                    pack_tile(2, dfb::hd32, 1);
                }
            }
            tile_regs_release();
            if (with_next) {
                hd32_dfb.push_back(n);
            }
        }
    }
    hn_dfb.push_back(Kt * Vt);
    hnew_dfb.push_back(Kt * Vt);
    if (with_next) {
        hd_dfb.push_back(Kt * Vt);
    }
}
#endif  // GDN_FUSED_DECAY
#endif  // GDN_FP32_STATE

}  // namespace

template <
    uint32_t Kt,
    uint32_t Vt,
    uint32_t T,
    uint32_t K,
    uint32_t AB2,
    uint32_t scale_bits,
    uint32_t inv_dv_bits,
    uint32_t VB>
TT_KERNEL void compute(uint32_t u, uint32_t b_idx, uint32_t s) {
    (void)u;
#if GDN_A3_HALF
    // the user's rows r0 .. r0+T-1 of the tile row: one tile half (faces 0-1 or 2-3) when they do not cross row 16
    const uint32_t a3_r0 = (u * T) % 32;
    const bool a3_half = (a3_r0 / 16) == ((a3_r0 + T - 1) / 16);
    const bool a3_bottom = a3_r0 >= 16;
#endif
#if GDN_GSC_SPLIT
    // gsc word of beta[t] / dec[t]: one 16-word (64 B) slot per token, so each value block writes its tokens' slots
    // into every block with 64 B-aligned NoC writes
    auto gsc_beta = [](uint32_t t) -> uint32_t { return 16 * t; };
    auto gsc_dec = [](uint32_t t) -> uint32_t { return 16 * t + 1; };
#elif GDN_GATE_SCALARS
    auto gsc_beta = [](uint32_t t) -> uint32_t { return t; };
    auto gsc_dec = [](uint32_t t) -> uint32_t { return T + t; };
#endif
    constexpr uint32_t Ch = 2 * Kt + Vt;
    static_assert(VB >= 1 && Vt % VB == 0, "value blocks must divide Vt");
    static_assert((VB > 1) == (GDN_VSPLIT != 0), "GDN_VSPLIT must be defined exactly when VB > 1");
    constexpr uint32_t Vs = Vt / VB;       // this block's value (state column) tiles; Vt at VB = 1
    constexpr uint32_t KVs = Kt * Vs;      // this block's state tiles; Kt*Vt at VB = 1
#if GDN_QKSHARE
    // QKSHARE: this block rebuilds and convolves only its q/k share [s*Qk, (s+1)*Qk) and its own v chunks (block order,
    // ChA = Chs = Qk + Vs); its Qk conv tiles go out through qkm and every block's qc / kc arrive from the writer
    constexpr uint32_t Qk = (2 * Kt) / VB;
    static_assert(Qk >= 1 && Kt % Qk == 0, "QKSHARE: a block's q/k share must lie within q or within k");
    constexpr uint32_t Chs = Qk + Vs;
    constexpr uint32_t ChA = Chs;  // src_in / taps / wc / wout hold this block's chunks only
#else
    constexpr uint32_t Chs = 2 * Kt + Vs;  // chunks this block convolves: q | k | its own v
    constexpr uint32_t ChA = Ch;           // src_in / taps / wc / wout hold all of the head's chunks
#endif
    const uint32_t v0 = s * Vs;            // first value tile of this block (0 at VB = 1)
    // head chunk of this block's chunk ci (the identity at VB = 1)
#if GDN_QKSHARE
    auto head_chunk = [](uint32_t ci) -> uint32_t { return ci; };  // wc / taps are already in block order
    (void)v0;
#else
    auto head_chunk = [v0](uint32_t ci) -> uint32_t { return ci < 2 * Kt ? ci : ci + v0; };
#endif
    DataflowBuffer src_in(dfb::src_in);
    DataflowBuffer taps(dfb::taps);
    DataflowBuffer z_in(dfb::z_in);
    DataflowBuffer sel(dfb::sel);
    DataflowBuffer mask_T(dfb::mask_T);
    DataflowBuffer e_t(dfb::e_t);
    DataflowBuffer w_in(dfb::w_in);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer eps_l2(dfb::eps_l2);
    DataflowBuffer eps_norm(dfb::eps_norm);
    DataflowBuffer state_in(dfb::state_in);
    DataflowBuffer wc(dfb::wc);
    DataflowBuffer shift(dfb::shift);
    DataflowBuffer cv(dfb::cv);
    DataflowBuffer tmp(dfb::tmp);
    DataflowBuffer stats(dfb::stats);
    DataflowBuffer scratch(dfb::scratch);
    DataflowBuffer inv(dfb::inv);
    DataflowBuffer qc(dfb::qc);
    DataflowBuffer kc(dfb::kc);
#if GDN_QKSHARE
    DataflowBuffer qkm(dfb::qkm);  // this block's Qk q/k conv tiles, to the writer's exchange
#endif
    DataflowBuffer vc(dfb::vc);
    DataflowBuffer qn(dfb::qn);
    DataflowBuffer kn(dfb::kn);
    DataflowBuffer vm(dfb::vm);
    DataflowBuffer kt(dfb::kt);
    DataflowBuffer beta_t(dfb::beta_t);
#if GDN_GATE_SCALARS
    DataflowBuffer gsc(dfb::gsc);  // beta[t] at word t, dec[t] at word T + t (the writer's gate math)
#else
    DataflowBuffer dec(dfb::dec);
#endif
#if !GDN_FAST
    DataflowBuffer ab_in(dfb::ab_in);
    DataflowBuffer rsel(dfb::rsel);
    DataflowBuffer csel(dfb::csel);
    DataflowBuffer dtb_s(dfb::dtb_s);
    DataflowBuffer nea_s(dfb::nea_s);
    DataflowBuffer g1(dfb::g1);
    DataflowBuffer a_s(dfb::a_s);
    DataflowBuffer b_s(dfb::b_s);
#else
    (void)b_idx;
#endif
    DataflowBuffer eb(dfb::eb);
    DataflowBuffer hd(dfb::hd);
#if GDN_FP32_STATE
    DataflowBuffer hd32(dfb::hd32);
#if !GDN_FUSED_DECAY  // FUSED_DECAY has no hn32 DFB (the factory does not allocate it)
    DataflowBuffer hn32(dfb::hn32);
    const uint32_t hn32_id = dfb::hn32;
#endif
#endif
    DataflowBuffer vread(dfb::vread);
    DataflowBuffer delta(dfb::delta);
    DataflowBuffer outer(dfb::outer);
    DataflowBuffer hn(dfb::hn);
    DataflowBuffer gq(dfb::gq);
    DataflowBuffer acc_a(dfb::acc_a);
    DataflowBuffer acc_b(dfb::acc_b);
    DataflowBuffer on(dfb::on);
    DataflowBuffer zs(dfb::zs);
    DataflowBuffer hnew(dfb::hnew);
    DataflowBuffer out(dfb::out);
    DataflowBuffer wout(dfb::wout);
#if GDN_HOIST_INV
    DataflowBuffer qsc(dfb::qsc);  // rsqrt(q stats + eps) * scale, computed once (see l2_scale_once)
#endif
#if GDN_VSPLIT
    DataflowBuffer acc_mine(dfb::acc_mine);
    DataflowBuffer acc_x(dfb::acc_x);
    DataflowBuffer tmpv(dfb::tmpv);  // Vs-tile blocks (vm - vread); `tmp` keeps Kt / Vt blocks (see the header)
#endif
    const uint32_t state_in_id = dfb::state_in;
    const uint32_t hn_id = dfb::hn;
    const uint32_t acc_a_id = dfb::acc_a;
    const uint32_t acc_b_id = dfb::acc_b;
    const uint32_t qc_id = dfb::qc;
    const uint32_t kc_id = dfb::kc;
    const uint32_t vc_id = dfb::vc;

    compute_kernel_hw_startup(dfb::src_in, dfb::state_in, dfb::out);
    scaler.wait_front(1);
    eps_l2.wait_front(1);
    eps_norm.wait_front(1);

    // zs = silu(z) first: it needs only the small early reads, so it overlaps the reader's bulk window / projection /
    // tap reads
    {
        DeviceZoneScopedN("c_silu");
        z_in.wait_front(Vt);
        silu_tiles(dfb::z_in, dfb::zs, zs, Vt);
        z_in.pop_front(Vt);
        zs.wait_front(Vt);
    }
#if !GDN_FAST
    // the T gate pairs (plain mode: one-hot gathers + SFPU sigmoid / softplus / exp); FAST: the reader delivers
    // beta_t[t] / dec[t] as ready fp32 scalar tiles
    dtb_s.wait_front(1);
    nea_s.wait_front(1);
    ab_in.wait_front(1 + AB2);
    csel.wait_front(2);
#pragma GCC unroll 1  // code size: the kernel binaries must fit the 69 KB kernel config buffer
    for (uint32_t t = 0; t < T; ++t) {
        gates_token<AB2>(rsel, g1, a_s, b_s, beta_t, dec, b_idx);
    }
    ab_in.pop_front(1 + AB2);
    csel.pop_front(2);
#endif

    // ---- A. window rebuild, shifted conv operands, 4-tap causal conv + SiLU, placement -- phased so every matmul /
    //      broadcast init and format reconfig happens once per phase and DST carries 4-8 tiles per acquire
    {
        DeviceZoneScopedN("c_wait_src");
        src_in.wait_front(2 * ChA);
        sel.wait_front(3 + K);
    }
    {
        DeviceZoneScopedN("c_A");
        // A1: W_c = WO_W @ win_c + WO_N @ qkv_c -> wc[c] (compute) and wout[c] (writer), bf16 (exact one-hot gathers)
        {
        DeviceZoneScopedN("c_A1");
        wc.reserve_back(ChA);
        wout.reserve_back(ChA);
        pack_reconfig_data_format(dfb::wc);
        reconfig_data_format<SrcOrder::Reverse>(dfb::sel, dfb::src_in);
        matmul_init(dfb::sel, dfb::src_in);
#pragma GCC unroll 1  // code size: the kernel binaries must fit the 69 KB kernel config buffer
        for (uint32_t c0 = 0; c0 < ChA; c0 += 4) {
            const uint32_t n = (ChA - c0) < 4 ? (ChA - c0) : 4;
            tile_regs_acquire();
#pragma GCC unroll 1
            for (uint32_t i = 0; i < n; ++i) {
                matmul_tiles(dfb::sel, dfb::src_in, 0, c0 + i, i);
                matmul_tiles(dfb::sel, dfb::src_in, 1, ChA + c0 + i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
#pragma GCC unroll 1
            for (uint32_t i = 0; i < n; ++i) {
                pack_tile(i, dfb::wc, c0 + i);
                pack_tile(i, dfb::wout, c0 + i);
            }
            tile_regs_release();
        }
        wc.push_back(ChA);
        wout.push_back(ChA);
        src_in.pop_front(2 * ChA);
        wc.wait_front(ChA);
        }
        {
        DeviceZoneScopedN("c_A2");
        // A2: shift[c*K + j] = SH_j @ W_c  (row r0+t = W[t+j]; bf16, exact), one chunk = K DST tiles per acquire
        //     (fp32 DST under the half-DST sync holds 4 tiles: indices >= 4 corrupt the half the packer is draining)
        shift.reserve_back(Chs * K);
        pack_reconfig_data_format(dfb::shift);
        reconfig_data_format<SrcOrder::Reverse>(dfb::sel, dfb::wc);
        matmul_init(dfb::sel, dfb::wc);
#pragma GCC unroll 1
        for (uint32_t c = 0; c < Chs; ++c) {
            const uint32_t hc = head_chunk(c);
            tile_regs_acquire();
#pragma GCC unroll 1
            for (uint32_t j = 0; j < K; ++j) {
                matmul_tiles(dfb::sel, dfb::wc, 2 + j, hc, j);
            }
            tile_regs_commit();
            tile_regs_wait();
#pragma GCC unroll 1
            for (uint32_t j = 0; j < K; ++j) {
                pack_tile(j, dfb::shift, c * K + j);
            }
            tile_regs_release();
        }
        shift.push_back(Chs * K);
        wc.pop_front(ChA);
        shift.wait_front(Chs * K);
        }
        {
        DeviceZoneScopedN("c_A3");
        // A3: cv[c] = silu(sum_j shift_j * tap_j) in gdn_decode_step_conv.cpp's causal_conv_silu_packed op order
        //     (DST_j = shift_j * tap_j, DST_0 += DST_1..3 on the SFPU, silu), one chunk per acquire (4 DST tiles)
        taps.wait_front(ChA);
        cv.reserve_back(Chs);
        pack_reconfig_data_format(dfb::cv);
        reconfig_data_format(dfb::shift, dfb::taps);
        mul_bcast_rows_init(dfb::shift, dfb::taps);
#pragma GCC unroll 1
        for (uint32_t c = 0; c < Chs; ++c) {
            const uint32_t hc = head_chunk(c);
            tile_regs_acquire();
#pragma GCC unroll 1
            for (uint32_t j = 0; j < K; ++j) {
                mul_tiles_bcast_rows(dfb::shift, dfb::taps, c * K + j, hc, j, j);
            }
            // SFPU ops with literal DST indices: loop-variable indices trip the TT GCC SFPU synth pass (ICE in
            // rvtt_synth_renumber) in this body
            add_binary_tile_init();
#if GDN_A3_HALF
            if (a3_half) {
                static_assert(K == 4, "A3_HALF: the K = 4 tap sum");
#ifdef TRISC_MATH  // the face routines exist on the math thread only (both branches are no-ops elsewhere)
                sfpu_half(
                    ckernel::sfpu::calculate_sfpu_binary<
                        APPROX,
                        ckernel::BinaryOp::ADD,
                        8,
                        DST_ACCUM_MODE,
                        ckernel::DstRoundingMode::Default>,
                    0,
                    a3_bottom,
                    0u,
                    1u,
                    0u);
                sfpu_half(
                    ckernel::sfpu::calculate_sfpu_binary<
                        APPROX,
                        ckernel::BinaryOp::ADD,
                        8,
                        DST_ACCUM_MODE,
                        ckernel::DstRoundingMode::Default>,
                    0,
                    a3_bottom,
                    0u,
                    2u,
                    0u);
                sfpu_half(
                    ckernel::sfpu::calculate_sfpu_binary<
                        APPROX,
                        ckernel::BinaryOp::ADD,
                        8,
                        DST_ACCUM_MODE,
                        ckernel::DstRoundingMode::Default>,
                    0,
                    a3_bottom,
                    0u,
                    3u,
                    0u);
#endif
                silu_tile_init();
#ifdef TRISC_MATH
                sfpu_half(ckernel::sfpu::calculate_silu<DST_ACCUM_MODE, 8>, 0, a3_bottom);
#endif
            } else
#endif
            {
            if constexpr (K == 4) {
                add_binary_tile(0, 1, 0);
                add_binary_tile(0, 2, 0);
                add_binary_tile(0, 3, 0);
            } else {
                for (uint32_t j = 1; j < K; ++j) {
                    add_binary_tile(0, j, 0);
                }
            }
            silu_tile_init();
            silu_tile(0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::cv, c);
            tile_regs_release();
        }
        cv.push_back(Chs);
        shift.pop_front(Chs * K);
        taps.pop_front(ChA);
        cv.wait_front(Chs);
        }
        {
        DeviceZoneScopedN("c_A4");
        // A4: qc | kc | vc [c] = P @ cv[c] -- the conv values cross a one-hot matmul as in the plain kernel's
        // scatter_conv
        //     (same operand roles: bf16 one-hot A, fp32 conv B); required for bit-identity with it
#if GDN_QKSHARE
        qkm.reserve_back(Qk);
        vc.reserve_back(Vs);
        pack_reconfig_data_format(vc_id);  // qkm / vc share the fp32 format
#else
        qc.reserve_back(Kt);
        kc.reserve_back(Kt);
        vc.reserve_back(Vs);
        pack_reconfig_data_format(qc_id);
#endif
        reconfig_data_format<SrcOrder::Reverse>(dfb::sel, dfb::cv);
        matmul_init(dfb::sel, dfb::cv);
#pragma GCC unroll 1  // code size: the kernel binaries must fit the 69 KB kernel config buffer
        for (uint32_t c0 = 0; c0 < Chs; c0 += 4) {
            const uint32_t n = (Chs - c0) < 4 ? (Chs - c0) : 4;
            tile_regs_acquire();
#pragma GCC unroll 1
            for (uint32_t i = 0; i < n; ++i) {
                matmul_tiles(dfb::sel, dfb::cv, 2 + K, c0 + i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
#pragma GCC unroll 1
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t c = c0 + i;
#if GDN_QKSHARE
                const uint32_t dst_id = (c < Qk) ? static_cast<uint32_t>(dfb::qkm) : vc_id;
                const uint32_t dst_i = (c < Qk) ? c : (c - Qk);
#else
                const uint32_t dst_id = (c < Kt) ? qc_id : ((c < 2 * Kt) ? kc_id : vc_id);
                const uint32_t dst_i = (c < Kt) ? c : ((c < 2 * Kt) ? (c - Kt) : (c - 2 * Kt));
#endif
                pack_tile(i, dst_id, dst_i);
            }
            tile_regs_release();
        }
#if GDN_QKSHARE
        qkm.push_back(Qk);  // the writer swaps it into every block's qc / kc (B waits for them below)
#else
        qc.push_back(Kt);
        kc.push_back(Kt);
#endif
        vc.push_back(Vs);
        cv.pop_front(Chs);
        sel.pop_front(3 + K);
        }
    }

    // ---- B. row-batched pre for the user's T rows
    {
        DeviceZoneScopedN("c_B");
        mask_T.wait_front(1);
        qc.wait_front(Kt);
        kc.wait_front(Kt);
        vc.wait_front(Vs);
        // k, v row-batched; then the q row sums of squares once into `stats` (kept through the token loop): qn_t itself
        // is formed per token with the e_t mask, exactly the plain kernel's inverse_l2(mask) + scale_rows, so o_t =
        // qn_t @ S_t has only the token's row non-zero
        l2norm_rows<Kt>(dfb::kc, dfb::mask_T, dfb::kn, kn, tmp, stats, scratch, inv, kOneBits);
        scale_rows(dfb::vc, dfb::mask_T, dfb::vm, vm, Vs);
        square_tiles(dfb::qc, dfb::tmp, tmp, Kt);
        reduce_rows(Kt);
        stats.wait_front(1);
#if GDN_HOIST_INV
        l2_scale_once(dfb::stats, dfb::eps_l2, dfb::qsc, qsc, scale_bits);
        qsc.wait_front(1);
#endif
        kc.pop_front(Kt);
        vc.pop_front(Vs);
        mask_T.pop_front(1);
        kn.wait_front(Kt);
        vm.wait_front(Vs);
        transpose_tiles(dfb::kn, dfb::kt, kt, Kt);
        kt.wait_front(Kt);
    }

    // ---- C. T serial delta-rule steps, state in L1
    {
        DeviceZoneScopedN("c_wait_state");
        state_in.wait_front(KVs);
    }
#if GDN_GATE_SCALARS
    gsc.wait_front(1);  // the writer's gate bits (computed while phase A ran); popped after the token loop
#endif
#if GDN_FUSED_DECAY
    // token 0's decayed state from state_in (decay_state_sfpu as below); every later token's is formed by its
    // predecessor's state add (outer_add_decay_sfpu)
    {
        DeviceZoneScopedN("c_decay0");
#if GDN_GATE_SCALARS
        decay_state_sfpu(state_in_id, 0, hd, hd32, KVs, gsc.read_tile_value<uint32_t>(0, gsc_dec(0)));
#else
        dec.wait_front(1);
        decay_state_sfpu(state_in_id, dfb::dec, hd, hd32, KVs, 0);
        dec.pop_front(1);
#endif
        state_in.pop_front(KVs);
    }
#endif
#pragma GCC unroll 1  // code size: the kernel binaries must fit the 69 KB kernel config buffer
    for (uint32_t t = 0; t < T; ++t) {
        const bool first = (t == 0);
        const bool last = (t + 1 == T);
        {
            DeviceZoneScopedN("c_wait_gate");
            e_t.wait_front(1);
#if GDN_GATE_SCALARS
            // beta / dec come from gsc (already waited on)
#elif GDN_FUSED_DECAY
            if (!last) {
                dec.wait_front(1);  // dec[t + 1]: this token's state add forms the successor's decayed state
            }
#else
            dec.wait_front(1);
#endif
#if !GDN_GATE_SCALARS
            beta_t.wait_front(1);
#endif
        }
        DeviceZoneScopedN("c_tok");
#if GDN_GATE_SCALARS
        {
            // beta_t[t] = the all-equal fp32 tile of the writer's beta bits (bitcast fill in DST, exact fp32 pack)
            const uint32_t beta_bits = gsc.read_tile_value<uint32_t>(0, gsc_beta(t));
            beta_t.reserve_back(1);
            pack_reconfig_data_format(dfb::beta_t);
            tile_regs_acquire();
            fill_tile_init();
            fill_tile_bitcast(0, beta_bits);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::beta_t);
            tile_regs_release();
            beta_t.push_back(1);
            beta_t.wait_front(1);
        }
#endif
        // eb = e_t * beta_t : column tile carrying beta at row r0 + t (0 elsewhere)
        scale_rows(dfb::e_t, dfb::beta_t, dfb::eb, eb, 1);  // beta_t is all-equal: column 0 broadcast == elementwise
        beta_t.pop_front(1);
        eb.wait_front(1);
        // qn_t = l2norm(qc) * scale on row r0 + t only (inverse_l2 with the e_t mask; other rows exactly 0)
#if GDN_HOIST_INV
        l2_mask(dfb::qsc, dfb::e_t, dfb::inv, inv);
#else
        inverse_l2(dfb::stats, dfb::eps_l2, dfb::e_t, dfb::scratch, scratch, dfb::inv, inv, scale_bits);
#endif
        inv.wait_front(1);
        scale_rows(dfb::qc, dfb::inv, dfb::qn, qn, Kt);
        inv.pop_front(1);
        qn.wait_front(Kt);
#if GDN_FUSED_DECAY
        hd.wait_front(KVs);  // hd / hd32 of this token: c_decay0 (t = 0) or the previous token's state add
#elif GDN_FP32_STATE
        // S_cur in exact fp32 (UnpackToDest): state_in at t = 0, else the hn32 produced at t - 1
        const uint32_t cur32_id = first ? state_in_id : hn32_id;
        DataflowBuffer& cur32 = first ? state_in : hn32;
        if (!first) {
            hn32.wait_front(KVs);
        }
        // hd (FPU operand of vread) = hd32 (SFPU operand of the add) = S_cur * dec, exact fp32 on the SFPU
#if GDN_GATE_SCALARS
        decay_state_sfpu(cur32_id, 0, hd, hd32, KVs, gsc.read_tile_value<uint32_t>(0, gsc_dec(t)));
#else
        decay_state_sfpu(cur32_id, dfb::dec, hd, hd32, KVs, 0);
        dec.pop_front(1);
#endif
        cur32.pop_front(KVs);
        hd.wait_front(KVs);
#else
        // S_cur = state_in at t = 0, else the hn produced at t - 1
        const uint32_t cur_id = first ? state_in_id : hn_id;
        DataflowBuffer& cur = first ? state_in : hn;
        // hd = S_cur * dec
        multiply_tiles<true>(cur_id, dfb::dec, dfb::hd, hd, KVs);
        dec.pop_front(1);
        cur.pop_front(KVs);
        hd.wait_front(KVs);
#endif
        // vread = kn @ hd ; delta = beta * (vm - vread) on row r0 + t (other rows exactly 0)
        {
        DeviceZoneScopedN("c_t_vread");
        row_times_matrix_acc(dfb::vread, dfb::kn, dfb::hd, dfb::vread, vread, Kt, Vs, false);
        vread.wait_front(Vs);
#if GDN_VSPLIT
        subtract_tiles(dfb::vm, dfb::vread, dfb::tmpv, tmpv, Vs);
        vread.pop_front(Vs);
        tmpv.wait_front(Vs);
        scale_rows(dfb::tmpv, dfb::eb, dfb::delta, delta, Vs);
        tmpv.pop_front(Vs);
#else
        subtract_tiles(dfb::vm, dfb::vread, dfb::tmp, tmp, Vs);
        vread.pop_front(Vs);
        tmp.wait_front(Vs);
        scale_rows(dfb::tmp, dfb::eb, dfb::delta, delta, Vs);
        tmp.pop_front(Vs);
#endif
        eb.pop_front(1);
        delta.wait_front(Vs);
        }
        {
        DeviceZoneScopedN("c_t_add");
        // S_next = hd + kt @ delta ; hnew for the writer's ring write
#if GDN_FUSED_DECAY
        hd.pop_front(KVs);  // the FPU copy served vread only; the add reads the exact hd32
        hd32.wait_front(KVs);
        // hn (o operand), hnew (writer) = hd32 + kt @ delta; plus (not last) the next token's hd / hd32 = S_next * dec
#if GDN_GATE_SCALARS
        outer_add_decay_sfpu(
            hn, hnew, hd, hd32, Kt, Vs, !last, last ? 0u : gsc.read_tile_value<uint32_t>(0, gsc_dec(t + 1)));
#else
        outer_add_decay_sfpu(hn, hnew, hd, hd32, Kt, Vs, !last, 0);  // pops this token's hd32 pair by pair
        if (!last) {
            dec.pop_front(1);
        }
#endif
        delta.pop_front(Vs);
#elif GDN_FP32_STATE
        hd.pop_front(KVs);  // the FPU copy served vread only; the add reads the exact hd32
        hd32.wait_front(KVs);
        // hn (o operand), hnew (writer), hn32 (next token's S_cur) = hd32 + kt @ delta, outer product fused in DST
        outer_add_state_sfpu(hn, hnew, hn32, Kt, Vs);
        hd32.pop_front(KVs);
        delta.pop_front(Vs);
#else
        outer_product4(dfb::kt, dfb::delta, dfb::outer, outer, Kt, Vs);
        delta.pop_front(Vs);
        outer.wait_front(KVs);
#if GDN_FAST
        add_tiles_n_dual(dfb::hd, dfb::outer, hn_id, hn, dfb::hnew, hnew, KVs);  // hn and hnew from one DST tile
        hd.pop_front(KVs);
        outer.pop_front(KVs);
#else
        add_tiles_n(dfb::hd, dfb::outer, hn_id, hn, KVs);
        hd.pop_front(KVs);
        outer.pop_front(KVs);
        hn.wait_front(KVs);
        copy_tiles(hn_id, dfb::hnew, hnew, KVs);
#endif
#endif
        }
        {
        DeviceZoneScopedN("c_t_o");
        hn.wait_front(KVs);
        // acc[row r0 + t] = qn_t @ S_next: token 0 is the plain kernel's row_times_matrix; later tokens accumulate
        // their row onto the DST-loaded accumulator (other rows unchanged: qn_t is zero there)
        DataflowBuffer& acc_cur = (t & 1u) ? acc_b : acc_a;
        const uint32_t acc_cur_id = (t & 1u) ? acc_b_id : acc_a_id;
        DataflowBuffer& acc_prev = (t & 1u) ? acc_a : acc_b;
        const uint32_t acc_prev_id = (t & 1u) ? acc_a_id : acc_b_id;
        if (!first) {
            acc_prev.wait_front(Vs);
        }
#if GDN_VSPLIT
        if (last) {
            // the last token's accumulator goes to the writer (exact fp32 pack), which assembles the head's full row
            row_times_matrix_acc(acc_prev_id, dfb::qn, hn_id, dfb::acc_mine, acc_mine, Kt, Vs, !first);
        } else {
            row_times_matrix_acc(acc_prev_id, dfb::qn, hn_id, acc_cur_id, acc_cur, Kt, Vs, !first);
        }
#else
        row_times_matrix_acc(acc_prev_id, dfb::qn, hn_id, acc_cur_id, acc_cur, Kt, Vs, !first);
#endif
        if (!first) {
            acc_prev.pop_front(Vs);
        }
#if GDN_FUSED_DECAY
        hn.pop_front(KVs);  // the next token's decayed state was formed from DST by this token's state add
#elif GDN_FP32_STATE
        hn.pop_front(KVs);  // the next token's S_cur is the exact hn32 copy
        if (last) {
            hn32.wait_front(KVs);  // the last token's hn32 has no consumer: drain it
            hn32.pop_front(KVs);
        }
#else
        if (last) {
            hn.pop_front(KVs);  // otherwise it is consumed as S_cur by the next token
        }
#endif
        qn.pop_front(Kt);
        e_t.pop_front(1);
        }
    }
    qc.pop_front(Kt);
    stats.pop_front(1);
#if GDN_GATE_SCALARS
    gsc.pop_front(1);
#endif
#if GDN_HOIST_INV
    qsc.pop_front(1);
#endif

    // ---- D. row-batched post: out = rmsnorm(acc) * w * silu(z)
    DeviceZoneScopedN("c_D");
    w_in.wait_front(Vt);  // only the final gate needs the norm weight (the reader pushes it with the bulk reads)
#if GDN_VS_TAIL
    // VS_TAIL: the row statistics need the head's full pre-norm row; the normalized / gated tiles only this block's
    // Vs columns (rmsnorm_rows' scale_rows and gated_out are per tile: the kept tiles see the same ops on the same
    // operands)
    acc_x.wait_front(Vt);
    square_tiles(dfb::acc_x, dfb::tmp, tmp, Vt);
    reduce_rows(Vt);
    stats.wait_front(1);
    inverse_rms(dfb::stats, dfb::eps_norm, dfb::scratch, scratch, dfb::inv, inv, inv_dv_bits);
    stats.pop_front(1);
    inv.wait_front(1);
    scale_rows_from(dfb::acc_x, v0, dfb::inv, dfb::on, on, Vs);
    inv.pop_front(1);
    acc_x.pop_front(Vt);
    on.wait_front(Vs);
    gated_out_from(dfb::on, dfb::w_in, dfb::zs, v0, dfb::gq, gq, Vs);
    on.pop_front(Vs);
    gq.wait_front(Vs);
    copy_tiles(dfb::gq, dfb::out, out, Vs);
    gq.pop_front(Vs);
#elif GDN_VSPLIT
    // the head's full pre-norm row (every block's last-token tiles, exact bytes), assembled by the writer
    acc_x.wait_front(Vt);
    rmsnorm_rows<Vt>(dfb::acc_x, dfb::on, on, tmp, stats, scratch, inv, inv_dv_bits);
    acc_x.pop_front(Vt);
#else
    DataflowBuffer& acc_fin = ((T - 1) & 1u) ? acc_b : acc_a;
    const uint32_t acc_fin_id = ((T - 1) & 1u) ? acc_b_id : acc_a_id;
    acc_fin.wait_front(Vt);
    rmsnorm_rows<Vt>(acc_fin_id, dfb::on, on, tmp, stats, scratch, inv, inv_dv_bits);
    acc_fin.pop_front(Vt);
#endif
#if !GDN_VS_TAIL
    on.wait_front(Vt);
    // gated -> gq (fp32), then one datacopy pass into `out`: gdn_decode_step_conv.cpp routes its gated result through
    // tmp -> out_acc -> out, i.e. through the SrcA unpack, so its stored output is the tf32-rounded product; the same
    // pass here makes the T = 1 output bit-identical to the plain op's (without it: max |d| 2.1e-3, every element)
    gated_out(dfb::on, dfb::w_in, dfb::zs, dfb::gq, gq, Vt);
    on.pop_front(Vt);
    gq.wait_front(Vt);
    if constexpr (VB == 1) {
        copy_tiles(dfb::gq, dfb::out, out, Vt);
    } else {
        copy_tiles_from(dfb::gq, v0, dfb::out, out, Vs);  // only this block's output columns
    }
    gq.pop_front(Vt);
#endif
    // (qn is pushed and popped Kt per token inside the token loop: nothing left to pop here)
    kn.pop_front(Kt);
    vm.pop_front(Vs);
    kt.pop_front(Kt);
    zs.pop_front(Vt);
    w_in.pop_front(Vt);
    scaler.pop_front(1);
    eps_l2.pop_front(1);
    eps_norm.pop_front(1);
#if !GDN_FAST
    dtb_s.pop_front(1);
    nea_s.pop_front(1);
#endif
}
