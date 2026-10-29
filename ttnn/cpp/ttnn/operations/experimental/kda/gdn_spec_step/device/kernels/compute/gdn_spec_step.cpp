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
//    tile, ab_in[b_idx]), zs = silu(z).
// C. T serial delta-rule steps, state resident in L1 (state_in -> hn -> hn ...), post-token state copied to `hnew`
//    for the writer, the token's output row accumulated through the e_t row mask.
// D. row-batched post: out = rmsnorm(acc) * w * zs (rows outside the user's stay exactly 0 in the `out` tiles; the
//    writer emits only the user's T rows, and the last user's cores zero the padding rows of the tile row).
// Per-token arithmetic and helpers are gdn_decode_step_conv.cpp's; a T-step run equals T chained T = 1 runs.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/compute/gdn_step_helpers.hpp"

using namespace gdn_step;

namespace {

// stats = row sums of `tmp` (n tiles); shared by the k l2norm, the q stats (kept through the token loop) and the
// rmsnorm (one DFB, three call sites)
inline void reduce_rows(uint32_t n) {
    compute_kernel_lib::
        reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, dfb::tmp, dfb::scaler, dfb::stats>(
            compute_kernel_lib::ReduceInputBlockShape::of(1, n));
}

constexpr uint32_t kOneBits = 0x3F800000u;     // 1.0f
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

// zs[i] = silu(z[i])
inline void silu_tiles(uint32_t a, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
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

}  // namespace

template <uint32_t Kt, uint32_t Vt, uint32_t T, uint32_t K, uint32_t AB2, uint32_t scale_bits, uint32_t inv_dv_bits>
TT_KERNEL void compute(uint32_t u, uint32_t b_idx) {
    (void)u;
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t Ch = 2 * Kt + Vt;
    DataflowBuffer src_in(dfb::src_in);
    DataflowBuffer taps(dfb::taps);
    DataflowBuffer z_in(dfb::z_in);
    DataflowBuffer ab_in(dfb::ab_in);
    DataflowBuffer sel(dfb::sel);
    DataflowBuffer mask_T(dfb::mask_T);
    DataflowBuffer e_t(dfb::e_t);
    DataflowBuffer rsel(dfb::rsel);
    DataflowBuffer csel(dfb::csel);
    DataflowBuffer w_in(dfb::w_in);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer eps_l2(dfb::eps_l2);
    DataflowBuffer eps_norm(dfb::eps_norm);
    DataflowBuffer dtb_s(dfb::dtb_s);
    DataflowBuffer nea_s(dfb::nea_s);
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
    DataflowBuffer vc(dfb::vc);
    DataflowBuffer qn(dfb::qn);
    DataflowBuffer kn(dfb::kn);
    DataflowBuffer vm(dfb::vm);
    DataflowBuffer kt(dfb::kt);
    DataflowBuffer g1(dfb::g1);
    DataflowBuffer a_s(dfb::a_s);
    DataflowBuffer b_s(dfb::b_s);
    DataflowBuffer beta_t(dfb::beta_t);
    DataflowBuffer dec(dfb::dec);
    DataflowBuffer eb(dfb::eb);
    DataflowBuffer hd(dfb::hd);
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
    w_in.wait_front(Vt);
    dtb_s.wait_front(1);
    nea_s.wait_front(1);

    // zs = silu(z) and the T gate pairs first: they need only the small early reads, so they overlap the reader's
    // bulk window / projection / tap reads
    z_in.wait_front(Vt);
    silu_tiles(dfb::z_in, dfb::zs, zs, Vt);
    z_in.pop_front(Vt);
    zs.wait_front(Vt);
    ab_in.wait_front(1 + AB2);
    csel.wait_front(2);
#pragma GCC unroll 1  // code size: the kernel binaries must fit the 69 KB kernel config buffer
    for (uint32_t t = 0; t < T; ++t) {
        gates_token<AB2>(rsel, g1, a_s, b_s, beta_t, dec, b_idx);
    }
    ab_in.pop_front(1 + AB2);
    csel.pop_front(2);

    // ---- A. window rebuild, shifted conv operands, 4-tap causal conv + SiLU, placement -- phased so every matmul /
    //      broadcast init and format reconfig happens once per phase and DST carries 4-8 tiles per acquire
    src_in.wait_front(2 * Ch);
    sel.wait_front(3 + K);
    // A1: W_c = WO_W @ win_c + WO_N @ qkv_c -> wc[c] (compute) and wout[c] (writer), bf16 (exact one-hot gathers)
    wc.reserve_back(Ch);
    wout.reserve_back(Ch);
    pack_reconfig_data_format(dfb::wc);
    reconfig_data_format<SrcOrder::Reverse>(dfb::sel, dfb::src_in);
    matmul_init(dfb::sel, dfb::src_in);
#pragma GCC unroll 1  // code size: the kernel binaries must fit the 69 KB kernel config buffer
    for (uint32_t c0 = 0; c0 < Ch; c0 += 4) {
        const uint32_t n = (Ch - c0) < 4 ? (Ch - c0) : 4;
        tile_regs_acquire();
#pragma GCC unroll 1
        for (uint32_t i = 0; i < n; ++i) {
            matmul_tiles(dfb::sel, dfb::src_in, 0, c0 + i, i);
            matmul_tiles(dfb::sel, dfb::src_in, 1, Ch + c0 + i, i);
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
    wc.push_back(Ch);
    wout.push_back(Ch);
    src_in.pop_front(2 * Ch);
    wc.wait_front(Ch);
    // A2: shift[c*K + j] = SH_j @ W_c  (row r0+t = W[t+j]; bf16, exact), one chunk = K DST tiles per acquire
    //     (fp32 DST under the half-DST sync holds 4 tiles: indices >= 4 corrupt the half the packer is draining)
    shift.reserve_back(Ch * K);
    pack_reconfig_data_format(dfb::shift);
    reconfig_data_format<SrcOrder::Reverse>(dfb::sel, dfb::wc);
    matmul_init(dfb::sel, dfb::wc);
#pragma GCC unroll 1
    for (uint32_t c = 0; c < Ch; ++c) {
        tile_regs_acquire();
#pragma GCC unroll 1
        for (uint32_t j = 0; j < K; ++j) {
            matmul_tiles(dfb::sel, dfb::wc, 2 + j, c, j);
        }
        tile_regs_commit();
        tile_regs_wait();
#pragma GCC unroll 1
        for (uint32_t j = 0; j < K; ++j) {
            pack_tile(j, dfb::shift, c * K + j);
        }
        tile_regs_release();
    }
    shift.push_back(Ch * K);
    wc.pop_front(Ch);
    shift.wait_front(Ch * K);
    // A3: cv[c] = silu(sum_j shift_j * tap_j) in gdn_decode_step_conv.cpp's causal_conv_silu_packed op order
    //     (DST_j = shift_j * tap_j, DST_0 += DST_1..3 on the SFPU, silu), one chunk per acquire (4 DST tiles)
    taps.wait_front(Ch);
    cv.reserve_back(Ch);
    pack_reconfig_data_format(dfb::cv);
    reconfig_data_format(dfb::shift, dfb::taps);
    mul_bcast_rows_init(dfb::shift, dfb::taps);
#pragma GCC unroll 1
    for (uint32_t c = 0; c < Ch; ++c) {
        tile_regs_acquire();
#pragma GCC unroll 1
        for (uint32_t j = 0; j < K; ++j) {
            mul_tiles_bcast_rows(dfb::shift, dfb::taps, c * K + j, c, j, j);
        }
        // SFPU ops with literal DST indices: loop-variable indices trip the TT GCC SFPU synth pass (ICE in
        // rvtt_synth_renumber) in this body
        add_binary_tile_init();
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
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::cv, c);
        tile_regs_release();
    }
    cv.push_back(Ch);
    shift.pop_front(Ch * K);
    taps.pop_front(Ch);
    cv.wait_front(Ch);
    // A4: qc | kc | vc [c] = P @ cv[c] -- the conv values cross a one-hot matmul as in the plain kernel's scatter_conv
    //     (same operand roles: bf16 one-hot A, fp32 conv B); required for bit-identity with it
    qc.reserve_back(Kt);
    kc.reserve_back(Kt);
    vc.reserve_back(Vt);
    pack_reconfig_data_format(qc_id);
    reconfig_data_format<SrcOrder::Reverse>(dfb::sel, dfb::cv);
    matmul_init(dfb::sel, dfb::cv);
#pragma GCC unroll 1  // code size: the kernel binaries must fit the 69 KB kernel config buffer
    for (uint32_t c0 = 0; c0 < Ch; c0 += 4) {
        const uint32_t n = (Ch - c0) < 4 ? (Ch - c0) : 4;
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
            const uint32_t dst_id = (c < Kt) ? qc_id : ((c < 2 * Kt) ? kc_id : vc_id);
            const uint32_t dst_i = (c < Kt) ? c : ((c < 2 * Kt) ? (c - Kt) : (c - 2 * Kt));
            pack_tile(i, dst_id, dst_i);
        }
        tile_regs_release();
    }
    qc.push_back(Kt);
    kc.push_back(Kt);
    vc.push_back(Vt);
    cv.pop_front(Ch);
    sel.pop_front(3 + K);

    // ---- B. row-batched pre for the user's T rows
    mask_T.wait_front(1);
    qc.wait_front(Kt);
    kc.wait_front(Kt);
    vc.wait_front(Vt);
    // k, v row-batched; then the q row sums of squares once into `stats` (kept through the token loop): qn_t itself is
    // formed per token with the e_t mask, exactly the plain kernel's inverse_l2(mask) + scale_rows, so
    // o_t = qn_t @ S_t has only the token's row non-zero
    l2norm_rows<Kt>(dfb::kc, dfb::mask_T, dfb::kn, kn, tmp, stats, scratch, inv, kOneBits);
    scale_rows(dfb::vc, dfb::mask_T, dfb::vm, vm, Vt);
    square_tiles(dfb::qc, dfb::tmp, tmp, Kt);
    reduce_rows(Kt);
    stats.wait_front(1);
    kc.pop_front(Kt);
    vc.pop_front(Vt);
    mask_T.pop_front(1);
    kn.wait_front(Kt);
    vm.wait_front(Vt);
    transpose_tiles(dfb::kn, dfb::kt, kt, Kt);
    kt.wait_front(Kt);

    // ---- C. T serial delta-rule steps, state in L1
    state_in.wait_front(KV);
#pragma GCC unroll 1  // code size: the kernel binaries must fit the 69 KB kernel config buffer
    for (uint32_t t = 0; t < T; ++t) {
        const bool first = (t == 0);
        const bool last = (t + 1 == T);
        e_t.wait_front(1);
        dec.wait_front(1);
        beta_t.wait_front(1);
        // eb = e_t * beta_t : column tile carrying beta at row r0 + t (0 elsewhere)
        scale_rows(dfb::e_t, dfb::beta_t, dfb::eb, eb, 1);  // beta_t is all-equal: column 0 broadcast == elementwise
        beta_t.pop_front(1);
        eb.wait_front(1);
        // qn_t = l2norm(qc) * scale on row r0 + t only (inverse_l2 with the e_t mask; other rows exactly 0)
        inverse_l2(dfb::stats, dfb::eps_l2, dfb::e_t, dfb::scratch, scratch, dfb::inv, inv, scale_bits);
        inv.wait_front(1);
        scale_rows(dfb::qc, dfb::inv, dfb::qn, qn, Kt);
        inv.pop_front(1);
        qn.wait_front(Kt);
        // S_cur = state_in at t = 0, else the hn produced at t - 1
        const uint32_t cur_id = first ? state_in_id : hn_id;
        DataflowBuffer& cur = first ? state_in : hn;
        // hd = S_cur * dec
        multiply_tiles<true>(cur_id, dfb::dec, dfb::hd, hd, KV);
        dec.pop_front(1);
        cur.pop_front(KV);
        hd.wait_front(KV);
        // vread = kn @ hd ; delta = beta * (vm - vread) on row r0 + t (other rows exactly 0)
        row_times_matrix_acc(dfb::vread, dfb::kn, dfb::hd, dfb::vread, vread, Kt, Vt, false);
        vread.wait_front(Vt);
        subtract_tiles(dfb::vm, dfb::vread, dfb::tmp, tmp, Vt);
        vread.pop_front(Vt);
        tmp.wait_front(Vt);
        scale_rows(dfb::tmp, dfb::eb, dfb::delta, delta, Vt);
        tmp.pop_front(Vt);
        eb.pop_front(1);
        delta.wait_front(Vt);
        // S_next = hd + kt @ delta ; copy to hnew for the writer's ring write
        outer_product(dfb::kt, dfb::delta, dfb::outer, outer, Kt, Vt);
        delta.pop_front(Vt);
        outer.wait_front(KV);
        add_tiles_n(dfb::hd, dfb::outer, hn_id, hn, KV);
        hd.pop_front(KV);
        outer.pop_front(KV);
        hn.wait_front(KV);
        copy_tiles(hn_id, dfb::hnew, hnew, KV);
        // acc[row r0 + t] = qn_t @ S_next: token 0 is the plain kernel's row_times_matrix; later tokens accumulate
        // their row onto the DST-loaded accumulator (other rows unchanged: qn_t is zero there)
        DataflowBuffer& acc_cur = (t & 1u) ? acc_b : acc_a;
        const uint32_t acc_cur_id = (t & 1u) ? acc_b_id : acc_a_id;
        DataflowBuffer& acc_prev = (t & 1u) ? acc_a : acc_b;
        const uint32_t acc_prev_id = (t & 1u) ? acc_a_id : acc_b_id;
        if (!first) {
            acc_prev.wait_front(Vt);
        }
        row_times_matrix_acc(acc_prev_id, dfb::qn, hn_id, acc_cur_id, acc_cur, Kt, Vt, !first);
        if (!first) {
            acc_prev.pop_front(Vt);
        }
        if (last) {
            hn.pop_front(KV);  // otherwise it is consumed as S_cur by the next token
        }
        qn.pop_front(Kt);
        e_t.pop_front(1);
    }
    qc.pop_front(Kt);
    stats.pop_front(1);

    // ---- D. row-batched post: out = rmsnorm(acc) * w * silu(z)
    DataflowBuffer& acc_fin = ((T - 1) & 1u) ? acc_b : acc_a;
    const uint32_t acc_fin_id = ((T - 1) & 1u) ? acc_b_id : acc_a_id;
    acc_fin.wait_front(Vt);
    rmsnorm_rows<Vt>(acc_fin_id, dfb::on, on, tmp, stats, scratch, inv, inv_dv_bits);
    acc_fin.pop_front(Vt);
    on.wait_front(Vt);
    // gated -> gq (fp32), then one datacopy pass into `out`: gdn_decode_step_conv.cpp routes its gated result through
    // tmp -> out_acc -> out, i.e. through the SrcA unpack, so its stored output is the tf32-rounded product; the same
    // pass here makes the T = 1 output bit-identical to the plain op's (without it: max |d| 2.1e-3, every element)
    gated_out(dfb::on, dfb::w_in, dfb::zs, dfb::gq, gq, Vt);
    on.pop_front(Vt);
    gq.wait_front(Vt);
    copy_tiles(dfb::gq, dfb::out, out, Vt);
    gq.pop_front(Vt);
    // (qn is pushed and popped Kt per token inside the token loop: nothing left to pop here)
    kn.pop_front(Kt);
    vm.pop_front(Vt);
    kt.pop_front(Kt);
    zs.pop_front(Vt);
    w_in.pop_front(Vt);
    scaler.pop_front(1);
    eps_l2.pop_front(1);
    eps_norm.pop_front(1);
    dtb_s.pop_front(1);
    nea_s.pop_front(1);
}
