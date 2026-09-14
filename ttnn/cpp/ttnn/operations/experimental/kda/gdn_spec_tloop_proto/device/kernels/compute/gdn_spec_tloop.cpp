// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// M0a scratch compute: one core = (user, value head), T candidate tokens in rows r0 = u*T .. r0+T-1 of the head's
// 32-row q/k/v tiles. Pre-processing (l2norms, value mask, kt = kn^T, gates from the a|b tile, silu(z)) either once
// for the T rows (row_batched) or per token; then T serial delta-rule steps with the state resident in L1
// (state_in -> hn_a -> hn_b -> ...), the post-token state copied to `hnew` for the writer's ring write, the token's
// output row accumulated through the e_t row mask; then the gated RMSNorm once (row_batched) or per token.
// Per-token arithmetic is gdn_decode_step_conv.cpp's (same helpers, same op order), so a T-step run equals T chained
// T = 1 runs by construction.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/compute/gdn_step_helpers.hpp"

using namespace gdn_step;

namespace {

constexpr uint32_t kOneBits = 0x3F800000u;     // 1.0f
constexpr uint32_t kTwentyBits = 0x41A00000u;  // 20.0f (softplus threshold, as ttnn.softplus(1.0, 20.0))

// out = a[ai] @ b[bi]  (single-tile matmul; the exact one-hot row / column extraction of the gate scalars)
inline void mm1(uint32_t a, uint32_t b, uint32_t ai, uint32_t bi, uint32_t out, DataflowBuffer& out_dfb) {
    out_dfb.reserve_back(1);
    pack_reconfig_data_format(out);
    reconfig_data_format<SrcOrder::Reverse>(a, b);
    matmul_init(a, b);
    tile_regs_acquire();
    matmul_tiles(a, b, ai, bi, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out, 0);
    tile_regs_release();
    out_dfb.push_back(1);
}

// beta_t = sigmoid(b_s)  (all-equal scalar tile)  -- gdn_decode_step_conv.cpp
inline void gate_beta(DataflowBuffer& beta_t) {
    beta_t.reserve_back(1);
    pack_reconfig_data_format(dfb::beta_t);
    reconfig_data_format_srca(dfb::b_s);
    copy_init(dfb::b_s);
    tile_regs_acquire();
    copy_tile(dfb::b_s, 0, 0);
    sigmoid_tile_init();
    sigmoid_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::beta_t, 0);
    tile_regs_release();
    beta_t.push_back(1);
}

// dec = exp(neg_exp_A * softplus(a + dt_bias))  (all-equal scalar tile)  -- gdn_decode_step_conv.cpp
inline void gate_decay(DataflowBuffer& dec) {
    dec.reserve_back(1);
    pack_reconfig_data_format(dfb::dec);
    reconfig_data_format_srca(dfb::a_s);
    copy_init(dfb::a_s);
    tile_regs_acquire();
    copy_tile(dfb::a_s, 0, 0);
    copy_tile(dfb::dtb_s, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    softplus_tile_init();
    softplus_tile(0, kOneBits, kOneBits, kTwentyBits);
    copy_init(dfb::nea_s);
    copy_tile(dfb::nea_s, 0, 2);
    mul_binary_tile_init();
    mul_binary_tile(0, 2, 0);
    exp_tile_init<false>();
    exp_tile<false>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::dec, 0);
    tile_regs_release();
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

// out1[i] = out2[i] = a[i] + b[i]: one FPU add, the DST tile packed twice (S_next for the next token AND hnew for the
// writer), replacing add_tiles_n + copy_tiles (16 fewer tile ops per token)
inline void add_tiles_dual(
    uint32_t a, uint32_t b, uint32_t out1, DataflowBuffer& o1, uint32_t out2, DataflowBuffer& o2, uint32_t n) {
    o1.reserve_back(n);
    o2.reserve_back(n);
    pack_reconfig_data_format(out1);  // out1 and out2 share the fp32 format
    reconfig_data_format(a, b);
    add_init(a, b);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        add_tiles(a, b, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out1, i);
        pack_tile(0, out2, i);
        tile_regs_release();
    }
    o1.push_back(n);
    o2.push_back(n);
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
    compute_kernel_lib::
        reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, dfb::tmp, dfb::scaler, dfb::stats>(
            compute_kernel_lib::ReduceInputBlockShape::of(1, N));
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
    compute_kernel_lib::
        reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, dfb::tmp, dfb::scaler, dfb::stats>(
            compute_kernel_lib::ReduceInputBlockShape::of(1, N));
    stats.wait_front(1);
    inverse_rms(dfb::stats, dfb::eps_norm, dfb::scratch, scratch, dfb::inv, inv, inv_n_bits);
    stats.pop_front(1);
    inv.wait_front(1);
    scale_rows(src, dfb::inv, dst, dst_dfb, N);
    inv.pop_front(1);
}

// gates of one token: a = ab[r0+t, h], b = ab[r0+t, Nv+h] extracted exactly with the bf16 one-hot selectors
// (g1 = rsel[t] @ ab: every row = row r0+t; a_s = g1 @ csel[0], b_s = g1 @ csel[1]: all-equal scalar tiles)
inline void gates_token(
    DataflowBuffer& rsel,
    DataflowBuffer& g1,
    DataflowBuffer& a_s,
    DataflowBuffer& b_s,
    DataflowBuffer& beta_t,
    DataflowBuffer& dec) {
    rsel.wait_front(1);
    mm1(dfb::rsel, dfb::ab_in, 0, 0, dfb::g1, g1);
    g1.wait_front(1);
    mm1(dfb::g1, dfb::csel, 0, 0, dfb::a_s, a_s);
    mm1(dfb::g1, dfb::csel, 0, 1, dfb::b_s, b_s);
    g1.pop_front(1);
    rsel.pop_front(1);
    b_s.wait_front(1);
    gate_beta(beta_t);
    b_s.pop_front(1);
    a_s.wait_front(1);
    gate_decay(dec);
    a_s.pop_front(1);
}

}  // namespace

template <
    uint32_t Kt,
    uint32_t Vt,
    uint32_t T,
    uint32_t row_batched,
    uint32_t opt_flags,
    uint32_t scale_bits,
    uint32_t inv_dv_bits>
TT_KERNEL void compute(uint32_t u) {
    (void)u;
    constexpr uint32_t KV = Kt * Vt;
    constexpr bool dual_pack = (opt_flags & 2u) != 0;
    constexpr bool writer_only = (opt_flags & 4u) != 0;
    DataflowBuffer q_in(dfb::q_in);
    DataflowBuffer k_in(dfb::k_in);
    DataflowBuffer v_in(dfb::v_in);
    DataflowBuffer z_in(dfb::z_in);
    DataflowBuffer ab_in(dfb::ab_in);
    DataflowBuffer mask_T(dfb::mask_T);
    DataflowBuffer e_t(dfb::e_t);
    DataflowBuffer rsel(dfb::rsel);
    DataflowBuffer csel(dfb::csel);
    DataflowBuffer state_in(dfb::state_in);
    DataflowBuffer w_in(dfb::w_in);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer eps_l2(dfb::eps_l2);
    DataflowBuffer eps_norm(dfb::eps_norm);
    DataflowBuffer dtb_s(dfb::dtb_s);
    DataflowBuffer nea_s(dfb::nea_s);
    DataflowBuffer tmp(dfb::tmp);
    DataflowBuffer stats(dfb::stats);
    DataflowBuffer scratch(dfb::scratch);
    DataflowBuffer inv(dfb::inv);
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
    DataflowBuffer hn_a(dfb::hn_a);
    DataflowBuffer hn_b(dfb::hn_b);
    DataflowBuffer o(dfb::o);
    DataflowBuffer on(dfb::on);
    DataflowBuffer gq(dfb::gq);
    DataflowBuffer acc_a(dfb::acc_a);
    DataflowBuffer acc_b(dfb::acc_b);
    DataflowBuffer zs(dfb::zs);
    DataflowBuffer hnew(dfb::hnew);
    DataflowBuffer out(dfb::out);
    const uint32_t state_in_id = dfb::state_in;
    const uint32_t hn_a_id = dfb::hn_a;
    const uint32_t hn_b_id = dfb::hn_b;
    const uint32_t acc_a_id = dfb::acc_a;
    const uint32_t acc_b_id = dfb::acc_b;

    compute_kernel_hw_startup(dfb::q_in, dfb::state_in, dfb::out);
    scaler.wait_front(1);
    eps_l2.wait_front(1);
    eps_norm.wait_front(1);
    w_in.wait_front(Vt);
    dtb_s.wait_front(1);
    nea_s.wait_front(1);
    q_in.wait_front(Kt);
    k_in.wait_front(Kt);
    v_in.wait_front(Vt);
    z_in.wait_front(Vt);
    ab_in.wait_front(1);
    csel.wait_front(2);
    state_in.wait_front(KV);

    if constexpr (writer_only) {
        // pure ring-write stream measurement: hnew <- state_in, T times; out <- v (bf16 copy); consume everything else
        for (uint32_t t = 0; t < T; ++t) {
            copy_tiles(dfb::state_in, dfb::hnew, hnew, KV);
        }
        copy_tiles(dfb::v_in, dfb::out, out, Vt);
        state_in.pop_front(KV);
        q_in.pop_front(Kt);
        k_in.pop_front(Kt);
        v_in.pop_front(Vt);
        z_in.pop_front(Vt);
        ab_in.pop_front(1);
        csel.pop_front(2);
        if constexpr (row_batched != 0) {
            mask_T.wait_front(1);
            mask_T.pop_front(1);
        }
        rsel.wait_front(T);
        rsel.pop_front(T);
        e_t.wait_front(T);
        e_t.pop_front(T);
        return;
    }

    // zs = silu(z) for all rows, once
    silu_tiles(dfb::z_in, dfb::zs, zs, Vt);
    z_in.pop_front(Vt);
    zs.wait_front(Vt);

    if constexpr (row_batched != 0) {
        // pre, once for the user's T rows: qn = l2norm(q)*scale, kn = l2norm(k), vm = v masked, kt = kn^T, T gate pairs
        mask_T.wait_front(1);
        l2norm_rows<Kt>(dfb::q_in, dfb::mask_T, dfb::qn, qn, tmp, stats, scratch, inv, scale_bits);
        l2norm_rows<Kt>(dfb::k_in, dfb::mask_T, dfb::kn, kn, tmp, stats, scratch, inv, kOneBits);
        scale_rows(dfb::v_in, dfb::mask_T, dfb::vm, vm, Vt);
        q_in.pop_front(Kt);
        k_in.pop_front(Kt);
        v_in.pop_front(Vt);
        mask_T.pop_front(1);
        qn.wait_front(Kt);
        kn.wait_front(Kt);
        vm.wait_front(Vt);
        transpose_tiles(dfb::kn, dfb::kt, kt, Kt);
        kt.wait_front(Kt);
        for (uint32_t t = 0; t < T; ++t) {
            gates_token(rsel, g1, a_s, b_s, beta_t, dec);
        }
    }

    for (uint32_t t = 0; t < T; ++t) {
        const bool first = (t == 0);
        const bool last = (t + 1 == T);
        e_t.wait_front(1);
        if constexpr (row_batched == 0) {
            // per-token pre (today's chain shape): masks to the token's row through e_t
            l2norm_rows<Kt>(dfb::q_in, dfb::e_t, dfb::qn, qn, tmp, stats, scratch, inv, scale_bits);
            l2norm_rows<Kt>(dfb::k_in, dfb::e_t, dfb::kn, kn, tmp, stats, scratch, inv, kOneBits);
            scale_rows(dfb::v_in, dfb::e_t, dfb::vm, vm, Vt);
            qn.wait_front(Kt);
            kn.wait_front(Kt);
            vm.wait_front(Vt);
            transpose_tiles(dfb::kn, dfb::kt, kt, Kt);
            kt.wait_front(Kt);
            gates_token(rsel, g1, a_s, b_s, beta_t, dec);
        }
        dec.wait_front(1);
        beta_t.wait_front(1);
        // eb = e_t * beta_t : column tile carrying beta at row r0 + t (0 elsewhere)
        multiply_tiles<true>(dfb::e_t, dfb::beta_t, dfb::eb, eb, 1);
        beta_t.pop_front(1);
        eb.wait_front(1);

        // state ping-pong: S_cur = state_in at t = 0, else the hn produced at t - 1; S_next -> hn_a / hn_b
        const uint32_t cur_id = first ? state_in_id : ((t & 1u) ? hn_a_id : hn_b_id);
        DataflowBuffer& cur = first ? state_in : ((t & 1u) ? hn_a : hn_b);
        const uint32_t nxt_id = (t & 1u) ? hn_b_id : hn_a_id;
        DataflowBuffer& nxt = (t & 1u) ? hn_b : hn_a;

        // hd = S_cur * dec
        multiply_tiles<true>(cur_id, dfb::dec, dfb::hd, hd, KV);
        dec.pop_front(1);
        cur.pop_front(KV);
        hd.wait_front(KV);
        // vread = kn @ hd ; delta = beta * (vm - vread) on row r0 + t (other rows exactly 0)
        row_times_matrix(dfb::kn, dfb::hd, dfb::vread, vread, Kt, Vt);
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
        if constexpr (dual_pack) {
            add_tiles_dual(dfb::hd, dfb::outer, nxt_id, nxt, dfb::hnew, hnew, KV);
            hd.pop_front(KV);
            outer.pop_front(KV);
            nxt.wait_front(KV);
        } else {
            add_tiles_n(dfb::hd, dfb::outer, nxt_id, nxt, KV);
            hd.pop_front(KV);
            outer.pop_front(KV);
            nxt.wait_front(KV);
            copy_tiles(nxt_id, dfb::hnew, hnew, KV);
        }
        // o = qn @ S_next  (row r0 + t is the token's output row)
        row_times_matrix(dfb::qn, nxt_id, dfb::o, o, Kt, Vt);
        if (last) {
            nxt.pop_front(KV);  // otherwise it is consumed as S_cur by the next token
        }
        o.wait_front(Vt);

        DataflowBuffer& acc_cur = (t & 1u) ? acc_b : acc_a;
        const uint32_t acc_cur_id = (t & 1u) ? acc_b_id : acc_a_id;
        DataflowBuffer& acc_prev = (t & 1u) ? acc_a : acc_b;
        const uint32_t acc_prev_id = (t & 1u) ? acc_a_id : acc_b_id;
        if constexpr (row_batched != 0) {
            // acc += o * e_t  (exact: the token's row lands in its own row of the accumulator)
            if (first) {
                scale_rows(dfb::o, dfb::e_t, acc_cur_id, acc_cur, Vt);
            } else {
                scale_rows(dfb::o, dfb::e_t, dfb::gq, gq, Vt);
                gq.wait_front(Vt);
                acc_prev.wait_front(Vt);
                add_tiles_n(acc_prev_id, dfb::gq, acc_cur_id, acc_cur, Vt);
                acc_prev.pop_front(Vt);
                gq.pop_front(Vt);
            }
            o.pop_front(Vt);
        } else {
            // per-token post: gated rmsnorm of the token's row (o has only that row non-zero), accumulated
            rmsnorm_rows<Vt>(dfb::o, dfb::on, on, tmp, stats, scratch, inv, inv_dv_bits);
            o.pop_front(Vt);
            on.wait_front(Vt);
            gated_out(dfb::on, dfb::w_in, dfb::zs, dfb::gq, gq, Vt);
            on.pop_front(Vt);
            gq.wait_front(Vt);
            if (first) {
                copy_tiles(dfb::gq, acc_cur_id, acc_cur, Vt);
            } else {
                acc_prev.wait_front(Vt);
                add_tiles_n(acc_prev_id, dfb::gq, acc_cur_id, acc_cur, Vt);
                acc_prev.pop_front(Vt);
            }
            gq.pop_front(Vt);
            qn.pop_front(Kt);
            kn.pop_front(Kt);
            vm.pop_front(Vt);
            kt.pop_front(Kt);
        }
        e_t.pop_front(1);
    }

    // post
    DataflowBuffer& acc_fin = ((T - 1) & 1u) ? acc_b : acc_a;
    const uint32_t acc_fin_id = ((T - 1) & 1u) ? acc_b_id : acc_a_id;
    acc_fin.wait_front(Vt);
    if constexpr (row_batched != 0) {
        // out = rmsnorm(acc) * w * silu(z), once for the T rows (rows outside the user's stay exactly 0)
        rmsnorm_rows<Vt>(acc_fin_id, dfb::on, on, tmp, stats, scratch, inv, inv_dv_bits);
        acc_fin.pop_front(Vt);
        on.wait_front(Vt);
        gated_out(dfb::on, dfb::w_in, dfb::zs, dfb::out, out, Vt);
        on.pop_front(Vt);
        qn.pop_front(Kt);
        kn.pop_front(Kt);
        vm.pop_front(Vt);
        kt.pop_front(Kt);
    } else {
        copy_tiles(acc_fin_id, dfb::out, out, Vt);
        acc_fin.pop_front(Vt);
        q_in.pop_front(Kt);
        k_in.pop_front(Kt);
        v_in.pop_front(Vt);
    }
    zs.pop_front(Vt);
    ab_in.pop_front(1);
    csel.pop_front(2);
}
