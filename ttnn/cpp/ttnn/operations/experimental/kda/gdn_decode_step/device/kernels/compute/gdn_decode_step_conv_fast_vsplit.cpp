// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Fused-conv variant of the GDN decode step, batched: each core handles one value head for a contiguous group of users
// [u0, u0 + nu) (rows of the projection tile). Per user: 4-tap causal conv + SiLU on the packed history/tap tiles (row
// 2c + parity(b) of a packed tile = channel chunk c), scattered into the row-block layout (row b of tile c) with 0/1
// selector tiles, beta = sigmoid(b) and decay = exp(-exp(A) * softplus(a + dt_bias)) from per-user scalar tiles, the
// recurrence with the row mask e_b, and the gated RMSNorm output (row b valid, other rows exactly 0). The users'
// outputs are summed into one accumulator and written once as the group's row span.
//
// VALUE BLOCKS (VB > 1; a copy of gdn_decode_step_conv_fast.cpp, which stays the VB = 1 program): this core owns the
// head's state column tiles [v0, v0 + Vs), v0 = s*Vs, Vs = Vt / VB (tt-lab's GDN head split). A state column's
// recurrence never reads another column, so the conv / l2norm / gate prefix runs as before (only the q / k chunks and
// this block's Vs v chunks are scattered) and the recurrence runs on KVs = Kt*Vs state tiles (tile (i, j) at i*Vs + j)
// with the same per-tile ops in the same order: every value is the VB = 1 kernel's. Per user, o = qn @ hn is packed
// straight into `acc_mine` (exact fp32, as into `o`); after the last user the writer swaps the users' tiles with the
// head's other blocks into `acc_x` (user ui's full pre-norm row at ui*Vt), and the gated RMSNorm runs per user on the
// full row as before, emitting only this block's Vs output tiles.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/compute/gdn_step_helpers.hpp"

using namespace gdn_step;

namespace {

constexpr uint32_t kOneBits = 0x3F800000u;     // 1.0f
constexpr uint32_t kTwentyBits = 0x41A00000u;  // 20.0f (softplus threshold, as ttnn.softplus(1.0, 20.0))

// conv_p = silu(hist[0]*taps[0] + hist[1]*taps[1] + hist[2]*taps[2] + cur*taps[3])  (one packed fp32 tile)
inline void causal_conv_silu_packed(DataflowBuffer& conv_p) {
    conv_p.reserve_back(1);
    pack_reconfig_data_format(dfb::conv_p);
    reconfig_data_format(dfb::hist, dfb::taps);
    mul_init(dfb::hist, dfb::taps);
    tile_regs_acquire();
    mul_tiles(dfb::hist, dfb::taps, 0, 0, 0);
    mul_tiles(dfb::hist, dfb::taps, 1, 1, 1);
    mul_tiles(dfb::hist, dfb::taps, 2, 2, 2);
    mul_tiles(dfb::cur, dfb::taps, 0, 3, 3);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    add_binary_tile(0, 2, 0);
    add_binary_tile(0, 3, 0);
    silu_tile_init();
    silu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::conv_p, 0);
    tile_regs_release();
    conv_p.push_back(1);
}

// Scatter: tile c (row b = the user's row of conv_p, other rows exactly 0) = sel[c] @ conv_p, packed into qc | kc | vc.
// VSPLIT: block chunk c < Chs = 2*Kt + Vs is head chunk c (q, k) or 2*Kt + v0 + (c - 2*Kt) (this block's v chunks).
template <uint32_t Kt, uint32_t Vs>
inline void scatter_conv(DataflowBuffer& qc, DataflowBuffer& kc, DataflowBuffer& vc, uint32_t v0) {
    constexpr uint32_t Ct = 2 * Kt + Vs;
    qc.reserve_back(Kt);
    kc.reserve_back(Kt);
    vc.reserve_back(Vs);
    pack_reconfig_data_format(dfb::qc);
    reconfig_data_format<SrcOrder::Reverse>(dfb::sel, dfb::conv_p);
    matmul_init(dfb::sel, dfb::conv_p);
    for (uint32_t c0 = 0; c0 < Ct; c0 += 4) {
        const uint32_t n = (Ct - c0) < 4 ? (Ct - c0) : 4;
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t c = c0 + i;
            matmul_tiles(dfb::sel, dfb::conv_p, c < 2 * Kt ? c : c + v0, 0, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t c = c0 + i;
            if (c < Kt) {
                pack_tile(i, dfb::qc, c);
            } else if (c < 2 * Kt) {
                pack_tile(i, dfb::kc, c - Kt);
            } else {
                pack_tile(i, dfb::vc, c - 2 * Kt);
            }
        }
        tile_regs_release();
    }
    qc.push_back(Kt);
    kc.push_back(Kt);
    vc.push_back(Vs);
}

// beta_t = sigmoid(b) (all-equal scalar tile)
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

// dec = exp(neg_exp_A * softplus(a + dt_bias))  (all-equal scalar tile)
inline void gate_decay(DataflowBuffer& dec) {
    dec.reserve_back(1);
    pack_reconfig_data_format(dfb::dec);
    reconfig_data_format_srca(dfb::a_s);  // a_s / dtb_s / nea_s are all fp32 scalar tiles: one copy init
    copy_init(dfb::a_s);
    tile_regs_acquire();
    copy_tile(dfb::a_s, 0, 0);
    copy_tile(dfb::dtb_s, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    softplus_tile_init();
    softplus_tile(0, kOneBits, kOneBits, kTwentyBits);
    copy_init(dfb::nea_s);  // FPU datacopy re-init after the SFPU ops
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

// gated = (on x w per column) * zs, with the final product on the SFPU (fp32); rows other than b stay exactly 0.
// VSPLIT: column tiles [first, first + n) of on / w_in / zs -> tmp[0, n).
inline void gated_user(DataflowBuffer& tmp, uint32_t first, uint32_t n) {
    tmp.reserve_back(n);
    pack_reconfig_data_format(dfb::tmp);
    reconfig_data_format(dfb::on, dfb::w_in);
    for (uint32_t i = 0; i < n; ++i) {
        mul_bcast_rows_init(dfb::on, dfb::w_in);
        tile_regs_acquire();
        mul_tiles_bcast_rows(dfb::on, dfb::w_in, first + i, first + i, 0);
        copy_init(dfb::zs);
        copy_tile(dfb::zs, first + i, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::tmp, i);
        tile_regs_release();
    }
    tmp.push_back(n);
}

// FAST: out[i] = a[i] + b[i], packed into two DFBs (hn for o = qn @ hn, hnew for the writer) from the same DST tile
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

#ifndef GDN_FP32_STATE
#define GDN_FP32_STATE 0
#endif

#if GDN_FP32_STATE  // hd32 DFB is only allocated when the op attribute fp32_state is set
// FP32_STATE: hd = hd32 = h * decay with an exact fp32 SFPU multiply. state_in and b_s are UnpackToDest DFBs, so
// copy_tile lands the full fp32 values in DST. The same DST tile is packed into hd (matmul operand, UnpackToSrc) and
// hd32 (UnpackToDest, for the SFPU state add). Copy init sequence mirrors gate_decay / binary_ng sfpu no_bcast.
inline void decay_state_sfpu(DataflowBuffer& hd_dfb, DataflowBuffer& hd32_dfb, uint32_t n) {
    hd_dfb.reserve_back(n);
    hd32_dfb.reserve_back(n);
    pack_reconfig_data_format(dfb::hd);
    reconfig_data_format_srca(dfb::state_in);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        copy_init(dfb::state_in);
        copy_tile(dfb::state_in, i, 0);
        copy_init(dfb::b_s);  // FPU datacopy re-init after the SFPU ops of the previous iteration
        copy_tile(dfb::b_s, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::hd, i);
        pack_tile(0, dfb::hd32, i);
        tile_regs_release();
    }
    hd_dfb.push_back(n);
    hd32_dfb.push_back(n);
}

// FP32_STATE: hn = hnew = hd32 + outer with an exact fp32 SFPU add (hd32 and outer are UnpackToDest DFBs).
inline void add_state_sfpu(DataflowBuffer& hn_dfb, DataflowBuffer& hnew_dfb, uint32_t n) {
    hn_dfb.reserve_back(n);
    hnew_dfb.reserve_back(n);
    pack_reconfig_data_format(dfb::hn);
    reconfig_data_format_srca(dfb::hd32);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        copy_init(dfb::hd32);
        copy_tile(dfb::hd32, i, 0);
        copy_init(dfb::outer);
        copy_tile(dfb::outer, i, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::hn, i);
        pack_tile(0, dfb::hnew, i);
        tile_regs_release();
    }
    hn_dfb.push_back(n);
    hnew_dfb.push_back(n);
}

#endif  // GDN_FP32_STATE

}  // namespace

template <uint32_t Kt, uint32_t Vt, uint32_t scale_bits, uint32_t inv_dv_bits, uint32_t VB>
TT_KERNEL void compute(uint32_t nu, uint32_t s) {
    static_assert(VB > 1 && Vt % VB == 0, "value blocks must divide Vt");
    constexpr uint32_t Vs = Vt / VB;   // this block's value (state column) tiles
    constexpr uint32_t KVs = Kt * Vs;  // this block's state tiles
    const uint32_t v0 = s * Vs;        // first value tile of this block
    constexpr uint32_t Ct = 2 * Kt + Vt;
    constexpr uint32_t one_bits = kOneBits;
    DataflowBuffer hist(dfb::hist);
    DataflowBuffer taps(dfb::taps);
    DataflowBuffer cur(dfb::cur);
    DataflowBuffer sel(dfb::sel);
    DataflowBuffer conv_p(dfb::conv_p);
    DataflowBuffer z_in(dfb::z_in);
    DataflowBuffer a_s(dfb::a_s);
    DataflowBuffer b_s(dfb::b_s);
    DataflowBuffer dtb_s(dfb::dtb_s);
    DataflowBuffer nea_s(dfb::nea_s);
    DataflowBuffer state_in(dfb::state_in);
    DataflowBuffer w_in(dfb::w_in);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer eps_l2(dfb::eps_l2);
    DataflowBuffer eps_norm(dfb::eps_norm);
    DataflowBuffer mask(dfb::mask);
    DataflowBuffer qc(dfb::qc);
    DataflowBuffer kc(dfb::kc);
    DataflowBuffer vc(dfb::vc);
    DataflowBuffer beta_t(dfb::beta_t);
    DataflowBuffer zs(dfb::zs);
    DataflowBuffer tmp(dfb::tmp);
    DataflowBuffer stats(dfb::stats);
    DataflowBuffer scratch(dfb::scratch);
    DataflowBuffer inv(dfb::inv);
    DataflowBuffer qn(dfb::qn);
    DataflowBuffer kn(dfb::kn);
    DataflowBuffer vm(dfb::vm);
    DataflowBuffer dec(dfb::dec);
    DataflowBuffer hd(dfb::hd);
#if GDN_FP32_STATE
    DataflowBuffer hd32(dfb::hd32);
#endif
    DataflowBuffer vread(dfb::vread);
    DataflowBuffer delta(dfb::delta);
    DataflowBuffer kt(dfb::kt);
    DataflowBuffer outer(dfb::outer);
    DataflowBuffer hn(dfb::hn);
    DataflowBuffer hnew(dfb::hnew);
    DataflowBuffer on(dfb::on);
    DataflowBuffer acc_mine(dfb::acc_mine);
    DataflowBuffer acc_x(dfb::acc_x);
    DataflowBuffer out_acc(dfb::out_acc);
    DataflowBuffer acc2(dfb::acc2);
    DataflowBuffer out(dfb::out);

    compute_kernel_hw_startup(dfb::hist, dfb::state_in, dfb::out);
    scaler.wait_front(1);
    eps_l2.wait_front(1);
    eps_norm.wait_front(1);
    taps.wait_front(4);
    // FAST: zs = silu(z) for all users of this head (rows = users), once per core, while the conv inputs arrive
    z_in.wait_front(Vt);
    silu_tiles(dfb::z_in, dfb::zs, zs, Vt);
    z_in.pop_front(Vt);
    zs.wait_front(Vt);

    for (uint32_t ui = 0; ui < nu; ++ui) {
        sel.wait_front(Ct);  // per-user selectors (row b <- packed row 2c + parity(b))
        mask.wait_front(1);  // per-user row mask e_b
        hist.wait_front(3);
        cur.wait_front(1);
        // FAST: a/b and the state are waited for where they are first used (the reader delivers them last)

        // conv + silu on the packed tile, scattered into qc, kc, vc ; gates
        causal_conv_silu_packed(conv_p);
        hist.pop_front(3);
        cur.pop_front(1);
        conv_p.wait_front(1);
        scatter_conv<Kt, Vs>(qc, kc, vc, v0);
        conv_p.pop_front(1);
        sel.pop_front(Ct);
        qc.wait_front(Kt);
        kc.wait_front(Kt);
        vc.wait_front(Vs);

        // qn = l2norm(q) * scale, kn = l2norm(k)  (rows other than b -> 0 through the mask)
        square_tiles(dfb::qc, dfb::tmp, tmp, Kt);
        compute_kernel_lib::
            reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, dfb::tmp, dfb::scaler, dfb::stats>(
                compute_kernel_lib::ReduceInputBlockShape::of(1, Kt));
        stats.wait_front(1);
        inverse_l2(dfb::stats, dfb::eps_l2, dfb::mask, dfb::scratch, scratch, dfb::inv, inv, scale_bits);
        stats.pop_front(1);
        inv.wait_front(1);
        scale_rows(dfb::qc, dfb::inv, dfb::qn, qn, Kt);
        inv.pop_front(1);
        qc.pop_front(Kt);
        square_tiles(dfb::kc, dfb::tmp, tmp, Kt);
        compute_kernel_lib::
            reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, dfb::tmp, dfb::scaler, dfb::stats>(
                compute_kernel_lib::ReduceInputBlockShape::of(1, Kt));
        stats.wait_front(1);
        inverse_l2(dfb::stats, dfb::eps_l2, dfb::mask, dfb::scratch, scratch, dfb::inv, inv, one_bits);
        stats.pop_front(1);
        inv.wait_front(1);
        scale_rows(dfb::kc, dfb::inv, dfb::kn, kn, Kt);
        inv.pop_front(1);
        kc.pop_front(Kt);
        scale_rows(dfb::vc, dfb::mask, dfb::vm, vm, Vs);  // vm = v masked to row b
        vc.pop_front(Vs);
        mask.pop_front(1);  // the norm below runs after the exchange and does not use the mask

        // FAST2: the reader delivers beta (a_s) and decay (b_s) as ready scalar tiles
        // hd = h * decay
        state_in.wait_front(KVs);
        b_s.wait_front(1);
#if GDN_FP32_STATE
        decay_state_sfpu(hd, hd32, KVs);
#else
        multiply_tiles<true>(dfb::state_in, dfb::b_s, dfb::hd, hd, KVs);
#endif
        b_s.pop_front(1);
        state_in.pop_front(KVs);
        hd.wait_front(KVs);
        kn.wait_front(Kt);

        // vread = kn @ hd ; delta = beta * (vm - vread)
        row_times_matrix(dfb::kn, dfb::hd, dfb::vread, vread, Kt, Vs);
        vread.wait_front(Vs);
        vm.wait_front(Vs);
        subtract_tiles(dfb::vm, dfb::vread, dfb::tmp, tmp, Vs);
        vread.pop_front(Vs);
        vm.pop_front(Vs);
        tmp.wait_front(Vs);
        a_s.wait_front(1);
        multiply_tiles<true>(dfb::tmp, dfb::a_s, dfb::delta, delta, Vs);
        tmp.pop_front(Vs);
        a_s.pop_front(1);
        delta.wait_front(Vs);

        // hn = hd + kn^T @ delta ; hnew (writer copy)
        transpose_tiles(dfb::kn, dfb::kt, kt, Kt);
        kn.pop_front(Kt);
        kt.wait_front(Kt);
        outer_product(dfb::kt, dfb::delta, dfb::outer, outer, Kt, Vs);
        kt.pop_front(Kt);
        delta.pop_front(Vs);
        outer.wait_front(KVs);
#if GDN_FP32_STATE
        hd32.wait_front(KVs);
        add_state_sfpu(hn, hnew, KVs);
        hd32.pop_front(KVs);
#else
        add_tiles_n_dual(dfb::hd, dfb::outer, dfb::hn, hn, dfb::hnew, hnew, KVs);
#endif
        hd.pop_front(KVs);
        outer.pop_front(KVs);
        hn.wait_front(KVs);

        // o = qn @ hn, this block's Vs tiles packed straight into acc_mine for the writer's exchange
        qn.wait_front(Kt);
        row_times_matrix(dfb::qn, dfb::hn, dfb::acc_mine, acc_mine, Kt, Vs);
        qn.pop_front(Kt);
        hn.pop_front(KVs);
    }

    // every user's full pre-norm row (all blocks' tiles, exact bytes), assembled by the writer: user ui at ui*Vt
    w_in.wait_front(Vt);
    acc_x.wait_front(nu * Vt);
    for (uint32_t ui = 0; ui < nu; ++ui) {
        // gated = rmsnorm(o) * w * silu(z) for this user's row; accumulate over the group's users
        square_tiles(dfb::acc_x, dfb::tmp, tmp, Vt);
        compute_kernel_lib::
            reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, dfb::tmp, dfb::scaler, dfb::stats>(
                compute_kernel_lib::ReduceInputBlockShape::of(1, Vt));
        stats.wait_front(1);
        inverse_rms(dfb::stats, dfb::eps_norm, dfb::scratch, scratch, dfb::inv, inv, inv_dv_bits);
        stats.pop_front(1);
        inv.wait_front(1);
        scale_rows(dfb::acc_x, dfb::inv, dfb::on, on, Vt);
        inv.pop_front(1);
        acc_x.pop_front(Vt);
        on.wait_front(Vt);
        gated_user(tmp, v0, Vs);  // only this block's output columns
        on.pop_front(Vt);
        tmp.wait_front(Vs);
        if (ui == 0) {
            copy_tiles(dfb::tmp, dfb::out_acc, out_acc, Vs);
        } else {
            out_acc.wait_front(Vs);
            add_tiles_n(dfb::out_acc, dfb::tmp, dfb::acc2, acc2, Vs);
            out_acc.pop_front(Vs);
            acc2.wait_front(Vs);
            copy_tiles(dfb::acc2, dfb::out_acc, out_acc, Vs);
            acc2.pop_front(Vs);
        }
        tmp.pop_front(Vs);
    }
    // the group's rows of this block's columns go out once
    out_acc.wait_front(Vs);
    copy_tiles(dfb::out_acc, dfb::out, out, Vs);
    out_acc.pop_front(Vs);
    zs.pop_front(Vt);
}
