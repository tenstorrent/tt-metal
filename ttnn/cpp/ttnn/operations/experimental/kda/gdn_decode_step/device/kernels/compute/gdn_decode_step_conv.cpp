// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Fused-conv variant of the GDN decode step, batched: each core handles one value head for a contiguous group of users
// [u0, u0 + nu) (rows of the projection tile). Per user: 4-tap causal conv + SiLU on the packed history/tap tiles (row
// 2c + parity(b) of a packed tile = channel chunk c), scattered into the row-block layout (row b of tile c) with 0/1
// selector tiles, beta = sigmoid(b) and decay = exp(-exp(A) * softplus(a + dt_bias)) from per-user scalar tiles, the
// recurrence with the row mask e_b, and the gated RMSNorm output (row b valid, other rows exactly 0). The users'
// outputs are summed into one accumulator and written once as the group's row span.
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
template <uint32_t Kt, uint32_t Vt>
inline void scatter_conv(DataflowBuffer& qc, DataflowBuffer& kc, DataflowBuffer& vc) {
    constexpr uint32_t Ct = 2 * Kt + Vt;
    qc.reserve_back(Kt);
    kc.reserve_back(Kt);
    vc.reserve_back(Vt);
    pack_reconfig_data_format(dfb::qc);
    reconfig_data_format<SrcOrder::Reverse>(dfb::sel, dfb::conv_p);
    matmul_init(dfb::sel, dfb::conv_p);
    for (uint32_t c0 = 0; c0 < Ct; c0 += 4) {
        const uint32_t n = (Ct - c0) < 4 ? (Ct - c0) : 4;
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            matmul_tiles(dfb::sel, dfb::conv_p, c0 + i, 0, i);
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
    vc.push_back(Vt);
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
inline void gated_user(DataflowBuffer& tmp, uint32_t n) {
    tmp.reserve_back(n);
    pack_reconfig_data_format(dfb::tmp);
    reconfig_data_format(dfb::on, dfb::w_in);
    for (uint32_t i = 0; i < n; ++i) {
        mul_bcast_rows_init(dfb::on, dfb::w_in);
        tile_regs_acquire();
        mul_tiles_bcast_rows(dfb::on, dfb::w_in, i, i, 0);
        copy_init(dfb::zs);
        copy_tile(dfb::zs, i, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::tmp, i);
        tile_regs_release();
    }
    tmp.push_back(n);
}

}  // namespace

template <uint32_t Kt, uint32_t Vt, uint32_t scale_bits, uint32_t inv_dv_bits>
TT_KERNEL void compute(uint32_t nu) {
    constexpr uint32_t KV = Kt * Vt;
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
    DataflowBuffer vread(dfb::vread);
    DataflowBuffer delta(dfb::delta);
    DataflowBuffer kt(dfb::kt);
    DataflowBuffer outer(dfb::outer);
    DataflowBuffer hn(dfb::hn);
    DataflowBuffer hnew(dfb::hnew);
    DataflowBuffer o(dfb::o);
    DataflowBuffer on(dfb::on);
    DataflowBuffer out_acc(dfb::out_acc);
    DataflowBuffer acc2(dfb::acc2);
    DataflowBuffer out(dfb::out);

    compute_kernel_hw_startup(dfb::hist, dfb::state_in, dfb::out);
    scaler.wait_front(1);
    eps_l2.wait_front(1);
    eps_norm.wait_front(1);
    w_in.wait_front(Vt);
    taps.wait_front(4);
    dtb_s.wait_front(1);
    nea_s.wait_front(1);
    // zs = silu(z) for all users of this head (rows = users), once per core
    z_in.wait_front(Vt);
    silu_tiles(dfb::z_in, dfb::zs, zs, Vt);
    z_in.pop_front(Vt);
    zs.wait_front(Vt);

    for (uint32_t ui = 0; ui < nu; ++ui) {
        sel.wait_front(Ct);  // per-user selectors (row b <- packed row 2c + parity(b))
        mask.wait_front(1);  // per-user row mask e_b
        hist.wait_front(3);
        cur.wait_front(1);
        a_s.wait_front(1);
        b_s.wait_front(1);
        state_in.wait_front(KV);

        // conv + silu on the packed tile, scattered into qc, kc, vc ; gates
        causal_conv_silu_packed(conv_p);
        hist.pop_front(3);
        cur.pop_front(1);
        conv_p.wait_front(1);
        scatter_conv<Kt, Vt>(qc, kc, vc);
        conv_p.pop_front(1);
        sel.pop_front(Ct);
        gate_beta(beta_t);
        b_s.pop_front(1);
        gate_decay(dec);
        a_s.pop_front(1);
        qc.wait_front(Kt);
        kc.wait_front(Kt);
        vc.wait_front(Vt);

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
        scale_rows(dfb::vc, dfb::mask, dfb::vm, vm, Vt);  // vm = v masked to row b
        vc.pop_front(Vt);

        // hd = h * decay
        dec.wait_front(1);
        multiply_tiles<true>(dfb::state_in, dfb::dec, dfb::hd, hd, KV);
        dec.pop_front(1);
        state_in.pop_front(KV);
        hd.wait_front(KV);
        kn.wait_front(Kt);

        // vread = kn @ hd ; delta = beta * (vm - vread)
        row_times_matrix(dfb::kn, dfb::hd, dfb::vread, vread, Kt, Vt);
        vread.wait_front(Vt);
        vm.wait_front(Vt);
        subtract_tiles(dfb::vm, dfb::vread, dfb::tmp, tmp, Vt);
        vread.pop_front(Vt);
        vm.pop_front(Vt);
        tmp.wait_front(Vt);
        beta_t.wait_front(1);
        multiply_tiles<true>(dfb::tmp, dfb::beta_t, dfb::delta, delta, Vt);
        tmp.pop_front(Vt);
        beta_t.pop_front(1);
        delta.wait_front(Vt);

        // hn = hd + kn^T @ delta ; hnew (writer copy)
        transpose_tiles(dfb::kn, dfb::kt, kt, Kt);
        kn.pop_front(Kt);
        kt.wait_front(Kt);
        outer_product(dfb::kt, dfb::delta, dfb::outer, outer, Kt, Vt);
        kt.pop_front(Kt);
        delta.pop_front(Vt);
        outer.wait_front(KV);
        add_tiles_n(dfb::hd, dfb::outer, dfb::hn, hn, KV);
        hd.pop_front(KV);
        outer.pop_front(KV);
        hn.wait_front(KV);
        copy_tiles(dfb::hn, dfb::hnew, hnew, KV);

        // o = qn @ hn
        qn.wait_front(Kt);
        row_times_matrix(dfb::qn, dfb::hn, dfb::o, o, Kt, Vt);
        qn.pop_front(Kt);
        hn.pop_front(KV);
        o.wait_front(Vt);

        // gated = rmsnorm(o) * w * silu(z) for this user's row; accumulate over the group's users
        square_tiles(dfb::o, dfb::tmp, tmp, Vt);
        compute_kernel_lib::
            reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, dfb::tmp, dfb::scaler, dfb::stats>(
                compute_kernel_lib::ReduceInputBlockShape::of(1, Vt));
        stats.wait_front(1);
        inverse_rms(dfb::stats, dfb::eps_norm, dfb::scratch, scratch, dfb::inv, inv, inv_dv_bits);
        stats.pop_front(1);
        inv.wait_front(1);
        scale_rows(dfb::o, dfb::inv, dfb::on, on, Vt);
        inv.pop_front(1);
        o.pop_front(Vt);
        on.wait_front(Vt);
        gated_user(tmp, Vt);
        on.pop_front(Vt);
        mask.pop_front(1);
        tmp.wait_front(Vt);
        if (ui == 0) {
            copy_tiles(dfb::tmp, dfb::out_acc, out_acc, Vt);
        } else {
            out_acc.wait_front(Vt);
            add_tiles_n(dfb::out_acc, dfb::tmp, dfb::acc2, acc2, Vt);
            out_acc.pop_front(Vt);
            acc2.wait_front(Vt);
            copy_tiles(dfb::acc2, dfb::out_acc, out_acc, Vt);
            acc2.pop_front(Vt);
        }
        tmp.pop_front(Vt);
    }
    // the group's rows go out once
    out_acc.wait_front(Vt);
    copy_tiles(dfb::out_acc, dfb::out, out, Vt);
    out_acc.pop_front(Vt);
    zs.pop_front(Vt);
}
