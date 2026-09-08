// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Fused-conv variant of the GDN decode step: additionally computes the 4-tap causal depthwise conv + SiLU from the
// three history rows and the new projection row, beta = sigmoid(b), decay = exp(-exp(A) * softplus(a + dt_bias)) from
// the per-head scalars, and gates the normalized output with silu(z). One core per value head, token in row 0.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/compute/gdn_step_helpers.hpp"

using namespace gdn_step;

namespace {

constexpr uint32_t kOneBits = 0x3F800000u;     // 1.0f
constexpr uint32_t kTwentyBits = 0x41A00000u;  // 20.0f (softplus threshold, as ttnn.softplus(1.0, 20.0))

// conv[t] = silu(hist1[t]*tap0[t] + hist2[t]*tap1[t] + hist3[t]*tap2[t] + cur[t]*tap3[t]) for the Ct = 2Kt+Vt head
// tiles, packed into qc (first Kt), kc (next Kt) and vc (last Vt).
template <uint32_t Kt, uint32_t Vt>
inline void causal_conv_silu(DataflowBuffer& qc, DataflowBuffer& kc, DataflowBuffer& vc) {
    constexpr uint32_t Ct = 2 * Kt + Vt;
    qc.reserve_back(Kt);
    kc.reserve_back(Kt);
    vc.reserve_back(Vt);
    pack_reconfig_data_format(dfb::qc);
    // all four operand pairs are bf16 tiles of the same shape: one unpack/math init serves every product
    reconfig_data_format(dfb::hist1, dfb::tap0);
    for (uint32_t t = 0; t < Ct; ++t) {
        // the previous tile's SFPU ops leave the math in SFPU state: cheap FPU re-init (no format reconfig) per tile
        mul_init(dfb::hist1, dfb::tap0);
        tile_regs_acquire();
        mul_tiles(dfb::hist1, dfb::tap0, t, t, 0);
        mul_tiles(dfb::hist2, dfb::tap1, t, t, 1);
        mul_tiles(dfb::hist3, dfb::tap2, t, t, 2);
        mul_tiles(dfb::cur, dfb::tap3, t, t, 3);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        add_binary_tile(0, 2, 0);
        add_binary_tile(0, 3, 0);
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        if (t < Kt) {
            pack_tile(0, dfb::qc, t);
        } else if (t < 2 * Kt) {
            pack_tile(0, dfb::kc, t - Kt);
        } else {
            pack_tile(0, dfb::vc, t - 2 * Kt);
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

// out[i] = (on[i] scaled per column by row 0 of w[i]) * zs[i]; the final product stays in DEST (SFPU fp32 multiply).
// on and zs are both fp32, so only the bcast pair needs a format reconfig (hoisted); the per-tile inits are cheap.
inline void gated_output(DataflowBuffer& out, uint32_t n) {
    out.reserve_back(n);
    pack_reconfig_data_format(dfb::out);
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
        pack_tile(0, dfb::out, i);
        tile_regs_release();
    }
    out.push_back(n);
}

}  // namespace

template <uint32_t Kt, uint32_t Vt, uint32_t scale_bits, uint32_t inv_dv_bits>
TT_KERNEL void compute(uint32_t wi_count) {
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t Ct = 2 * Kt + Vt;
    constexpr uint32_t one_bits = kOneBits;
    DataflowBuffer hist1(dfb::hist1);
    DataflowBuffer hist2(dfb::hist2);
    DataflowBuffer hist3(dfb::hist3);
    DataflowBuffer cur(dfb::cur);
    DataflowBuffer tap0(dfb::tap0);
    DataflowBuffer tap1(dfb::tap1);
    DataflowBuffer tap2(dfb::tap2);
    DataflowBuffer tap3(dfb::tap3);
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
    DataflowBuffer out(dfb::out);

    compute_kernel_hw_startup(dfb::hist1, dfb::state_in, dfb::out);
    scaler.wait_front(1);
    eps_l2.wait_front(1);
    eps_norm.wait_front(1);
    mask.wait_front(1);
    w_in.wait_front(Vt);
    tap0.wait_front(Ct);
    tap1.wait_front(Ct);
    tap2.wait_front(Ct);
    tap3.wait_front(Ct);

    for (uint32_t wi = 0; wi < wi_count; ++wi) {
        hist1.wait_front(Ct);
        hist2.wait_front(Ct);
        hist3.wait_front(Ct);
        cur.wait_front(Ct);
        z_in.wait_front(Vt);
        a_s.wait_front(1);
        b_s.wait_front(1);
        dtb_s.wait_front(1);
        nea_s.wait_front(1);
        state_in.wait_front(KV);

        // conv + silu -> qc, kc, vc ; gates
        causal_conv_silu<Kt, Vt>(qc, kc, vc);
        hist1.pop_front(Ct);
        hist2.pop_front(Ct);
        hist3.pop_front(Ct);
        cur.pop_front(Ct);
        gate_beta(beta_t);
        b_s.pop_front(1);
        gate_decay(dec);
        a_s.pop_front(1);
        dtb_s.pop_front(1);
        nea_s.pop_front(1);
        silu_tiles(dfb::z_in, dfb::zs, zs, Vt);
        z_in.pop_front(Vt);
        qc.wait_front(Kt);
        kc.wait_front(Kt);
        vc.wait_front(Vt);

        // qn = l2norm(q) * scale (rows 1..31 -> 0 through the mask)
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

        // kn = l2norm(k)
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

        // vm = v masked to row 0
        scale_rows(dfb::vc, dfb::mask, dfb::vm, vm, Vt);
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

        // out = rmsnorm(o) * w * silu(z)
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
        zs.wait_front(Vt);
        gated_output(out, Vt);
        on.pop_front(Vt);
        zs.pop_front(Vt);
    }
}
