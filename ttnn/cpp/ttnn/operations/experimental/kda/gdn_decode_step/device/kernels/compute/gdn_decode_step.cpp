// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One decode step of the gated delta rule for one value head per core (B = 1, token in row 0 of every input tile):
//   qn = l2norm(q) * scale, kn = l2norm(k), h = h * exp(g), v_read = kn @ h, delta = beta * (v - v_read),
//   h += kn^T @ delta, o = qn @ h, out = rmsnorm(o) * w.
// Rows 1..31 of q/k/v are zeroed through the mask column so the padding rows never touch the state.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/compute/gdn_step_helpers.hpp"

using namespace gdn_step;
template <uint32_t Kt, uint32_t Vt, uint32_t scale_bits, uint32_t inv_dv_bits>
TT_KERNEL void compute(uint32_t wi_count) {
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t one_bits = 0x3F800000u;
    DataflowBuffer q_in(dfb::q_in);
    DataflowBuffer k_in(dfb::k_in);
    DataflowBuffer v_in(dfb::v_in);
    DataflowBuffer beta_s(dfb::beta_s);
    DataflowBuffer g_s(dfb::g_s);
    DataflowBuffer state_in(dfb::state_in);
    DataflowBuffer w_in(dfb::w_in);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer eps_l2(dfb::eps_l2);
    DataflowBuffer eps_norm(dfb::eps_norm);
    DataflowBuffer mask(dfb::mask);
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

    compute_kernel_hw_startup(dfb::q_in, dfb::state_in, dfb::out);
    scaler.wait_front(1);
    eps_l2.wait_front(1);
    eps_norm.wait_front(1);
    mask.wait_front(1);
    w_in.wait_front(Vt);

    for (uint32_t wi = 0; wi < wi_count; ++wi) {
        q_in.wait_front(Kt);
        k_in.wait_front(Kt);
        v_in.wait_front(Vt);
        beta_s.wait_front(1);
        g_s.wait_front(1);
        state_in.wait_front(KV);

        // qn = l2norm(q) * scale (rows 1..31 -> 0 through the mask)
        square_tiles(dfb::q_in, dfb::tmp, tmp, Kt);
        compute_kernel_lib::
            reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, dfb::tmp, dfb::scaler, dfb::stats>(
                compute_kernel_lib::ReduceInputBlockShape::of(1, Kt));
        stats.wait_front(1);
        inverse_l2(dfb::stats, dfb::eps_l2, dfb::mask, dfb::scratch, scratch, dfb::inv, inv, scale_bits);
        stats.pop_front(1);
        inv.wait_front(1);
        scale_rows(dfb::q_in, dfb::inv, dfb::qn, qn, Kt);
        inv.pop_front(1);
        q_in.pop_front(Kt);

        // kn = l2norm(k)
        square_tiles(dfb::k_in, dfb::tmp, tmp, Kt);
        compute_kernel_lib::
            reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, dfb::tmp, dfb::scaler, dfb::stats>(
                compute_kernel_lib::ReduceInputBlockShape::of(1, Kt));
        stats.wait_front(1);
        inverse_l2(dfb::stats, dfb::eps_l2, dfb::mask, dfb::scratch, scratch, dfb::inv, inv, one_bits);
        stats.pop_front(1);
        inv.wait_front(1);
        scale_rows(dfb::k_in, dfb::inv, dfb::kn, kn, Kt);
        inv.pop_front(1);
        k_in.pop_front(Kt);

        // vm = v masked to row 0
        scale_rows(dfb::v_in, dfb::mask, dfb::vm, vm, Vt);
        v_in.pop_front(Vt);

        // hd = h * exp(g)
        exp_tile_copy(dfb::g_s, dfb::dec, dec);
        g_s.pop_front(1);
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
        multiply_tiles<true>(dfb::tmp, dfb::beta_s, dfb::delta, delta, Vt);
        tmp.pop_front(Vt);
        beta_s.pop_front(1);
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

        // out = rmsnorm(o) * w
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
        scale_cols(dfb::on, dfb::w_in, dfb::out, out, Vt);
        on.pop_front(Vt);
    }
}
