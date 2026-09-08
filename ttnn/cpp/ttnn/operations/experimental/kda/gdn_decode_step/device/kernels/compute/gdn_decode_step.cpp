// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One decode step of the gated delta rule for one value head per core (B = 1, token in row 0 of every input tile):
//   qn = l2norm(q) * scale, kn = l2norm(k), h = h * exp(g), v_read = kn @ h, delta = beta * (v - v_read),
//   h += kn^T @ delta, o = qn @ h, out = rmsnorm(o) * w.
// Rows 1..31 of q/k/v are zeroed through the mask column so the padding rows never touch the state.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace {

// out[i] = a[i] * b[i]  (plain eltwise; b_index_fixed -> b[0] for every i)
template <bool b_fixed>
FORCE_INLINE void multiply_tiles(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, b);
    mul_init(a, b);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        mul_tiles(a, b, i, b_fixed ? 0 : i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out[i] = a[i] - b[i]
FORCE_INLINE void subtract_tiles(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, b);
    sub_init(a, b);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        sub_tiles(a, b, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out[i] = a[i] + b[i]
FORCE_INLINE void add_tiles_n(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, b);
    add_init(a, b);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        add_tiles(a, b, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out[i] = a[i] scaled per row by column 0 of col_tile (tile 0 of b)
FORCE_INLINE void scale_rows(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, b);
    mul_bcast_cols_init(a, b);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(a, b, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out[i] = a[i] scaled per column by row 0 of b[i]
FORCE_INLINE void scale_cols(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, b);
    mul_bcast_rows_init(a, b);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(a, b, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out[i] = a[i] * a[i]  (bf16 or fp32 in, fp32 out)
FORCE_INLINE void square_tiles(uint32_t a, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, a);
    mul_init(a, a);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        mul_tiles(a, a, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// l2 variant: inv = (rsqrt(stats + eps) * post) * mask   (stats: row sums of squares in column 0)
FORCE_INLINE void inverse_l2(
    uint32_t stats,
    uint32_t eps,
    uint32_t mask,
    uint32_t scratch,
    DataflowBuffer& scratch_dfb,
    uint32_t inv,
    DataflowBuffer& inv_dfb,
    uint32_t post_scale_bits) {
    scratch_dfb.reserve_back(1);
    pack_reconfig_data_format(scratch);
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
    pack_tile(0, scratch, 0);
    tile_regs_release();
    scratch_dfb.push_back(1);
    scratch_dfb.wait_front(1);
    inv_dfb.reserve_back(1);
    pack_reconfig_data_format(inv);
    reconfig_data_format(scratch, mask);
    mul_init(scratch, mask);
    tile_regs_acquire();
    mul_tiles(scratch, mask, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, inv, 0);
    tile_regs_release();
    inv_dfb.push_back(1);
    scratch_dfb.pop_front(1);
}

// rms variant: inv = rsqrt(stats * inv_n + eps)
FORCE_INLINE void inverse_rms(
    uint32_t stats,
    uint32_t eps,
    uint32_t scratch,
    DataflowBuffer& scratch_dfb,
    uint32_t inv,
    DataflowBuffer& inv_dfb,
    uint32_t inv_n_bits) {
    scratch_dfb.reserve_back(1);
    pack_reconfig_data_format(scratch);
    reconfig_data_format_srca(stats);
    copy_init(stats);
    tile_regs_acquire();
    copy_tile(stats, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_n_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, scratch, 0);
    tile_regs_release();
    scratch_dfb.push_back(1);
    scratch_dfb.wait_front(1);
    inv_dfb.reserve_back(1);
    pack_reconfig_data_format(inv);
    reconfig_data_format(scratch, eps);
    add_init(scratch, eps);
    tile_regs_acquire();
    add_tiles(scratch, eps, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, inv, 0);
    tile_regs_release();
    inv_dfb.push_back(1);
    scratch_dfb.pop_front(1);
}

// out[j] = sum_i a[i] @ b[i*Nt + j]   (a: 1 x Kt row of tiles, b: Kt x Nt tiles)
FORCE_INLINE void row_times_matrix(
    uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t Kt, uint32_t Nt) {
    out_dfb.reserve_back(Nt);
    pack_reconfig_data_format(out);
    reconfig_data_format<SrcOrder::Reverse>(a, b);
    matmul_init(a, b);
    for (uint32_t j = 0; j < Nt; ++j) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < Kt; ++i) {
            matmul_tiles(a, b, i, i * Nt + j, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, j);
        tile_regs_release();
    }
    out_dfb.push_back(Nt);
}

// out[i*Nt + j] = a[i] @ b[j]   (outer product of a column block and a row block, K = 1 tile)
FORCE_INLINE void outer_product(
    uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t Kt, uint32_t Nt) {
    out_dfb.reserve_back(Kt * Nt);
    pack_reconfig_data_format(out);
    reconfig_data_format<SrcOrder::Reverse>(a, b);
    matmul_init(a, b);
    for (uint32_t i = 0; i < Kt; ++i) {
        for (uint32_t j = 0; j < Nt; ++j) {
            tile_regs_acquire();
            matmul_tiles(a, b, i, j, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out, i * Nt + j);
            tile_regs_release();
        }
    }
    out_dfb.push_back(Kt * Nt);
}

// out[i] = transpose(a[i])  (32-bit in-DST transpose)
FORCE_INLINE void transpose_tiles(uint32_t a, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(a);
    for (uint32_t i = 0; i < n; ++i) {
        // transpose_dest_init reprograms the math pipeline, so the datacopy must be re-initialised per tile
        copy_init(a);
        tile_regs_acquire();
        copy_tile(a, i, 0);
        transpose_dest_init<true>(a);
        transpose_dest<true>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out[i] = a[i]
FORCE_INLINE void copy_tiles(uint32_t a, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(a);
    copy_init(a);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        copy_tile(a, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out = exp(a[0])
FORCE_INLINE void exp_tile_copy(uint32_t a, uint32_t out, DataflowBuffer& out_dfb) {
    out_dfb.reserve_back(1);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(a);
    copy_init(a);
    tile_regs_acquire();
    copy_tile(a, 0, 0);
    exp_tile_init<false>();
    exp_tile<false>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out, 0);
    tile_regs_release();
    out_dfb.push_back(1);
}

}  // namespace

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
