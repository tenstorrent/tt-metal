// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel: the whole T=1 gated delta rule recurrent step for one head,
// fp32 accumulation throughout. Mirrors the python graph
// `recurrent_gated_delta_rule_decode_ttnn` (bf16/fp32 IO tiles, math in fp32):
//
//   qn = l2norm(q) * scale        (scale == K**-0.5; folded into the norm)
//   kn = l2norm(k)
//   h  = state * exp(g)           (per-head scalar g broadcast)
//   v_read = kn @ h               ([1,K]@[K,V] -> [1,V])
//   delta  = v - v_read           ([1,V])
//   outer  = (kn)^T @ (beta*delta)  ([K,1]@[1,V] rank-1)
//   new_h  = h + outer
//   o      = qn @ new_h           ([1,K]@[K,V] -> [1,V])
//
// All inputs arrive as row-0 tiles (reader gather). Padding rows are exact
// zeros, so the rank-1 outer product and row-broadcast norms are exact.
//
// Compile args: {Kt, Vt, has_s0, EPS_BITS, SCALE_BITS} (eps/scale fp32 bits).

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

namespace {

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4;
constexpr uint32_t cb_state = 5, cb_ones = 6;
constexpr uint32_t cb_qsq = 7, cb_ksq = 8, cb_sc = 9;
constexpr uint32_t cb_qn = 10, cb_kn = 11, cb_kcol = 12, cb_gexp = 13;
constexpr uint32_t cb_sdec = 14, cb_vread = 15, cb_delta = 16;
constexpr uint32_t cb_outer = 17, cb_sout = 18, cb_out = 19;

inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }

// out[Mt,Nt] = A[Mt,Kt] @ (tr ? B[Nt,Kt]^T : B[Kt,Nt]). Inputs must be available.
void mm(uint32_t a, uint32_t b, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t Nt, bool tr) {
    cb_reserve_back(o, Mt * Nt);
    pack_reconfig_data_format(o);
    reconfig_data_format(b, a);  // matmul_tiles(a,b): in0=a->srcB, in1=b->srcA
    matmul_init(a, b, tr ? 1 : 0);
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
}

// out = A (op) B elementwise, n tiles. op: 0 add, 1 sub, 2 mul.
void ew(uint32_t a, uint32_t b, uint32_t o, uint32_t n, int op) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);  // binary(a,b): a->srcA, b->srcB
    if (op == 0) {
        add_init(a, b);
    } else if (op == 1) {
        sub_init(a, b);
    } else {
        mul_init(a, b);
    }
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
}

// out = exp(in), n tiles (copy to fp32 DST, SFPU exp).
void expc(uint32_t in, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
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

// out = S * scalar[0,0], n tiles.
void bcast_scalar_mul(uint32_t a, uint32_t scal, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, scal);  // bcast(a,scal): a->srcA, scal->srcB
    mul_bcast_scalar_init(a, scal);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(a, scal, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out[Mt,Nt] = A[Mt,Nt] * col[Mt,1] (broadcast col's single column across N).
void bcast_cols_mul(uint32_t a, uint32_t col, uint32_t o, uint32_t Mt, uint32_t Nt) {
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

// rowsum_k: o[1 tile] = in[1,Kt] @ ones — every element [r,c] of the output
// holds the row-r sum (row 0 = this head's sum of squares of `in`). Reuses the
// reader-built fp32 ones tile as the contraction operand.
void rowsum_k(uint32_t in, uint32_t o, uint32_t Kt) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(cb_ones, in);  // matmul(in, ones): in->srcB, ones->srcA
    matmul_init(in, cb_ones, 0);
    tile_regs_acquire();
    for (uint32_t ki = 0; ki < Kt; ki++) {
        matmul_tiles(in, cb_ones, ki, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

// inv_rms: o = rsqrt(in + eps) [* scale]. in holds per-row sums of squares
// (rowsum_k output); every column of a row carries the same factor, so the
// result feeds bcast_cols_mul directly. eps/scale arrive as fp32-bit-cast
// uint32 compile args. Same CB for in/out: requires 2 pages (reserved by the
// program factory).
void inv_rms(uint32_t in, uint32_t o, uint32_t eps_bits, uint32_t scale_bits, bool do_scale) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    tile_regs_acquire();
    copy_tile(in, 0, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps_bits);  // + eps  (== python rms_norm(x, eps/K))
    rsqrt_tile_init();
    rsqrt_tile(0);  // 1/sqrt(sumsq + eps)
    if (do_scale) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scale_bits);  // * scale (q only; == python q*K**-0.5)
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

// o[Kt,1] = transpose(in[1,Kt]): tile ki of o is the transposed tile ki of in
// (row 0 -> column 0), the column form needed for the rank-1 outer product.
void transpose_row(uint32_t in, uint32_t o, uint32_t Kt) {
    cb_reserve_back(o, Kt);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    transpose_init(in);
    for (uint32_t ki = 0; ki < Kt; ki++) {
        tile_regs_acquire();
        transpose_tile(in, ki, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, ki);
        tile_regs_release();
    }
    cb_push_back(o, Kt);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t has_s0 = get_compile_time_arg_val(2);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(3);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(4);
    (void)has_s0;

    constexpr uint32_t kv = Kt * Vt;

    compute_kernel_hw_startup(cb_q, cb_k, cb_out);

    WAIT(cb_q, Kt);
    WAIT(cb_k, Kt);
    WAIT(cb_v, Vt);
    WAIT(cb_g, 1);
    WAIT(cb_beta, 1);
    WAIT(cb_state, kv);
    WAIT(cb_ones, 1);

    // ---- L2 norm q (scale folded): qn = q * scale / sqrt(||q||^2 + eps) ----
    ew(cb_q, cb_q, cb_qsq, Kt, 2);  // q^2
    WAIT(cb_qsq, Kt);
    rowsum_k(cb_qsq, cb_sc, Kt);  // [0,*] = ||q||^2
    WAIT(cb_sc, 1);
    POP(cb_qsq, Kt);
    inv_rms(cb_sc, cb_sc, EPS_BITS, SCALE_BITS, true);
    POP(cb_sc, 1);  // drop the sumsq page; front is now the inv-norm page
    bcast_cols_mul(cb_q, cb_sc, cb_qn, 1, Kt);
    WAIT(cb_qn, Kt);
    POP(cb_q, Kt);
    POP(cb_sc, 1);

    // ---- L2 norm k (no scale) ----
    ew(cb_k, cb_k, cb_ksq, Kt, 2);
    WAIT(cb_ksq, Kt);
    rowsum_k(cb_ksq, cb_sc, Kt);
    WAIT(cb_sc, 1);
    POP(cb_ksq, Kt);
    inv_rms(cb_sc, cb_sc, EPS_BITS, SCALE_BITS, false);
    POP(cb_sc, 1);
    bcast_cols_mul(cb_k, cb_sc, cb_kn, 1, Kt);
    WAIT(cb_kn, Kt);
    POP(cb_k, Kt);
    POP(cb_sc, 1);

    // ---- decay: h = state * exp(g) ----
    expc(cb_g, cb_gexp, 1);  // [0,0] = exp(g_h)
    WAIT(cb_gexp, 1);
    POP(cb_g, 1);
    bcast_scalar_mul(cb_state, cb_gexp, cb_sdec, kv);
    WAIT(cb_sdec, kv);
    POP(cb_state, kv);
    POP(cb_gexp, 1);

    // ---- v_read = kn @ h; delta = v - v_read; delta *= beta ----
    mm(cb_kn, cb_sdec, cb_vread, 1, Kt, Vt, false);
    WAIT(cb_vread, Vt);
    ew(cb_v, cb_vread, cb_delta, Vt, 1);  // delta = v - v_read
    WAIT(cb_delta, Vt);
    POP(cb_v, Vt);
    POP(cb_vread, Vt);
    bcast_scalar_mul(cb_delta, cb_beta, cb_delta, Vt);  // in-place (2-page CB)
    POP(cb_beta, 1);
    POP(cb_delta, Vt);  // drop the pre-beta page

    // ---- rank-1 write: new_h = h + (kn)^T @ (beta*delta) ----
    transpose_row(cb_kn, cb_kcol, Kt);  // kn -> column form
    WAIT(cb_kcol, Kt);
    POP(cb_kn, Kt);
    mm(cb_kcol, cb_delta, cb_outer, Kt, 1, Vt, false);  // [K,1]@[1,V]
    WAIT(cb_outer, kv);
    POP(cb_delta, Vt);
    POP(cb_kcol, Kt);
    ew(cb_sdec, cb_outer, cb_sout, kv, 0);  // new state (io dtype)
    WAIT(cb_sout, kv);
    POP(cb_sdec, kv);
    POP(cb_outer, kv);

    // ---- o = qn @ new_h ----
    mm(cb_qn, cb_sout, cb_out, 1, Kt, Vt, false);
    POP(cb_qn, Kt);
}
