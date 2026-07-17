// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B (scan) compute kernel: the sequential-over-chunk recurrence for one head.
// Consumes the state-independent per-chunk quantities produced by the prep phase
// (u, w, q_decay, intra, k_dec_t, dl) and carries the recurrent state S [K,V] on-core.
//
// Per chunk (Ct=C/32, Kt=K/32, Vt=V/32):
//   v_prime = w @ S ; v_new = u - v_prime
//   o       = q_decay @ S + intra @ v_new
//   s_upd   = k_dec_t @ v_new
//   S       = S * dl + s_upd        (dl = exp(g_sum), scalar in dl tile [0,0])
// No matrix inverse here — that (the expensive part) lives entirely in the prep phase.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t cb_dl = 11, cb_Tinv = 13;
constexpr uint32_t cb_S = 8, cb_out = 16;
constexpr uint32_t cb_vbeta = 17, cb_kd = 18, cb_qdecay = 19, cb_intra = 20;
constexpr uint32_t cb_s2 = 21, cb_vnew = 22, cb_ointer = 23, cb_kdec_t = 24;
constexpr uint32_t cb_supd = 25, cb_stmp = 26, cb_final = 27;
constexpr uint32_t cb_scr1 = 28, cb_s3 = 31;

inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }

// out[Mt,Nt] = A[Mt,Kt] @ (tr ? B[Nt,Kt]^T : B[Kt,Nt]). Inputs must be available.
void mm(uint32_t a, uint32_t b, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t Nt, bool tr) {
    cb_reserve_back(o, Mt * Nt);
    pack_reconfig_data_format(o);  // mixed bf16/fp32 CBs: set packer to this output's format
    // matmul_tiles(a,b): in0=a->srcB, in1=b->srcA. The op init only asserts formats, it does not
    // set them, so reconfig the unpack src formats explicitly (else CBs read at the wrong format).
    reconfig_data_format(b, a);
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
        add_tiles_init(a, b);
    } else if (op == 1) {
        sub_tiles_init(a, b);
    } else {
        mul_tiles_init(a, b);
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

// out = A * scalar, n tiles. scalar is the [0,0] element of the single `scal` tile.
void bcast_scalar_mul(uint32_t a, uint32_t scal, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, scal);  // bcast(a,scal): a->srcA, scal->srcB
    mul_tiles_bcast_scalar_init_short(a, scal);
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

}  // namespace

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);
    const uint32_t NC = get_arg_val<uint32_t>(0);

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t kc = Kt * Ct;

    compute_kernel_hw_startup(cb_kd, cb_vbeta, cb_out);

    for (uint32_t c = 0; c < NC; c++) {
        // State uses THREE single-producer CBs so no CB is produced by both the reader and
        // compute (that reader->compute producer switch desyncs CB page pointers and deadlocks):
        //   cb_S      : reader-produced initial state, consumed only by chunk 0.
        //   cb_s2/cb_s3: compute-only ping-pong for chunk outputs.
        const uint32_t cur_S = (c == 0) ? cb_S : ((c & 1u) ? cb_s2 : cb_s3);
        const uint32_t nxt_S = (c & 1u) ? cb_s3 : cb_s2;
        const bool last = (c == NC - 1);
        const uint32_t dst = last ? cb_final : nxt_S;

        // v_new = T_inv @ (v_beta - kd@S)  -- apply the inverse AFTER the subtraction so the WY
        // inverse's fp error is not amplified by the cancellation (vs the u - w@S form).
        WAIT(cb_kd, ck);
        WAIT(cur_S, kv);
        mm(cb_kd, cur_S, cb_scr1, Ct, Kt, Vt, false);  // kdS = kd @ S -> scr1
        WAIT(cb_scr1, cv);
        POP(cb_kd, ck);
        WAIT(cb_vbeta, cv);
        ew(cb_vbeta, cb_scr1, cb_ointer, cv, 1);  // diff = v_beta - kdS -> ointer
        WAIT(cb_ointer, cv);
        POP(cb_vbeta, cv);
        POP(cb_scr1, cv);
        WAIT(cb_Tinv, cc);
        mm(cb_Tinv, cb_ointer, cb_vnew, Ct, Ct, Vt, false);  // v_new = T_inv @ diff -> vnew
        WAIT(cb_vnew, cv);
        POP(cb_Tinv, cc);
        POP(cb_ointer, cv);

        // o = q_decay @ S + intra @ v_new
        WAIT(cb_qdecay, ck);
        mm(cb_qdecay, cur_S, cb_ointer, Ct, Kt, Vt, false);  // o_inter = q_decay @ S
        WAIT(cb_ointer, cv);
        POP(cb_qdecay, ck);
        WAIT(cb_intra, cc);
        mm(cb_intra, cb_vnew, cb_scr1, Ct, Ct, Vt, false);  // intra_v = intra @ v_new
        WAIT(cb_scr1, cv);
        POP(cb_intra, cc);
        ew(cb_ointer, cb_scr1, cb_out, cv, 0);  // o -> cb_out (drained by writer)
        POP(cb_ointer, cv);
        POP(cb_scr1, cv);

        // s_upd = k_dec_t @ v_new
        WAIT(cb_kdec_t, kc);
        mm(cb_kdec_t, cb_vnew, cb_supd, Kt, Ct, Vt, false);
        WAIT(cb_supd, kv);
        POP(cb_kdec_t, kc);
        POP(cb_vnew, cv);

        // S_new = cur_S * dl + s_upd  (dl scalar in cb_dl tile [0,0])
        WAIT(cb_dl, 1);
        bcast_scalar_mul(cur_S, cb_dl, cb_stmp, kv);
        WAIT(cb_stmp, kv);
        POP(cb_dl, 1);
        POP(cur_S, kv);
        ew(cb_stmp, cb_supd, dst, kv, 0);
        POP(cb_stmp, kv);
        POP(cb_supd, kv);
    }
}
