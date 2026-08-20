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
//
// The math bodies live in chunk_gdn_math.hpp (shared with the prep kernel); this file is just
// the CB map and the chunk loop calling scan_step().

#include <cstdint>
#include "api/compute/common.h"
#include "chunk_gdn_math.hpp"

namespace {

// The seven per-chunk inputs live at PREP'S OUTPUT indices (v_beta=14, kd=18, q_decay=19,
// intra=20, k_dec_t=24, dl=22, t_inv=13) so the fused program can declare ONE hand-off CB set on
// the producer/receiver core union. That put dl at 22 (the slot prep's compute pushes dl into)
// and moved the v_new scratch to the freed 11. Indices are plumbing only — the phased path stays
// numerically identical across this renumber.
constexpr uint32_t cb_dl = 22, cb_Tinv = 13;
constexpr uint32_t cb_S = 8, cb_out = 16;
constexpr uint32_t cb_vbeta = 14, cb_kd = 18, cb_qdecay = 19, cb_intra = 20;
constexpr uint32_t cb_s2 = 21, cb_vnew = 11, cb_ointer = 23, cb_kdec_t = 24;
constexpr uint32_t cb_supd = 25, cb_stmp = 26, cb_final = 27;
constexpr uint32_t cb_scr1 = 28, cb_s3 = 31;

constexpr GdnScanCbs CBS{
    .dl = cb_dl,
    .Tinv = cb_Tinv,
    .out = cb_out,
    .vbeta = cb_vbeta,
    .kd = cb_kd,
    .qdecay = cb_qdecay,
    .intra = cb_intra,
    .vnew = cb_vnew,
    .ointer = cb_ointer,
    .kdec_t = cb_kdec_t,
    .supd = cb_supd,
    .stmp = cb_stmp,
    .scr1 = cb_scr1};

}  // namespace

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);
    const uint32_t NC = get_arg_val<uint32_t>(0);

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

        scan_step(CBS, cur_S, dst, Ct, Kt, Vt);
    }
}
