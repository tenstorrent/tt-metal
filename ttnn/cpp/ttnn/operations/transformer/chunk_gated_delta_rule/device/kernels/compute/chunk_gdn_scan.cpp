// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B (scan) compute kernel: the sequential-over-chunk recurrence for one head.
// Consumes the state-independent per-chunk quantities produced by the prep phase
// (u, w, q_decay, intra, k_dec_t, dl) and carries the recurrent state S [K,V] on-core.
//
// Per chunk (Ct=C/32, Kt=K/32, Vt=V/32):
//   v_new = T_inv @ (v_beta + nkd @ S)      (nkd = -(k_beta*decay_exp), handed off negated)
//   o     = q_decay @ S + intra @ v_new         (one DST accumulate)
//   S     = (dl*I) @ S + k_dec_t @ v_new        (one DST accumulate; dl = exp(g_sum) on the diagonal)
// No matrix inverse here — that (the expensive part) lives entirely in the prep phase.
//

#include <cstdint>
#include "api/compute/common.h"
#include "api/debug/kernel_profiler.hpp"
// Scan-only: batch four fp32 output tiles per DST acquire in the shared math helpers.
// The prep kernel stays per-tile (its Ct=2 binary is at the kernel-config-buffer limit).
#define GDN_DST_TILES 4
#include "chunk_gdn_math.hpp"
#include "../dataflow/chunk_gdn_handoff.hpp"

namespace {

// The seven per-chunk inputs live at PREP'S OUTPUT indices (v_beta=14, nkd=18, q_decay=19,
// intra=20, k_dec_t=24, dl=22, t_inv=13) so the fused program can declare ONE hand-off CB set on
// the producer/receiver core union. That put dl at 22 (the slot prep's compute pushes dl into)
// and moved the v_new scratch to the freed 11.
constexpr uint32_t cb_dl = 22, cb_Tinv = 13;
constexpr uint32_t cb_S = 8, cb_out = 16;
constexpr uint32_t cb_vbeta = 14, cb_nkd = 18, cb_qdecay = 19, cb_intra = 20;
constexpr uint32_t cb_s2 = 21, cb_vnew = 11, cb_ointer = 23, cb_kdec_t = 24;
constexpr uint32_t cb_final = 27;
constexpr uint32_t cb_eye = 5;  // one 32x32 identity tile, written by the reader (scan_step's I @ v_beta operand)
static_assert(
    cb_Tinv == gdn_handoff::kCbTinv && cb_vbeta == gdn_handoff::kCbVbeta && cb_nkd == gdn_handoff::kCbNkd &&
        cb_qdecay == gdn_handoff::kCbQdecay && cb_intra == gdn_handoff::kCbIntra &&
        cb_kdec_t == gdn_handoff::kCbKdecT && cb_dl == gdn_handoff::kCbDl,
    "scan input CB indices drifted from ../dataflow/chunk_gdn_handoff.hpp");
constexpr uint32_t cb_s3 = 31;
#ifdef GDN_EXACT_DL
// FPU-read views of the state (the exact state CBs 8/21/31 are unpacked straight to DST for the SFPU update)
constexpr uint32_t cb_vS = 9, cb_v2 = 10, cb_v3 = 12;
#endif

constexpr GdnScanCbs CBS{
    .dl = cb_dl,
    .Tinv = cb_Tinv,
    .out = cb_out,
    .vbeta = cb_vbeta,
    .nkd = cb_nkd,
    .qdecay = cb_qdecay,
    .intra = cb_intra,
    .vnew = cb_vnew,
    .ointer = cb_ointer,
    .kdec_t = cb_kdec_t,
    .eye = cb_eye};

}  // namespace

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);
    const uint32_t NC = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_nkd, cb_vbeta, cb_out);

    WAIT(cb_eye, 1);  // the reader's identity tile (never popped)
#ifdef GDN_EXACT_DL
    constexpr uint32_t kv = Kt * Vt;
#endif

    for (uint32_t c = 0; c < NC; c++) {
        // State uses three single-producer CBs:
        //   cb_S      : reader-produced initial state, consumed only by chunk 0.
        //   cb_s2/cb_s3: compute-only ping-pong for chunk outputs.
        const uint32_t cur_S = (c == 0) ? cb_S : ((c & 1u) ? cb_s2 : cb_s3);
#ifdef GDN_EXACT_DL
        const uint32_t cur_view = (c == 0) ? cb_vS : ((c & 1u) ? cb_v2 : cb_v3);
#endif
        const uint32_t nxt_S = (c & 1u) ? cb_s3 : cb_s2;
#ifdef GDN_EXACT_DL
        const uint32_t nxt_view = (c & 1u) ? cb_v3 : cb_v2;
#endif
        const bool last = (c == NC - 1);
        const uint32_t dst = last ? cb_final : nxt_S;
#ifdef GDN_EXACT_DL
        const uint32_t dst_view = last ? cb_final : nxt_view;  // == dst: no view for the final state
        if (c == 0) {
            // view copy of the reader's initial state (exact DST copy, packed for the FPU reads)
            WAIT(cb_S, kv);
            CircularBuffer(cb_vS).reserve_back(kv);
            for (uint32_t t0 = 0; t0 < kv; t0 += kDstTiles) {
                const uint32_t nb = (kv - t0 < kDstTiles) ? (kv - t0) : kDstTiles;
                tile_regs_acquire();
                copy_init(cb_S);
                for (uint32_t j = 0; j < nb; j++) {
                    copy_tile(cb_S, t0 + j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < nb; j++) {
                    pack_tile<true>(j, cb_vS, t0 + j);
                }
                tile_regs_release();
            }
            CircularBuffer(cb_vS).push_back(kv);
        }
#endif

#if defined(PROFILE_KERNEL)
        {
            // Diagnostic only (Tracy device runs): wait for all seven inputs up front so the zone below
            // measures pure compute.
            DeviceZoneScopedN("scan_wait_in");
            WAIT(cb_nkd, Ct * Kt);
            WAIT(cb_vbeta, Ct * Vt);
            WAIT(cb_Tinv, Ct * Ct);
            WAIT(cb_qdecay, Ct * Kt);
            WAIT(cb_intra, Ct * Ct);
            WAIT(cb_kdec_t, Kt * Ct);
            WAIT(cb_dl, 1);
        }
#endif
        {
            DeviceZoneScopedN("scan_step");
#ifdef GDN_EXACT_DL
            scan_step<Ct, Kt, Vt>(CBS, cur_S, cur_view, dst, dst_view);
#else
            scan_step<Ct, Kt, Vt>(CBS, cur_S, dst);
#endif
        }
    }
}
