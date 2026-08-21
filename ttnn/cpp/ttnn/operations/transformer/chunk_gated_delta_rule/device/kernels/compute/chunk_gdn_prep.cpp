// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel: full chunked Gated Delta Rule forward for one head, sequential
// over chunks, holding state S [K,V] on-core. Derived from flash-linear-attention
// `naive_chunk_gated_delta_rule`. fp32 / HiFi4 throughout.
//
// Per chunk (C=chunk, K=key dim, V=val dim; Ct=C/32, Kt=K/32, Vt=V/32):
//   v_beta = v*beta ; k_beta = k*beta
//   decay = cumsum(g) = tril @ g ; decay_exp = exp(decay)
//   L_mask = tril( exp(decay_i - decay_j) )
//   N = strictly_lower(k_beta@k^T * L_mask) ; T_inv = (I+N)^-1  (Horner)
//   u = T_inv @ v_beta ; w = T_inv @ (k_beta*decay_exp)
//   intra = (q@k^T) * L_mask
//   q_decay = q*decay_exp ; k_dec_t = transpose(k * exp(decay_last - decay))
//   v_prime = w@S ; v_new = u - v_prime ; o = q_decay@S + intra@v_new
//   S = S*exp(decay_last) + k_dec_t@v_new
//
// The math bodies live in chunk_gdn_math.hpp (shared with the scan kernel); this file is just
// the CB map and the work-item loop calling prep_chunk().

#include <cstdint>
#include "api/compute/common.h"
#include "chunk_gdn_math.hpp"

namespace {

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4;
constexpr uint32_t cb_eye = 5, cb_tril = 6, cb_ones = 7, cb_S = 8;
constexpr uint32_t cb_decay = 9, cb_decay_exp = 10, cb_decayfac = 11;
constexpr uint32_t cb_lmask = 12, cb_Tinv = 13, cb_vbeta = 14, cb_kbeta = 15;
constexpr uint32_t cb_out = 16, cb_u = 17, cb_w = 18, cb_qdecay = 19;
constexpr uint32_t cb_intra = 20, cb_s2 = 21, cb_vnew = 22, cb_ointer = 23;
constexpr uint32_t cb_kdec_t = 24, cb_supd = 25, cb_stmp = 26, cb_final = 27;
constexpr uint32_t cb_scr1 = 28, cb_scr2 = 29, cb_scr3 = 30, cb_s3 = 31;
// PHASE A output for the scan step's state decay (reuses a scan-only index, unused in prep).
constexpr uint32_t cb_dl = cb_vnew;
// WY-inverse quadrant masks (3 tiles: 0=Qtl, 1=Qbr, 2=Q10). Reuses the cb_u slot (unused in
// the stable-form prep); the reader loads them once. Used only by invert_block.
constexpr uint32_t cb_mask = cb_u;

constexpr GdnPrepCbs CBS{
    .q = cb_q,
    .k = cb_k,
    .v = cb_v,
    .g = cb_g,
    .beta = cb_beta,
    .eye = cb_eye,
    .tril = cb_tril,
    .ones = cb_ones,
    .S = cb_S,
    .decay = cb_decay,
    .decay_exp = cb_decay_exp,
    .decayfac = cb_decayfac,
    .lmask = cb_lmask,
    .Tinv = cb_Tinv,
    .vbeta = cb_vbeta,
    .kbeta = cb_kbeta,
    .w = cb_w,
    .qdecay = cb_qdecay,
    .intra = cb_intra,
    .s2 = cb_s2,
    .ointer = cb_ointer,
    .kdec_t = cb_kdec_t,
    .supd = cb_supd,
    .stmp = cb_stmp,
    .final_s = cb_final,
    .scr1 = cb_scr1,
    .scr2 = cb_scr2,
    .scr3 = cb_scr3,
    .s3 = cb_s3,
    .dl = cb_dl,
    .mask = cb_mask,
    // GDN_SFPU_TINV prototype: bf16 L-staging tile. cb_out (c_16) is declared bf16 by both
    // factories and untouched by prep otherwise; harmless when the define is off.
    .lstage = cb_out};

}  // namespace

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);
    // OPT-B: QK_NORM=1 => L2-normalize q/k over K in-kernel (host skipped it), folding q's `scale`
    // into the norm. scale/eps arrive as fp32 bits. Only valid for Ct==1 (uses cb_supd/cb_stmp, which
    // are free outside the Ct==2 inverse branch) — the op host gates QK_NORM on chunk_size==32.
    constexpr uint32_t QK_NORM = get_compile_time_arg_val(3);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(4);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(5);
    // Chunk-parallel: NC here is this core's local work-item count (chunks assigned to it), NOT the
    // sequence-wide chunk count. Each work-item is an independent (head, chunk) prep — no cross-item
    // state — so the loop just processes `NC` items regardless of which (h, c) they map to.
    const uint32_t NC = get_arg_val<uint32_t>(0);

    constexpr uint32_t cc = Ct * Ct;

    compute_kernel_hw_startup(cb_q, cb_k, cb_u);

    // Constants (loaded once by reader). Initial state is in cb_S (reader pushed it).
    WAIT(cb_eye, cc);
    WAIT(cb_tril, cc);
    WAIT(cb_ones, cc);
    WAIT(cb_mask, 3);  // Qtl, Qbr, Q10 (used by invert_block)

    // PHASE A (prep): state-independent per-chunk quantities. No recurrent state here; the
    // sequential state scan lives in the separate scan kernel. Outputs (per chunk) u, w, k_dec_t,
    // q_decay, intra, dl are pushed to their CBs and streamed to DRAM by the prep writer.
    for (uint32_t c = 0; c < NC; c++) {
        prep_chunk(CBS, Ct, Kt, Vt, QK_NORM != 0, SCALE_BITS, EPS_BITS);
    }
}
