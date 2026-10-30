// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel: the sequential-over-token recurrence for one head. Carries the recurrent state
// S [K,V] on-core across T tokens. Derived from flash-linear-attention
// `naive_recurrent_gated_delta_rule`; the vLLM `fused_sigmoid_gating_delta_rule_update` is this
// same recurrence over the K+1 speculative tokens.
//
// Per token (q pre-scaled + L2-normed and k L2-normed on the host; g is the log decay):
//   e      = exp(g)                      (SFPU, exact)
//   c      = q . k                       (FPU)
//   r      = [k; q] . S_old              (FPU pass 1: r_k = k . S_old, r_q = q . S_old)
//   u      = beta * (v - e * r_k)        (SFPU, fp32)
//   o      = e * r_q + c * u             (SFPU)
//   S_new  = e * S_old + k^T (x) u       (per S tile: exact seed, SFPU decay, FPU rank-1 accumulate)
// The passes are decode_token<Kt, Vt> in fused_recurrent_gdn_math.hpp; this file is the token loop and the state
// ring: S_cur is the reader's S at t = 0 and then alternates between s2 and s3, and the state goes to state_out for
// every token (per_token) or the last one. CB ids and tile dims: ../fused_recurrent_gdn_cb.hpp.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

#include "../fused_recurrent_gdn_cb.hpp"

namespace cb = gdn_decode_cb;

namespace {

// ---- STUB of decode_token, until fused_recurrent_gdn_math.hpp lands ------------------------------------------------
// Consumes and produces exactly the CBs of the real decode_token, in the same counts, so the reader tiles, the CB
// sizes and the writer layout can be exercised without hangs: o = v (row 0 of v16 -> row 0 of out16) and
// S_next = S_cur (exact copies). ecol32 / r16 / craw16 / ccol32 / u16 are compute-internal and not touched.
constexpr uint32_t DST_TILES = 4;  // fp32 DEST tiles per acquire (half-sync)

template <uint32_t Kt, uint32_t Vt>
void decode_token(uint32_t cb_s_cur, uint32_t cb_s_next, bool emit_state) {
    constexpr uint32_t kv = Kt * Vt;
    CircularBuffer qk(cb::qk16), kT(cb::kT16), qc(cb::qcol32), v(cb::v16), g(cb::gcol32), b(cb::bcol16);
    CircularBuffer s_cur(cb_s_cur), s_next(cb_s_next), out(cb::out16), st(cb::state_out);

    qk.wait_front(Kt);
    kT.wait_front(Kt);
    qc.wait_front(Kt);
    v.wait_front(Vt);
    g.wait_front(1);
    b.wait_front(1);
    s_cur.wait_front(kv);

    // o = v: 16x32 -> 16x32.
    out.reserve_back(Vt);
    copy_init(cb::v16);
    pack_reconfig_data_format<true>(cb::out16);
    for (uint32_t j = 0; j < Vt; j++) {
        tile_regs_acquire();
        copy_tile(cb::v16, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb::out16, j);
        tile_regs_release();
    }
    out.push_back(Vt);

    // S_next = S_cur (and state_out): 32x32 -> 32x32, groups of DST_TILES.
    s_next.reserve_back(kv);
    if (emit_state) {
        st.reserve_back(kv);
    }
    copy_init(cb_s_cur);
    pack_reconfig_data_format<true>(cb_s_next);
    for (uint32_t i0 = 0; i0 < kv; i0 += DST_TILES) {
        const uint32_t nn = (kv - i0 < DST_TILES) ? (kv - i0) : DST_TILES;
        tile_regs_acquire();
        for (uint32_t j = 0; j < nn; j++) {
            copy_tile(cb_s_cur, i0 + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nn; j++) {
            pack_tile<true>(j, cb_s_next, i0 + j);
            if (emit_state) {
                pack_tile<true>(j, cb::state_out, i0 + j);
            }
        }
        tile_regs_release();
    }
    s_next.push_back(kv);
    if (emit_state) {
        st.push_back(kv);
    }

    qk.pop_front(Kt);
    kT.pop_front(Kt);
    qc.pop_front(Kt);
    v.pop_front(Vt);
    g.pop_front(1);
    b.pop_front(1);
    s_cur.pop_front(kv);
}
// ---- end of STUB -----------------------------------------------------------------------------------------------

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t per_token = get_compile_time_arg_val(2);
    const uint32_t T = get_arg_val<uint32_t>(0);

    // Every CB is fp32 (see the program factory); the passes reconfigure the per-CB tile dims as they go.
    compute_kernel_hw_startup(cb::qk16, cb::S, cb::out16);

    for (uint32_t t = 0; t < T; t++) {
        const uint32_t cur_S_id = (t == 0) ? cb::S : ((t & 1u) ? cb::s2 : cb::s3);
        const uint32_t nxt_S_id = (t & 1u) ? cb::s3 : cb::s2;
        const bool last = (t == T - 1);
        // decode_token pops S_cur and pushes S_next; S_next is intentionally left in its CB: the next iteration
        // reads it as S_cur.
        decode_token<Kt, Vt>(cur_S_id, nxt_S_id, per_token || last);
    }
}
