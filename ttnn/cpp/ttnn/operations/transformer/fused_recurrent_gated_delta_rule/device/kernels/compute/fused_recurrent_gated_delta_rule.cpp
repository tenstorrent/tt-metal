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
// The passes are decode_token<Kt, Vt, true> (fused read + vector pass) in fused_recurrent_gdn_math.hpp; this file is
// the token loop and the state
// ring: S_cur is the reader's S at t = 0 and then alternates between s2 and s3 (each a utd/mm id pair over one
// allocation), and the state goes to state_out for every token (per_token) or the last one. CB ids and tile dims:
// ../fused_recurrent_gdn_cb.hpp.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

#include "../fused_recurrent_gdn_cb.hpp"
#include "fused_recurrent_gdn_math.hpp"

namespace cb = gdn_decode_cb;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t per_token = get_compile_time_arg_val(2);
    const uint32_t T = get_arg_val<uint32_t>(0);

    // Every CB is fp32 (see the program factory). Matmul operand order: in0 (qk16) -> SrcB, in1 (S) -> SrcA; the
    // passes re-initialise the unpacker / packer per CB tile shape as they go.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb::qk16, cb::S, cb::out16);

    // r16 and ccol32 are named by TokenCbs but never touched by the fused read+vector form; their ids are not
    // allocated (see the CB header).
    constexpr gdn_decode::TokenCbs tcb{
        .qk16 = cb::qk16,
        .kT16 = cb::kT16,
        .qcol32 = cb::qcol32,
        .v16 = cb::v16,
        .gcol32 = cb::gcol32,
        .bcol16 = cb::bcol16,
        .ones32 = cb::ones32,
        .ecol32 = cb::ecol32,
        .r16 = cb::freed_r16,
        .craw16 = cb::craw16,
        .ccol32 = cb::freed_ccol32,
        .u16 = cb::u16,
        .out16 = cb::out16,
        .state_out = cb::state_out,
        .ebc16 = cb::ebc16,
    };
    // Each state lives under two ids over one allocation (utd for the exact copies, mm for the matmul in1).
    constexpr gdn_decode::StateCbs S0{.utd = cb::S, .mm = cb::S_mm};
    constexpr gdn_decode::StateCbs S2{.utd = cb::s2, .mm = cb::s2_mm};
    constexpr gdn_decode::StateCbs S3{.utd = cb::s3, .mm = cb::s3_mm};

    for (uint32_t t = 0; t < T; t++) {
        const gdn_decode::StateCbs cur_S = (t == 0) ? S0 : ((t & 1u) ? S2 : S3);
        const gdn_decode::StateCbs nxt_S = (t & 1u) ? S3 : S2;
        const bool last = (t == T - 1);
        // decode_token pops the token's inputs and S_cur (pop_inputs = true); S_next is intentionally left in its
        // CBs: the next iteration reads it as S_cur.
        gdn_decode::decode_token<Kt, Vt, /*FuseReadVector=*/true>(tcb, cur_S, nxt_S, per_token || last);
    }
}
