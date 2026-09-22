// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Types for the fused T=1 (decode step) gated delta rule ttnn op.
// One Tensix core per (B*H) head; the whole recurrent step
// (L2-norm q/k, decay, v_read, delta, rank-1 write, o = q@h) runs in one
// reader/compute/writer program, matching the python graph
// `recurrent_gated_delta_rule_decode_ttnn`.

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct DecodeGatedDeltaRuleParams {
    uint32_t B;    // batch
    uint32_t H;    // heads (== v heads; no GQA in the decode graph)
    uint32_t BH;   // B * H (one core per head)
    uint32_t K;    // key dim (multiple of 32)
    uint32_t V;    // value dim (multiple of 32)
    bool has_initial_state;
    bool inplace_state;  // write new state back into initial_state's buffer
    float scale;         // folded into q's L2-norm factor (K**-0.5 by default)
    tt::tt_metal::MemoryConfig output_mem_config;
};

// All inputs are python-facing shapes, TILE layout, same dtype (bf16 or fp32),
// on device. T=1 inputs ([B,1,H,*]) share TILE pages across 32 heads; the
// reader gathers each head's row out of the shared pages. Outputs: o comes
// back ROW_MAJOR (page bh == head bh's [V] stick; full-page writes only), the
// new state is TILE [B,H,K,V].
struct DecodeGatedDeltaRuleInputs {
    Tensor q;                             // [B,1,H,K]
    Tensor k;                             // [B,1,H,K]
    Tensor v;                             // [B,1,H,V]
    Tensor beta;                          // [B,1,H]
    Tensor g;                             // [B,1,H]  log-space decay
    std::optional<Tensor> initial_state;  // [B,H,K,V] (absent => zeros)
};

}  // namespace ttnn::prim
