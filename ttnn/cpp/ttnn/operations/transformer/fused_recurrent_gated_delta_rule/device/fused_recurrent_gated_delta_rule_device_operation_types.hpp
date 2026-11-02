// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Types for the standalone fused_recurrent_gated_delta_rule ttnn op.
// Algorithm: flash-linear-attention `naive_recurrent_gated_delta_rule` (the recurrent form the
// vLLM `fused_sigmoid_gating_delta_rule_update` implements). One Tensix core per head walks the
// T token axis sequentially, holding the recurrent state S [K,V] on-core.
//
// Per token t (S = state, g_t log-decay, beta_t gate, q pre-scaled + L2-normed, k L2-normed):
//   S      <- exp(g_t) * S                 (scalar decay)
//   u_t    =  beta_t * (v_t - k_t . S)      (delta write value)
//   S      <- S + k_t^T (x) u_t             (rank-1 update)
//   o_t    =  q_t . S                        (read from the POST-update state)

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

struct FusedRecurrentGatedDeltaRuleParams {
    uint32_t BH;       // B * HV  (one Tensix core per head)
    uint32_t T;        // number of tokens processed sequentially per head
    uint32_t key_dim;  // K (multiple of 32)
    uint32_t val_dim;  // V (multiple of 32)
    // GQA group size HV/H. When > 1 the caller passes q/k with only H query heads and the READER
    // replicates head hv/G onto core hv, so no repeat_interleave tensor is built on the host.
    uint32_t gqa;
    bool output_final_state;
    bool output_per_token_state;  // emit S after EVERY token (spec-decode verify slots)
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    // gb_fused (decode, T == 1): `decay` carries the raw a|b row [1,1,2*BH] (a = cols [0,BH), b = [BH,2BH)) and
    // `beta` the per-head constants [1,1,2*BH] (-exp(A_log) | dt_bias), all fp32. The kernel computes
    // decay = exp(-exp(A_log) * softplus(a + dt_bias)) and beta = sigmoid(b) itself (QWEN36_1C_GDN_GB).
    bool gb_fused = false;
    // fused_l2norm (decode, T == 1): q and k arrive un-normalized; the kernel computes q*l2_scale/||q|| and k/||k||
    // (eps 1e-6 under the square root, as rms_norm(x, 1e-6/K) * K^-0.5 does) before the recurrence (QWEN36_1C_GDN_L2).
    bool fused_l2norm = false;
    float l2_scale = 1.0f;
};

// All device-op input tensors are fp32, TILE layout, interleaved (DRAM or L1).
// Head-split, L2-norm, q-scale, exp(g)->decay and the per-token tile layout are done host-side;
// the compute kernel does only the pure recurrence.
//   q, k : [BH*T, 1, K]   (q already scaled by `scale`; both already L2-normalized over K)
//   v    : [BH*T, 1, V]
//   decay: [BH*T, 1, 1]   (= exp(g_t), one scalar per (head, token) in tile element [0,0])
//   beta : [BH*T, 1, 1]
//   initial_state: [BH, K, V] or absent (zeros).
//
// "Ring" mode (initial_state_block_idx present): initial_state is the [BH*T, K, V] ring and also the state output;
// see the Python binding for the contract.
struct FusedRecurrentGatedDeltaRuleInputs {
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor decay;
    Tensor beta;
    std::optional<Tensor> initial_state;
    std::optional<Tensor> initial_state_block_idx;
};

}  // namespace ttnn::prim
