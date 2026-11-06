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
    uint32_t B;        // batch rows of the call
    uint32_t H;        // key heads; value head hv reads key head hv / (HV / H)
    uint32_t HV;       // value heads
    uint32_t BH;       // B * HV  (one Tensix core per head)
    uint32_t T;        // number of tokens processed sequentially per head
    uint32_t key_dim;  // K (multiple of 32)
    uint32_t val_dim;  // V (multiple of 32)
    bool output_final_state;
    bool output_per_token_state;  // emit S after EVERY token (spec-decode verify slots)
    // The state output IS initial_state, written in place: a [Bs, HV, K, V] tensor whose rows [0:B] are the batch.
    // Head h's tiles are its pages h*kv .. h*kv + kv - 1 in either rank, so the kernels address it as [BH, K, V].
    bool inplace_final_state;
    bool use_qk_l2norm_in_kernel;  // raw q, k rows in; the kernel folds 1/|k| and scale/|q| into its passes
    float scale;                   // the query scale, folded into the kernel with the norm
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

// The input tensors are the layer's own: TILE layout, 32x32 tiles, interleaved (DRAM or L1), bf16 or fp32 (q, k and
// v of one dtype; g and beta each bf16 or fp32). The reader gathers each core's rows and does the GQA mapping; the
// compute kernel does the recurrence, exp(g), the optional L2 norm and the q-scale included.
//   q, k : [B, T, H, K]   (raw with use_qk_l2norm_in_kernel, else L2-normalized over K by the caller; never scaled)
//   v    : [B, T, HV, V]
//   g    : [B, T, HV]     (the log decay g_t)
//   beta : [B, T, HV]
//   initial_state: the caller's [Bs, HV, K, V] fp32 with Bs*HV >= BH (head h's tiles are its pages h*kv..), or
//                  absent (zeros built in-kernel); with inplace_final_state it is also the state output.
// Outputs: o [B, T, HV, V] in the input dtype (the caller's output_tensor when given); the state [B, HV, K, V] (final),
// [B, T, HV, K, V] (per token), the ring (ring mode) or initial_state itself (in place).
//
// "Ring" mode (initial_state_block_idx present): initial_state is the [BH*T, K, V] ring and also the state output;
// see the Python binding for the contract.
struct FusedRecurrentGatedDeltaRuleInputs {
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor g;
    Tensor beta;
    std::optional<Tensor> initial_state;
    std::optional<Tensor> initial_state_block_idx;
    std::optional<Tensor> output_tensor;  // the caller's o, [B, T, HV, V] in the input dtype; the output aliases it
};

}  // namespace ttnn::prim
