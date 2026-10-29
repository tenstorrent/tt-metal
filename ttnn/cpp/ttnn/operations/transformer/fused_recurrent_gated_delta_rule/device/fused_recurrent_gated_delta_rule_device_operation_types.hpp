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

// Everything that shapes the program is an attribute here and therefore part of the program-cache hash
// (the device-operation adapter hashes the attribute struct and the tensor specs when an op defines no custom
// hash). The geometry fields are carried from the start so the v2 stages that use them (V-split units, units
// looped per core) change the factory, not the hash.
struct FusedRecurrentGatedDeltaRuleParams {
    uint32_t B;        // sequences in the call (bucket width)
    uint32_t HV;       // value heads per device
    uint32_t H;        // key heads per device; HV % H == 0, value head hv reads key head hv / (HV / H)
    uint32_t BH;       // B * HV: the heads of the call (one Tensix core per head in this build)
    uint32_t T;        // tokens processed sequentially per head
    uint32_t key_dim;  // K (multiple of 32)
    uint32_t val_dim;  // V (multiple of 32)
    // Geometry of the v2 plan: NV V-slices per head and the units looped per core. This build runs one whole head
    // per core, so both are 1; stage 2 of the plan adds NV in {2, 4} and units_per_core > 1.
    uint32_t num_slices;
    uint32_t units_per_core;
    bool output_final_state;
    bool output_per_token_state;  // emit S after EVERY token (spec-decode verify slots)
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
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
