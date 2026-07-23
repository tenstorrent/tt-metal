// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

// Non-tensor configuration for the gated-delta prefill-then-query op.
//
// Semantics (target — NOT yet implemented by the placeholder kernels):
//   Starting from `state` (per V-head d_k x d_v recurrent matrix), run the gated
//   delta-rule recurrence over the `seq_len` K/V tokens using a per-head constant
//   decay `g` (log-space) and write-strength `beta`, then apply the single query
//   `q` to the final state to emit the first decode output token.
struct GatedDeltaPrefillQueryParams {
    uint32_t num_k_heads = 0;  // Nk — Q/K heads (key side)
    uint32_t num_v_heads = 0;  // Nv — V heads / number of per-head states
    uint32_t seq_len = 0;      // S  — number of prefill K/V tokens
    uint32_t head_dim = 0;     // d  — head_k_dim == head_v_dim
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

// Input tensors. Shapes / layouts (Nk=16, Nv=48, d=128 for Qwen3.6-27B):
//   q     : [1, 1,  Nk, d ]  ROW_MAJOR, bf16  — single query token
//   k     : [1, Nk, S,  d ]  TILE,      bf16
//   v     : [1, Nv, S,  d ]  TILE,      bf16
//   gate  : [1, Nv, 1,  1 ]  TILE,      fp32  — beta (write strength), scalar per head
//   decay : [1, Nv, 1,  1 ]  TILE,      fp32  — g (log-space decay), scalar per head
//   state : [1, Nv, d,  d ]  TILE,      fp32  — recurrent state, d_k x d_v per head
struct GatedDeltaPrefillQueryInputs {
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor gate;
    Tensor decay;
    Tensor state;
};

}  // namespace ttnn::experimental::prim
