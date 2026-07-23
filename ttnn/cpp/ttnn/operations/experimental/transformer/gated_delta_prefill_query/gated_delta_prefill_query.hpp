// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental {

// Gated DeltaNet: prefill the recurrent state over a K/V sequence, then query.
//
// Runs the gated delta-rule recurrence over the `seq_len` K/V tokens starting from
// `state`, using a per-head constant decay `g` and write-strength `beta`, then applies
// the single query `q` to the final state to emit the first decode output token.
//
// EXPERIMENTAL / SCAFFOLDING: the current kernels wire the data path end-to-end but do
// not yet implement the recurrence (state' is a passthrough copy of `state`; O is a
// placeholder). The interface and shapes are final.
//
// Args (Nk=16, Nv=48, d=128 for Qwen3.6-27B; the op is general in Nk/Nv/seq/d):
//   q     : [1, 1,  Nk, d]  ROW_MAJOR bf16 — single query token
//   k     : [1, Nk, S,  d]  TILE bf16
//   v     : [1, Nv, S,  d]  TILE bf16
//   gate  : [1, Nv, 1,  1]  TILE fp32 — beta (write strength), scalar per V-head
//   decay : [1, Nv, 1,  1]  TILE fp32 — g (log-space decay), scalar per V-head
//   state : [1, Nv, d,  d]  TILE fp32 — recurrent state
//
// Returns:
//   O      : [1, 1,  Nv, d]  TILE bf16 — first output token
//   state' : [1, Nv, d,  d]  TILE fp32 — updated recurrent state
std::tuple<ttnn::Tensor, ttnn::Tensor> gated_delta_prefill_query(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& decay,
    const ttnn::Tensor& state,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental
