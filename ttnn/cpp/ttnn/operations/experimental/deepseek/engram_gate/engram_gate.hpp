// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek {

// Fused Engram gate (deepseek_v41_flash `DeepSeekV41Engram.forward`, after the wkv matmul).
//
// Replaces the Python sequence in engram.py:
//
//     key, value = kv[..., :hc*D] -> [B, hc, D], kv[..., hc*D:] -> [B, 1, D]
//     rstd = rsqrt(mean(x^2) + eps) * rsqrt(mean(key^2) + eps)          over D
//     dot  = sum(x * gate_weight * key) * rstd                          over D
//     gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))
//     out  = x + gate * value
//
// `streams` and the output are ROW_MAJOR WIDTH_SHARDED over D (rows = B*hc, at most 32 cores).
// `kv` is ROW_MAJOR WIDTH_SHARDED on any grid; each core NoC-reads the key / value columns it
// owns. The D-reductions are summed across cores with an all-to-all exchange of per-core partials.
//
// Args:
//   streams: [B, 1, hc, D] bf16, ROW_MAJOR L1 WIDTH_SHARDED, shard [B*hc, D / num_cores].
//   kv: [1, 1, B, (hc+1)*D] bf16, ROW_MAJOR L1 WIDTH_SHARDED.
//   gate_weight: [1, 1, 32, D] fp32, TILE L1 WIDTH_SHARDED on the streams grid, row i holding
//       q_weight * k_weight / sqrt(D) of copy i % hc (needs 32 % hc == 0).
//   eps: RMS epsilon.
//
// Returns: same shape, dtype and memory config as `streams`.
Tensor engram_gate(
    const Tensor& streams,
    const Tensor& kv,
    const Tensor& gate_weight,
    float eps,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental::deepseek
