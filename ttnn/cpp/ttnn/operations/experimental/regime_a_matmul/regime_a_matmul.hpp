// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_device_operation_types.hpp"

namespace ttnn::operations::experimental::regime_a_matmul {

// Re-export the config type.
using RegimeAMatmulConfig = ttnn::experimental::prim::RegimeAMatmulConfig;

}  // namespace ttnn::operations::experimental::regime_a_matmul

namespace ttnn::experimental {

// Single-output Regime-A matmul with optional fusions:
//   - bias:       Y = A@B + bias                          (bias [.., 1, N] / [.., N])
//   - activation: Y = activation(A@B + bias)              (UnaryWithParam; bias applied first)
//   - addcmul:    Y = residual + scalar*(A@B + bias)*gate (residual [M,N], gate [1,N]/[M,N])
// activation and addcmul are mutually exclusive. For Pk>1 (split-K) the fusion is applied EXACTLY ONCE
// after the partials are reduced (at the reduction root band), never per-partial.
// Numerics are FIXED (BF16 in/out, HiFi2, FP32 dest-accumulation, DRAM-interleaved output) — there are no
// dtype / memory_config / compute_kernel_config arguments (they were previously accepted but ignored).
ttnn::Tensor regime_a_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::experimental::prim::RegimeAMatmulConfig>& config = std::nullopt,
    const std::optional<ttnn::Tensor>& bias_tensor = std::nullopt,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation = std::nullopt,
    std::optional<float> fused_ternary_scalar = std::nullopt,
    const std::optional<ttnn::Tensor>& fused_ternary_input_a = std::nullopt,
    const std::optional<ttnn::Tensor>& fused_ternary_input_b = std::nullopt);

// Output column-split sibling (mirrors minimal_matmul_split): returns `chunks` equal-width [.., M, N/chunks]
// output tensors written directly (no full-output materialize + slice). `dim` must be -1 (kept for API
// compatibility, validated in the wrapper, not forwarded); N%chunks==0 and N/chunks tile-aligned. Fusions
// compose with chunking. Fixed numerics as above.
std::vector<ttnn::Tensor> regime_a_matmul_split(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    int32_t chunks,
    int32_t dim,
    const std::optional<const ttnn::experimental::prim::RegimeAMatmulConfig>& config = std::nullopt,
    const std::optional<ttnn::Tensor>& bias_tensor = std::nullopt,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation = std::nullopt,
    std::optional<float> fused_ternary_scalar = std::nullopt,
    const std::optional<ttnn::Tensor>& fused_ternary_input_a = std::nullopt,
    const std::optional<ttnn::Tensor>& fused_ternary_input_b = std::nullopt);

}  // namespace ttnn::experimental
