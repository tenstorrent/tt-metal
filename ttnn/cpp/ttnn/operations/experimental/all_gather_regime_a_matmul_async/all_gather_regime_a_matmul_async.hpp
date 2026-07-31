// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/all_gather_regime_a_matmul_async/device/all_gather_regime_a_matmul_async_device_operation_types.hpp"

namespace ttnn::operations::experimental::all_gather_regime_a_matmul_async {

// Re-export the config type.
using RegimeAMatmulConfig = ttnn::experimental::prim::RegimeAMatmulConfig;

}  // namespace ttnn::operations::experimental::all_gather_regime_a_matmul_async

namespace ttnn::experimental {

// Single-output Regime-A matmul with optional fusions:
//   - bias:       Y = A@B + bias                          (bias [.., 1, N] / [.., N])
//   - activation: Y = activation(A@B + bias)              (UnaryWithParam; bias applied first)
//   - addcmul:    Y = residual + scalar*(A@B + bias)*gate (residual [M,N], gate [1,N]/[M,N])
// activation and addcmul are mutually exclusive. For Pk>1 (split-K) the fusion is applied EXACTLY ONCE
// after the partials are reduced (at the reduction root band), never per-partial.
// Numerics are FIXED (BF16 in/out, HiFi2, FP32 dest-accumulation, DRAM-interleaved output) — there are no
// dtype / memory_config / compute_kernel_config arguments (they were previously accepted but ignored).
// in0 is [M, K/TP], K-sharded over an even TP group along `cluster_axis`; in1 is the device-local FULL
// [K, N]. Returns all_gather(in0, dim=-1) @ in1, REPLICATED across the TP group.
//
// PHASE 0 (current): implemented as a composition -- a standalone all-gather materialises [M, K] into
// `persistent_output_buffer`, then the single-chip Regime-A matmul runs on it. Correct, no overlap. This
// is baseline 3 ("unfused all-gather+matmul") from REGIME_A_AGMM_DESIGN_SPEC.md and the oracle the fused
// implementation will be diffed against.
ttnn::Tensor all_gather_regime_a_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::experimental::prim::RegimeAMatmulConfig>& config = std::nullopt,
    const std::optional<ttnn::Tensor>& bias_tensor = std::nullopt,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation = std::nullopt,
    std::optional<float> fused_ternary_scalar = std::nullopt,
    const std::optional<ttnn::Tensor>& fused_ternary_input_a = std::nullopt,
    const std::optional<ttnn::Tensor>& fused_ternary_input_b = std::nullopt,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore = {},
    const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
    const std::optional<ttnn::Tensor>& persistent_output_buffer = std::nullopt,
    uint32_t num_links = 1,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<uint32_t> cluster_axis = std::nullopt);

// Output column-split sibling (mirrors minimal_matmul_split): returns `chunks` equal-width [.., M, N/chunks]
// output tensors written directly (no full-output materialize + slice). `dim` must be -1 (kept for API
// compatibility, validated in the wrapper, not forwarded); N%chunks==0 and N/chunks tile-aligned. Fusions
// compose with chunking. Fixed numerics as above.
std::vector<ttnn::Tensor> all_gather_regime_a_matmul_async_split(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    int32_t chunks,
    int32_t dim,
    const std::optional<const ttnn::experimental::prim::RegimeAMatmulConfig>& config = std::nullopt,
    const std::optional<ttnn::Tensor>& bias_tensor = std::nullopt,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation = std::nullopt,
    std::optional<float> fused_ternary_scalar = std::nullopt,
    const std::optional<ttnn::Tensor>& fused_ternary_input_a = std::nullopt,
    const std::optional<ttnn::Tensor>& fused_ternary_input_b = std::nullopt,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore = {},
    const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
    const std::optional<ttnn::Tensor>& persistent_output_buffer = std::nullopt,
    uint32_t num_links = 1,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<uint32_t> cluster_axis = std::nullopt);

}  // namespace ttnn::experimental
