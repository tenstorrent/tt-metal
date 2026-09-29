// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "device/hybrid_routed_expert_ffn_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn {

using unified::RoutedExpertActivation;

// The merged op: ONE dispatch carrying both routed-expert implementations.
//
// `hybrid_token_threshold` is the model's measured split, applied per expert against the
// device-resident token counts -- at or below it an expert runs on the fused implementation,
// above it on the unified one. Zero leaves every expert on the unified half.
//
// This replaces dispatching moe_fused_swiglu and unified_routed_expert_moe back to back, which is
// what lets the layer be overlapped with the combine op downstream.
ttnn::Tensor hybrid_routed_expert_moe(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& global_expert_idx_table,
    const std::vector<ttnn::Tensor>& gate_projs,
    const std::vector<ttnn::Tensor>& up_projs,
    const std::vector<ttnn::Tensor>& down_projs,
    uint32_t max_dispatched_tokens_per_expert,
    uint32_t hybrid_token_threshold = 0,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    RoutedExpertActivation activation = RoutedExpertActivation::Silu,
    const std::optional<std::vector<ttnn::Tensor>>& gate_biases = std::nullopt,
    const std::optional<std::vector<ttnn::Tensor>>& up_biases = std::nullopt,
    const std::optional<std::vector<ttnn::Tensor>>& down_biases = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn

namespace ttnn {
using operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::hybrid_routed_expert_moe;
}  // namespace ttnn
