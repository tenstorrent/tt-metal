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
//
// `output_dtype` picks the tilized output's dtype when x is ROW_MAJOR (bfloat8_b by default); a TILE x is
// written back in its own dtype.
//
// Passing `dispatched_metadata` overlaps combine_fabric2d in the same program: combine runs on rows
// 0-1, takes each expert as soon as it is written, and its output is what this returns. The routed
// expert's output is then an internal bfloat16 buffer, so x must be bfloat16 ROW_MAJOR. The remaining
// combine arguments are required in that mode and ignored otherwise; the index table there is the full
// (groups, extent, experts_per_chip) one, replicated on every device.
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
    const std::optional<std::vector<ttnn::Tensor>>& down_biases = std::nullopt,
    const std::optional<ttnn::Tensor>& dispatched_metadata = std::nullopt,
    const std::optional<ttnn::Tensor>& expert_offsets = std::nullopt,
    const std::optional<ttnn::Tensor>& replicated_global_expert_idx_table = std::nullopt,
    uint32_t combine_axis = 0,
    uint32_t combine_num_links = 2,
    uint32_t num_experts_per_tok = 0,
    uint32_t seq_len_per_chip = 0,
    std::optional<tt::tt_metal::DataType> output_dtype = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn

namespace ttnn {
using operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::hybrid_routed_expert_moe;
}  // namespace ttnn
