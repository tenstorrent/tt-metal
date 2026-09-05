// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "device/moe_fused_swiglu_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu {

// Fused routed-expert SwiGLU over ALL local experts in ONE device program: the reader, writer and
// compute kernels loop the experts_per_chip local experts, resolving each one's global id, token
// count and region offset device-side. Uses a rectangular reduce-scatter and a resident-or-streamed
// W_down schedule. By default the complete compute-with-storage grid is used; core_grid selects an
// explicit rectangular prefix. Every token count stays device-resident:
// counts[global_expert_idx_table[e]] — no host sync, no per-expert dispatch.
//
// An expert whose count is zero costs no CB traffic, no collective round and no semaphore: the
// skip is uniform across the grid, so a masked counts vector is a valid way to route a subset of
// experts to this op and the rest elsewhere.
ttnn::Tensor moe_fused_swiglu(
    const ttnn::Tensor& activations,
    const std::vector<ttnn::Tensor>& w_gates,
    const std::vector<ttnn::Tensor>& w_ups,
    const std::vector<ttnn::Tensor>& w_downs,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    const std::optional<uint32_t>& input_m_tiles = std::nullopt,
    const std::optional<tt::tt_metal::DataType>& dtype = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<tt::tt_metal::CoreCoord>& core_grid = std::nullopt,
    const std::optional<ttnn::Tensor>& output = std::nullopt,
    const std::optional<ttnn::Tensor>& expert_region_offsets = std::nullopt,
    bool read_x_at_offset = false,
    RoutedExpertActivation activation = RoutedExpertActivation::Silu,
    // Active-token band this op owns: an expert whose count falls outside [min, max] is
    // dropped like a zero count. Wide open by default; a hybrid dispatch narrows it so this
    // op and unified_routed_expert_moe split the experts by load over ONE counts vector.
    uint32_t min_active_tokens = 0,
    uint32_t max_active_tokens = std::numeric_limits<uint32_t>::max(),
    // BFLOAT8_B (default, the original) or BFLOAT16: format of the gate/up K-partials and the
    // reduce-scatter landing buffers. BFLOAT16 matches the composite op's bf16 partial sums.
    const std::optional<tt::tt_metal::DataType>& intermediate_dtype = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu

namespace ttnn {
using operations::experimental::deepseek_prefill::moe_fused_swiglu::moe_fused_swiglu;
}  // namespace ttnn
