// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "device/moe_fused_swiglu_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu {

// Fused routed-expert SwiGLU using a rectangular reduce-scatter and resident-or-streamed W_down
// schedule. By default the complete compute-with-storage grid is used; core_grid selects an
// explicit rectangular prefix. The actual token count remains device-resident:
// counts[global_expert_idx_table[local_expert_id]].
ttnn::Tensor moe_fused_swiglu(
    const ttnn::Tensor& activations,
    const ttnn::Tensor& w_gate,
    const ttnn::Tensor& w_up,
    const ttnn::Tensor& w_down,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    uint32_t local_expert_id,
    const std::optional<uint32_t>& input_m_tiles = std::nullopt,
    const std::optional<tt::tt_metal::DataType>& dtype = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<tt::tt_metal::CoreCoord>& core_grid = std::nullopt,
    const std::optional<ttnn::Tensor>& output = std::nullopt,
    const std::optional<ttnn::Tensor>& expert_region_offsets = std::nullopt,
    bool read_x_at_offset = false,
    RoutedExpertActivation activation = RoutedExpertActivation::Silu);

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu

namespace ttnn {
using operations::experimental::deepseek_prefill::moe_fused_swiglu::moe_fused_swiglu;
}  // namespace ttnn
