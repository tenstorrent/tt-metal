// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/unified_routed_expert_ffn_types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu {

using unified_routed_expert_ffn::RoutedExpertActivation;

struct OperationArguments {
    uint32_t experts_per_chip = 1;
    uint32_t m_tiles = 0;
    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    bool read_x_at_offset = false;
    RoutedExpertActivation activation = RoutedExpertActivation::Silu;
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::BFLOAT8_B;
    tt::tt_metal::MemoryConfig output_memory_config{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "experts_per_chip",
        "m_tiles",
        "grid_x",
        "grid_y",
        "read_x_at_offset",
        "activation",
        "output_dtype",
        "output_memory_config",
        "compute_kernel_config");

    auto attribute_values() const {
        return std::forward_as_tuple(
            experts_per_chip,
            m_tiles,
            grid_x,
            grid_y,
            read_x_at_offset,
            activation,
            output_dtype,
            output_memory_config,
            compute_kernel_config);
    }
};

struct TensorArguments {
    Tensor activations;
    // One weight tensor per local expert. Expert 0 is the layout representative:
    // the program is built once and the kernels reuse a single accessor layout
    // descriptor per role, varying only the per-expert base address.
    std::vector<Tensor> w_gates;
    std::vector<Tensor> w_ups;
    std::vector<Tensor> w_downs;
    Tensor counts;
    Tensor global_expert_idx_table;
    std::optional<Tensor> optional_output;
    std::optional<Tensor> expert_region_offsets;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu
