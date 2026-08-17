// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu {

struct OperationArguments {
    uint32_t local_expert_id = 0;
    uint32_t m_tiles = 0;
    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    bool read_x_at_offset = false;
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::BFLOAT8_B;
    tt::tt_metal::MemoryConfig output_memory_config{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "local_expert_id",
        "m_tiles",
        "grid_x",
        "grid_y",
        "read_x_at_offset",
        "output_dtype",
        "output_memory_config",
        "compute_kernel_config");

    auto attribute_values() const {
        return std::forward_as_tuple(
            local_expert_id,
            m_tiles,
            grid_x,
            grid_y,
            read_x_at_offset,
            output_dtype,
            output_memory_config,
            compute_kernel_config);
    }
};

struct TensorArguments {
    Tensor activations;
    Tensor w_gate;
    Tensor w_up;
    Tensor w_down;
    Tensor counts;
    Tensor global_expert_idx_table;
    std::optional<Tensor> optional_output;
    std::optional<Tensor> expert_region_offsets;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu
