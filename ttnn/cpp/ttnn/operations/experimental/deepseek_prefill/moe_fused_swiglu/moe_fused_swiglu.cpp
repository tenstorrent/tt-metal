// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "moe_fused_swiglu.hpp"

#include "device/moe_fused_swiglu_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu {

ttnn::Tensor moe_fused_swiglu(
    const ttnn::Tensor& activations,
    const std::vector<ttnn::Tensor>& w_gates,
    const std::vector<ttnn::Tensor>& w_ups,
    const std::vector<ttnn::Tensor>& w_downs,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    const std::optional<uint32_t>& input_m_tiles,
    const std::optional<tt::tt_metal::DataType>& dtype,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<tt::tt_metal::CoreCoord>& core_grid,
    const std::optional<ttnn::Tensor>& output,
    const std::optional<ttnn::Tensor>& expert_region_offsets,
    bool read_x_at_offset,
    RoutedExpertActivation activation) {
    constexpr uint32_t TILE = 32;
    TT_FATAL(!w_gates.empty(), "moe_fused_swiglu: at least one local expert is required");
    TT_FATAL(
        w_gates.size() == w_ups.size() && w_gates.size() == w_downs.size(),
        "moe_fused_swiglu: gate/up/down lists must have one entry per local expert ({}, {}, {})",
        w_gates.size(),
        w_ups.size(),
        w_downs.size());
    const uint32_t experts_per_chip = static_cast<uint32_t>(w_gates.size());
    const uint32_t capacity_tiles = activations.padded_shape()[-2] / TILE;
    const uint32_t m_tiles = input_m_tiles.value_or(capacity_tiles);
    const auto device_grid = activations.device()->compute_with_storage_grid_size();
    if (core_grid.has_value()) {
        TT_FATAL(
            core_grid->x <= device_grid.x && core_grid->y <= device_grid.y,
            "moe_fused_swiglu: requested grid {}x{} exceeds device grid {}x{}",
            core_grid->x,
            core_grid->y,
            device_grid.x,
            device_grid.y);
    }
    const uint32_t grid_x = core_grid.has_value() ? core_grid->x : device_grid.x;
    const uint32_t grid_y = core_grid.has_value() ? core_grid->y : device_grid.y;

    const auto output_dtype = output.has_value() ? output->dtype() : dtype.value_or(tt::tt_metal::DataType::BFLOAT8_B);
    const auto output_memory_config =
        output.has_value() ? output->memory_config()
                           : memory_config.value_or(tt::tt_metal::MemoryConfig{
                                 tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});
    TT_FATAL(
        !output.has_value() || !dtype.has_value() || *dtype == output->dtype(),
        "moe_fused_swiglu: dtype contradicts supplied output dtype");
    TT_FATAL(
        !output.has_value() || !memory_config.has_value() || *memory_config == output->memory_config(),
        "moe_fused_swiglu: memory_config contradicts supplied output memory config");

    const auto resolved_compute_config = init_device_compute_kernel_config(
        activations.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::LoFi,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false,
        /*default_dst_full_sync_en=*/false);

    return ttnn::prim::moe_fused_swiglu(
        activations,
        w_gates,
        w_ups,
        w_downs,
        counts,
        global_expert_idx_table,
        experts_per_chip,
        m_tiles,
        grid_x,
        grid_y,
        read_x_at_offset,
        activation,
        output_dtype,
        output_memory_config,
        resolved_compute_config,
        output,
        expert_region_offsets);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu
