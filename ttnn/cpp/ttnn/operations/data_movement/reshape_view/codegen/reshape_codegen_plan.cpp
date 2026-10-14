// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_plan.hpp"

#include <tt-metalium/hal.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

tt::tt_metal::TensorSpec output_spec(
    const Tensor& input, const ttnn::Shape& logical_output_shape, const tt::tt_metal::MemoryConfig& output_mem_config) {
    return tt::tt_metal::TensorSpec(
        logical_output_shape,
        tt::tt_metal::TensorLayout(
            input.dtype(), tt::tt_metal::PageConfig(input.layout(), input.tensor_spec().tile()), output_mem_config));
}

tt::tt_metal::CoreRangeSet reshape_core_grid(
    const Tensor& input, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (sub_core_grids.has_value()) {
        return *sub_core_grids;
    }
    const auto grid = input.device()->compute_with_storage_grid_size();
    return tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
}

uint32_t reshape_core_count(const Tensor& input, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (sub_core_grids.has_value()) {
        return static_cast<uint32_t>(sub_core_grids->num_cores());
    }
    const auto grid = input.device()->compute_with_storage_grid_size();
    return static_cast<uint32_t>(grid.x * grid.y);
}

uint64_t static_l1_budget(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec) {
    auto* device = input.device();
    const uint64_t base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint64_t ceiling = static_cast<uint64_t>(device->l1_size_per_core());
    uint64_t budget = ceiling > base ? ceiling - base : 0;

    const auto l1_banks = device->allocator()->get_num_banks(tt::tt_metal::BufferType::L1);
    const auto l1_alignment = device->allocator()->get_alignment(tt::tt_metal::BufferType::L1);
    const auto subtract_footprint = [&](const tt::tt_metal::TensorSpec& spec) {
        const uint64_t footprint = spec.compute_consumed_memory_bytes_per_bank(l1_alignment, l1_banks);
        budget = budget > footprint ? budget - footprint : 0;
    };
    if (input.memory_config().buffer_type() == tt::tt_metal::BufferType::L1) {
        subtract_footprint(input.tensor_spec());
    }
    if (output_spec.memory_config().buffer_type() == tt::tt_metal::BufferType::L1) {
        subtract_footprint(output_spec);
    }
    return budget;
}

uint64_t live_l1_budget(const Tensor& input) { return get_max_l1_space(input); }

}  // namespace ttnn::operations::data_movement::reshape_codegen
