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

uint64_t static_l1_budget(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec) {
    auto* device = input.device();
    // The same ceiling the live side caps at (available_program_l1_capacity): the bank already excludes the allocator
    // base and any reserved L1_SMALL region.
    uint64_t budget = device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1);

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

uint64_t pending_output_l1_bytes(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec) {
    // require_constructible: output_spec is the spec the op allocates, so a failure to build it is a bug, not a
    // reason to reserve nothing and route a call the factory then rejects.
    return get_pending_l1_output_reservation(
        input,
        output_spec.padded_shape(),
        output_spec.memory_config(),
        output_spec.data_type(),
        output_spec.layout(),
        /*require_constructible=*/true);
}

std::optional<TileRowsChoice> live_tile_rows_choice(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const TileRowsChoice& planned,
    uint64_t reserved_l1_bytes) {
    const uint32_t tile_bytes =
        input.tensor_spec().tile().get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(input.dtype()));
    const uint64_t max_l1 = get_max_l1_space(input);
    const uint64_t free_l1 = max_l1 > reserved_l1_bytes ? max_l1 - reserved_l1_bytes : 0;
    const auto fits = [&](const TileRowsChoice& choice) { return tile_rows_l1_bytes(choice, tile_bytes) <= free_l1; };
    if (fits(planned)) {
        return planned;
    }
    return plan_tile_rows_fitting(input.logical_shape(), logical_output_shape, fits);
}

}  // namespace ttnn::operations::data_movement::reshape_codegen
