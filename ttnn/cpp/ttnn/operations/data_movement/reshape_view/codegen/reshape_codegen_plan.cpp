// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_plan.hpp"

#include <tt-metalium/hal.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

namespace {

uint32_t tile_bytes_of(const Tensor& input) {
    return input.tensor_spec().tile().get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(input.dtype()));
}

std::optional<Shape3> normalize(const ttnn::Shape& shape) {
    if (shape.rank() == 0) {
        return std::nullopt;
    }
    for (size_t i = 0; i < shape.rank(); ++i) {
        if (shape[i] == 0) {
            return std::nullopt;
        }
    }
    Shape3 out;
    out.w = shape[-1];
    if (shape.rank() >= 2) {
        out.h = shape[-2];
    }
    for (size_t i = 0; i + 2 < shape.rank(); ++i) {
        out.c *= shape[i];
    }
    return out;
}

std::optional<TileRowsChoice> fit_tile_rows(
    const Tensor& input, const ttnn::Shape& logical_output_shape, const TileRowsChoice& planned, uint64_t free_l1) {
    const uint32_t tile_bytes = tile_bytes_of(input);
    const auto fits = [&](const TileRowsChoice& choice) { return tile_rows_l1_bytes(choice, tile_bytes) <= free_l1; };
    if (fits(planned)) {
        return planned;
    }
    return plan_tile_rows_fitting(input.logical_shape(), logical_output_shape, fits);
}

}  // namespace

std::optional<std::pair<Shape3, Shape3>> normalize_pair(const ttnn::Shape& in_shape, const ttnn::Shape& out_shape) {
    const auto in = normalize(in_shape);
    const auto out = normalize(out_shape);
    if (!in || !out || in->c * in->h * in->w != out->c * out->h * out->w) {
        return std::nullopt;
    }
    return std::make_pair(*in, *out);
}

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

std::optional<TileRowsChoice> live_tile_rows_choice(
    const Tensor& input, const ttnn::Shape& logical_output_shape, const TileRowsChoice& planned) {
    return fit_tile_rows(input, logical_output_shape, planned, get_max_l1_space(input));
}

std::optional<TileMappedChoice> fit_tile_mapped(
    const Tensor& input, const ttnn::Shape& logical_output_shape, uint32_t grid_cores, uint64_t l1_budget) {
    const uint32_t tile_bytes = tile_bytes_of(input);
    const auto layout = tile_mapped_layout(
        input.logical_shape(),
        logical_output_shape,
        static_cast<uint32_t>(input.element_size()),
        tile_bytes,
        grid_cores,
        l1_budget);
    if (!layout.has_value()) {
        return std::nullopt;
    }
    return TileMappedChoice{.layout = *layout, .input_depth = tile_mapped_input_depth(*layout, tile_bytes, l1_budget)};
}

std::optional<TileChoice> live_tile_choice(
    const Tensor& input, const ttnn::Shape& logical_output_shape, uint32_t grid_cores, const LaunchPlan& planned) {
    const uint64_t free_l1 = get_max_l1_space(input);
    if (planned.tile_route == TileRoute::TileRows) {
        if (auto rows = fit_tile_rows(input, logical_output_shape, planned.tile_rows, free_l1)) {
            return TileChoice{std::move(*rows)};
        }
    }
    if (auto mapped = fit_tile_mapped(input, logical_output_shape, grid_cores, free_l1)) {
        return TileChoice{*mapped};
    }
    return std::nullopt;
}

}  // namespace ttnn::operations::data_movement::reshape_codegen
