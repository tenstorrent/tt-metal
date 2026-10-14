// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_supported.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_plan.hpp"
#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_tile_rows_plan.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

namespace {

bool is_plain_interleaved(const tt::tt_metal::MemoryConfig& config) {
    const bool on_supported_buffer =
        config.buffer_type() == tt::tt_metal::BufferType::DRAM || config.buffer_type() == tt::tt_metal::BufferType::L1;
    return config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED && !config.is_sharded() &&
           !config.nd_shard_spec().has_value() && on_supported_buffer;
}

bool standard_tile_with_tight_padding(const Tensor& input) {
    const auto& tile = input.tensor_spec().tile();
    if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH ||
        tile.get_transpose_of_faces() || tile.get_transpose_within_face()) {
        return false;
    }
    const auto& logical = input.logical_shape();
    const auto& padded = input.padded_shape();
    if (logical.rank() < 2 || padded.rank() != logical.rank()) {
        return false;
    }
    for (size_t i = 0; i + 2 < logical.rank(); ++i) {
        if (padded[i] != logical[i]) {
            return false;
        }
    }
    return padded[-1] == tt::round_up(logical[-1], tile.get_width()) &&
           padded[-2] == tt::round_up(logical[-2], tile.get_height());
}

// Whether native's tiled reshape measures faster than the tile-rows program (thresholds fitted on Blackhole). Native
// rejects uint16.
bool tile_native_faster(const Tensor& input, const ttnn::Shape& logical_output_shape, uint32_t num_cores) {
    return input.dtype() != DataType::UINT16 &&
           !tile_rows_beats_native(input.logical_shape(), logical_output_shape, num_cores);
}

// Every check of the gate that needs no plan. None builds a spec or reads the allocator, so they run before
// anything that does, and the core grid is only built for a call that passes the rest.
bool in_scope(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (input.storage_type() != ttnn::StorageType::DEVICE || input.buffer() == nullptr) {
        return false;
    }
    // Every program moves bytes, so any 2 or 4 B type the kernels copy in halfwords or words works alike.
    switch (input.dtype()) {
        case DataType::BFLOAT16:
        case DataType::FLOAT32:
        case DataType::INT32:
        case DataType::UINT32:
        case DataType::UINT16: break;
        default: return false;
    }
    if (!is_plain_interleaved(input.memory_config()) || !is_plain_interleaved(output_mem_config)) {
        return false;
    }
    const auto& in_shape = input.logical_shape();
    if (in_shape == logical_output_shape || in_shape.volume() == 0 ||
        in_shape.volume() != logical_output_shape.volume()) {
        return false;
    }
    if (input.layout() != Layout::TILE || !standard_tile_with_tight_padding(input) ||
        !is_tile_row_permutation(in_shape, logical_output_shape)) {
        return false;
    }
    return supported_execution_controls(input, sub_core_grids);
}

// The budget-independent half of the program, built once per call and fitted to both the static and the live budget.
class Geometry {
public:
    Geometry(
        const Tensor& input,
        const ttnn::Shape& logical_output_shape,
        const tt::tt_metal::MemoryConfig& output_mem_config) :
        input_(input),
        logical_output_shape_(logical_output_shape),
        output_spec_(reshape_codegen::output_spec(input, logical_output_shape, output_mem_config)) {}

    const tt::tt_metal::TensorSpec& output_spec() const { return output_spec_; }

    bool tile_rows_fit(uint64_t l1_budget) const { return tile_rows_fits(input_, logical_output_shape_, l1_budget); }

private:
    const Tensor& input_;
    const ttnn::Shape& logical_output_shape_;
    tt::tt_metal::TensorSpec output_spec_;
};

}  // namespace

bool supported_execution_controls(
    const Tensor& input, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (!sub_core_grids.has_value()) {
        return true;
    }
    const auto grid = input.device()->compute_with_storage_grid_size();
    const tt::tt_metal::CoreRangeSet compute_grid(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    return sub_core_grids->num_cores() > 0 && compute_grid.contains(*sub_core_grids);
}

CodegenDecision decide_codegen(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    bool routing,
    bool plan_launch) {
    CodegenDecision decision;
    if (!in_scope(input, logical_output_shape, output_mem_config, sub_core_grids)) {
        return decision;
    }
    // Routing runs native whenever native is faster, admitted or not, so that is settled first: it needs only the
    // memoized plan layout and the core count, and a native-routed call builds no spec and reads no allocator state.
    if (routing && tile_native_faster(input, logical_output_shape, reshape_core_count(input, sub_core_grids))) {
        decision.native_faster = true;
        return decision;
    }
    Geometry geometry(input, logical_output_shape, output_mem_config);
    const uint64_t static_budget = static_l1_budget(input, geometry.output_spec());
    if (!geometry.tile_rows_fit(static_budget)) {
        return decision;
    }
    decision.supported = true;
    if (!plan_launch) {
        return decision;
    }

    // The launch allocates the output before the program runs, so hold its reservation back now.
    const auto& spec = geometry.output_spec();
    const uint64_t reservation =
        get_pending_l1_output_reservation(input, spec.padded_shape(), output_mem_config, input.dtype(), input.layout());
    const uint64_t live = live_l1_budget(input);
    LaunchPlan plan{.l1_budget = live > reservation ? live - reservation : 0, .output_spec = spec};
    if (!geometry.tile_rows_fit(plan.l1_budget)) {
        return decision;
    }
    decision.plan = std::move(plan);
    return decision;
}

bool supported_by_codegen(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    return decide_codegen(
               input, logical_output_shape, output_mem_config, sub_core_grids, /*routing=*/false, /*plan_launch=*/false)
        .supported;
}

bool supported_by_codegen(const Tensor& input, const ttnn::Shape& logical_output_shape) {
    return supported_by_codegen(input, logical_output_shape, input.memory_config());
}

}  // namespace ttnn::operations::data_movement::reshape_codegen
