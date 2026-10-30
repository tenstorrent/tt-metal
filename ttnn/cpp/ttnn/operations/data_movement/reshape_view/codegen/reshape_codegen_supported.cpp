// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_supported.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

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
    // The writers walk 16x16 faces; a 32x32 tile built with another face shape lays its datums out differently.
    const auto face = tile.get_face_shape();
    if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH ||
        face[0] != tt::constants::FACE_HEIGHT || face[1] != tt::constants::FACE_WIDTH ||
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

// Why the codegen program cannot run this call before any plan is built (placement, layout, dtype, shapes, grid), or
// nullptr. Cheapest first.
const char* scope_violation(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (input.storage_type() != ttnn::StorageType::DEVICE || !input.is_allocated()) {
        return "input is not an allocated device tensor";
    }
    // buffer() throws when no device of the mesh is local; native's launch skips such a mesh instead.
    if (input.device()->is_remote_only()) {
        return "the mesh has no local devices";
    }
    // Every program moves bytes, so any 2 or 4 B type the kernels copy in halfwords or words works alike.
    switch (input.dtype()) {
        case tt::tt_metal::DataType::BFLOAT16:
        case tt::tt_metal::DataType::FLOAT32:
        case tt::tt_metal::DataType::INT32:
        case tt::tt_metal::DataType::UINT32:
        case tt::tt_metal::DataType::UINT16: break;
        default: return "dtype is not bfloat16, float32, int32, uint32 or uint16";
    }
    if (!is_plain_interleaved(input.memory_config()) || !is_plain_interleaved(output_mem_config)) {
        return "input or output is not interleaved in DRAM or L1";
    }
    const auto& in_shape = input.logical_shape();
    if (in_shape == logical_output_shape) {
        return "output shape equals the input shape";
    }
    if (in_shape.volume() == 0 || in_shape.volume() != logical_output_shape.volume()) {
        return "input is empty or the volume changes";
    }
    if (input.layout() != tt::tt_metal::Layout::TILE || !standard_tile_with_tight_padding(input)) {
        return "input is not TILE with 32x32 tiles padded only in the last two dims";
    }
    if (!supported_execution_controls(input, sub_core_grids)) {
        return "sub_core_grids is not a non-empty subset of the compute grid";
    }
    return nullptr;
}

// Native reads every source tile each output tile touches; tile rows reads each source once per batch. Thresholds
// are judged on the full plan, fitted on Blackhole and checked on Wormhole.
constexpr uint64_t kTileRowsMinReadCut = 8;
// Source reads saved per output tile that pay for copying its rows one run at a time, whatever the ratio. Width
// changes below the read cut save at most 6 and lose; a merge of a few heads saves 24.
constexpr uint64_t kTileRowsMinReadsSavedPerTile = 16;
// Two runs per output tile means every output tile is whole face halves of its sources.
constexpr uint64_t kWholeHalfRunsPerTile = 2;

bool native_faster(const TileRowsChoice& choice, uint32_t num_cores) {
    // A width change copies one row per run and loses below the read cut unless the reads it saves outweigh the runs;
    // a refold of whole face halves wins once native needs more than one round.
    const uint64_t out_pages = choice.layout.num_output_pages;
    const bool read_cut = choice.native_reads >= kTileRowsMinReadCut * choice.batch_reads;
    const bool reads_saved = choice.native_reads >= choice.batch_reads + kTileRowsMinReadsSavedPerTile * out_pages;
    const bool whole_halves = choice.runs <= kWholeHalfRunsPerTile * out_pages && out_pages > num_cores;
    return !read_cut && !reads_saved && !whole_halves;
}

constexpr const char* kTileCapsViolation = "no TILE program takes these shapes within its size caps";
constexpr const char* kTileStaticFitViolation = "no TILE program fits the static L1 budget even at its smallest plan";

}  // namespace

const char* call_contract_violation(
    const Tensor& input,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const std::optional<ttnn::PadValue>& pad_value) {
    if (padded_shape != logical_shape || logical_shape.volume() != input.logical_volume()) {
        return "explicit padded shape or volume change";
    }
    // A rank-one output is a single row, so its tile pads the height from 1 to 32.
    const bool padded_tile_output = logical_shape.rank() < 2 || logical_shape[-1] % tt::constants::TILE_WIDTH != 0 ||
                                    logical_shape[-2] % tt::constants::TILE_HEIGHT != 0;
    if (input.layout() == tt::tt_metal::Layout::TILE && pad_value.has_value() && padded_tile_output) {
        return "explicit pad_value on a padded tile output";
    }
    return nullptr;
}

bool supported_execution_controls(
    const Tensor& input, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    return !sub_core_grids.has_value() ||
           (sub_core_grids->num_cores() > 0 && reshape_core_grid(input).contains(*sub_core_grids));
}

const char* unsupported_reason(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    const LaunchPlan* plan) {
    if (const char* violation = scope_violation(input, logical_output_shape, output_mem_config, sub_core_grids)) {
        return violation;
    }
    const auto& in_shape = input.logical_shape();
    const uint32_t num_cores = reshape_core_grid(input, sub_core_grids).num_cores();
    if (plan == nullptr) {
        const bool within_caps = plan_tile_rows(in_shape, logical_output_shape).has_value() ||
                                 segment_map_within_caps(input, logical_output_shape, num_cores);
        return within_caps ? kTileStaticFitViolation : kTileCapsViolation;
    }
    const uint64_t budget = static_l1_budget(input, output_spec(input, logical_output_shape, output_mem_config));
    if (plan->tile_route == TileRoute::TileRows) {
        const uint32_t tile_bytes =
            input.tensor_spec().tile().get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(input.dtype()));
        const auto fits = [&](const TileRowsChoice& c) { return tile_rows_l1_bytes(c, tile_bytes) <= budget; };
        if (fits(plan->tile_rows) || plan_tile_rows_fitting(in_shape, logical_output_shape, fits).has_value()) {
            return nullptr;
        }
    } else if (fit_tile_mapped(input, logical_output_shape, num_cores, budget).has_value()) {
        return nullptr;
    }
    return kTileStaticFitViolation;
}

bool supported_by_codegen(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    const LaunchPlan& plan) {
    return unsupported_reason(input, logical_output_shape, output_mem_config, sub_core_grids, &plan) == nullptr;
}

std::optional<LaunchPlan> plan_codegen(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (scope_violation(input, logical_output_shape, output_mem_config, sub_core_grids) != nullptr) {
        return std::nullopt;
    }
    if (auto rows = plan_tile_rows(input.logical_shape(), logical_output_shape)) {
        LaunchPlan plan{.tile_route = TileRoute::TileRows, .tile_rows = std::move(*rows)};
        if (supported_by_codegen(input, logical_output_shape, output_mem_config, sub_core_grids, plan)) {
            return plan;
        }
    }
    LaunchPlan mapped{.tile_route = TileRoute::TileMapped};
    if (supported_by_codegen(input, logical_output_shape, output_mem_config, sub_core_grids, mapped)) {
        return mapped;
    }
    return std::nullopt;
}

bool segment_map_within_caps(const Tensor& input, const ttnn::Shape& logical_output_shape, uint32_t num_cores) {
    return tile_mapped_within_caps(
        input.logical_shape(),
        logical_output_shape,
        static_cast<uint32_t>(input.element_size()),
        input.tensor_spec().tile().get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(input.dtype())),
        num_cores);
}

bool is_demoted(
    const Tensor& input, const ttnn::Shape& logical_output_shape, uint32_t num_cores, const LaunchPlan& plan) {
    // Native rejects uint16, so demoting would fail the call.
    if (input.dtype() == tt::tt_metal::DataType::UINT16) {
        return false;
    }
    if (plan.tile_route == TileRoute::TileRows) {
        // Tile rows falls back to the segment map under live L1 pressure, so it beats native only while the segment
        // map is within its size caps.
        return native_faster(plan.tile_rows, num_cores) ||
               !segment_map_within_caps(input, logical_output_shape, num_cores);
    }
    // The segment map loses to native only on a tile-row permutation: both widths whole tiles.
    return input.logical_shape()[-1] % kTileDim == 0 && logical_output_shape[-1] % kTileDim == 0;
}

}  // namespace ttnn::operations::data_movement::reshape_codegen
