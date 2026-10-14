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

// Checks that need neither a plan nor a spec, cheapest first.
const char* scope_violation(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (input.storage_type() != ttnn::StorageType::DEVICE || input.buffer() == nullptr) {
        return "input is not an allocated device tensor";
    }
    // The kernels access NoC-filled L1 through cached RISC pointers and copy within L1 with NoC reads from the core's
    // own L1; Quasar keeps neither coherent, so native runs there.
    if (input.device()->arch() == tt::ARCH::QUASAR) {
        return "the codegen kernels do not support Quasar";
    }
    // The program moves bytes, so any 2 or 4 B type the kernels copy in halfwords or words works alike.
    switch (input.dtype()) {
        case DataType::BFLOAT16:
        case DataType::FLOAT32:
        case DataType::INT32:
        case DataType::UINT32:
        case DataType::UINT16: break;
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
    if (input.layout() != Layout::TILE || !standard_tile_with_tight_padding(input)) {
        return "input is not TILE with 32x32 tiles padded only in the last two dims";
    }
    if (!is_tile_row_permutation(in_shape, logical_output_shape)) {
        return "input or output last dim is not a multiple of 32";
    }
    if (!supported_execution_controls(input, sub_core_grids)) {
        return "sub_core_grids is not a non-empty subset of the compute grid";
    }
    return nullptr;
}

// Gates on the smallest plan, not the full one: the program factory shrinks the plan to what live L1 holds, so the
// call is in scope whenever any plan fits the static window.
const char* static_fit_violation(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const TileRowsChoice& choice) {
    const uint32_t tile_bytes =
        input.tensor_spec().tile().get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(input.dtype()));
    const uint64_t budget = static_l1_budget(input, output_spec(input, logical_output_shape, output_mem_config));
    const auto fits = [&](const TileRowsChoice& c) { return tile_rows_l1_bytes(c, tile_bytes) <= budget; };
    if (fits(choice) || plan_tile_rows_fitting(input.logical_shape(), logical_output_shape, fits)) {
        return nullptr;
    }
    return "tile-row buffers do not fit the static L1 budget even at the smallest plan";
}

constexpr const char* kPlanCapsViolation = "tile-row plan exceeds its size caps";

// Native reshapes one output tile per core per round and reads every source tile it touches; tile rows reads each
// source once per batch. Tile rows wins on any of three clauses: native reads at least 8x what the batches read,
// the batches save at least 16 reads per output tile, or every output tile is whole face halves of its sources.
// Both thresholds are judged on the full plan. Fitted on Blackhole.
constexpr uint64_t kTileRowsMinReadCut = 8;
// Source reads tile rows saves per output tile that pay for copying its rows one run at a time, whatever the ratio.
// Width changes below the read cut save at most 6 and lose; a merge of a few heads gathers each output tile from 32
// padded source tiles and saves 24. Fitted on Blackhole, holds on Wormhole.
constexpr uint64_t kTileRowsMinReadsSavedPerTile = 16;
// Two runs per output tile means every output tile is whole face halves of its sources.
constexpr uint64_t kWholeHalfRunsPerTile = 2;

bool native_faster(const Tensor& input, const TileRowsChoice& choice, uint32_t num_cores) {
    if (input.dtype() == DataType::UINT16) {
        return false;
    }
    // A width change copies one row per run and loses below the read cut unless the reads it saves outweigh the runs;
    // a refold of whole face halves wins once native needs more than one round.
    const uint64_t out_pages = choice.layout.num_output_pages;
    const bool read_cut = choice.native_reads >= kTileRowsMinReadCut * choice.batch_reads;
    const bool reads_saved = choice.native_reads >= choice.batch_reads + kTileRowsMinReadsSavedPerTile * out_pages;
    const bool whole_halves = choice.runs <= kWholeHalfRunsPerTile * out_pages && out_pages > num_cores;
    return !read_cut && !reads_saved && !whole_halves;
}

}  // namespace

const char* call_contract_violation(
    const Tensor& input,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const std::optional<ttnn::PadValue>& pad_value) {
    if (padded_shape != logical_shape || logical_shape.volume() != input.logical_volume()) {
        return "explicit padded shape or volume change";
    }
    const bool padded_tile_output = logical_shape.rank() >= 2 && (logical_shape[-1] % tt::constants::TILE_WIDTH != 0 ||
                                                                  logical_shape[-2] % tt::constants::TILE_HEIGHT != 0);
    if (input.layout() == Layout::TILE && pad_value.has_value() && padded_tile_output) {
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
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (const char* violation = scope_violation(input, logical_output_shape, output_mem_config, sub_core_grids)) {
        return violation;
    }
    const auto choice = plan_tile_rows(input.logical_shape(), logical_output_shape);
    if (!choice) {
        return kPlanCapsViolation;
    }
    return static_fit_violation(input, logical_output_shape, output_mem_config, *choice);
}

bool supported_by_codegen(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    return unsupported_reason(input, logical_output_shape, output_mem_config, sub_core_grids) == nullptr;
}

bool supported_by_codegen(const Tensor& input, const ttnn::Shape& logical_output_shape) {
    return supported_by_codegen(input, logical_output_shape, input.memory_config());
}

bool is_demoted(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (input.storage_type() != ttnn::StorageType::DEVICE || input.layout() != Layout::TILE) {
        return false;
    }
    const auto choice = plan_tile_rows(input.logical_shape(), logical_output_shape);
    return choice.has_value() && native_faster(input, *choice, reshape_core_grid(input, sub_core_grids).num_cores());
}

std::optional<LaunchPlan> routed_launch_plan(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (scope_violation(input, logical_output_shape, output_mem_config, sub_core_grids) != nullptr) {
        return std::nullopt;
    }
    auto choice = plan_tile_rows(input.logical_shape(), logical_output_shape);
    // Demotion is settled before the static fit, so a native-routed call builds no output spec.
    if (!choice || native_faster(input, *choice, reshape_core_grid(input, sub_core_grids).num_cores()) ||
        static_fit_violation(input, logical_output_shape, output_mem_config, *choice) != nullptr) {
        return std::nullopt;
    }
    // Native needs only a few CB pages, so a call whose smallest plan does not fit live L1 runs there instead of
    // reaching the program factory's no-plan error. The output is reserved because the op allocates it before the
    // program factory samples live L1.
    const auto reserved = pending_output_l1_bytes(input, output_spec(input, logical_output_shape, output_mem_config));
    if (!live_tile_rows_choice(input, logical_output_shape, *choice, reserved)) {
        return std::nullopt;
    }
    return LaunchPlan{.tile_rows = std::move(*choice)};
}

LaunchPlan forced_launch_plan(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    const char* violation = scope_violation(input, logical_output_shape, output_mem_config, sub_core_grids);
    TT_FATAL(violation == nullptr, "reshape_force_codegen: outside the codegen scope: {}", violation);
    auto choice = plan_tile_rows(input.logical_shape(), logical_output_shape);
    TT_FATAL(choice.has_value(), "reshape_force_codegen: outside the codegen scope: {}", kPlanCapsViolation);
    violation = static_fit_violation(input, logical_output_shape, output_mem_config, *choice);
    TT_FATAL(violation == nullptr, "reshape_force_codegen: outside the codegen scope: {}", violation);
    return LaunchPlan{.tile_rows = std::move(*choice)};
}

}  // namespace ttnn::operations::data_movement::reshape_codegen
