// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_supported.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <span>
#include <string_view>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement::tilize_codegen {

namespace {

using tt::tt_metal::DataType;

// Tile::operator== ignores the transpose flags, so every attribute is compared individually.
bool is_default_tile(const tt::tt_metal::Tile& tile) {
    const auto default_tile = tt::tt_metal::Tile();
    return tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH &&
           tile.get_face_shape() == default_tile.get_face_shape() && !tile.get_transpose_within_face() &&
           !tile.get_transpose_of_faces();
}

constexpr int kWidthAxis = -1;
constexpr int kHeightAxis = -2;
constexpr size_t kMinRank = 2;

constexpr uint32_t kCompactReaderMaxWt = 8;
constexpr uint32_t kBatchedWriteTiles = 4;
constexpr uint32_t kDoubleBuffered = 2;
constexpr uint32_t kSingleBuffered = 1;
// A column split needs at least this many tiles per row to amortize its extra reader geometry.
constexpr uint32_t kColumn2DMinWt = 3;
constexpr uint32_t kMinColumnBlocks = 2;
constexpr uint32_t kMinCoreColBytes = 1024;
constexpr uint32_t kMinFragmentBytes = 256;
constexpr uint32_t kTargetCores = 24;
constexpr uint32_t kMinSplitCores = 5;

std::optional<uint32_t> row_major_element_size(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16:
        case DataType::UINT16: return 2;
        case DataType::FLOAT32:
        case DataType::UINT32:
        case DataType::INT32: return 4;
        default: return std::nullopt;
    }
}

bool supported_dtype_pair(DataType input, DataType output) {
    return row_major_element_size(input).has_value() &&
           (output == input || (input == DataType::BFLOAT16 && output == DataType::INT32));
}

uint32_t largest_divisor_at_most(uint32_t n, uint32_t limit) {
    for (uint32_t d = std::min(n, limit); d > 1; --d) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

// Largest divisor of wt in [kMinColumnBlocks, max_blocks], or 1 when there is none.
uint32_t column_blocks(uint32_t wt, uint32_t max_blocks) {
    for (uint32_t d = std::min(wt, max_blocks); d >= kMinColumnBlocks; --d) {
        if (wt % d == 0) {
            return d;
        }
    }
    return 1;
}

// Most column blocks worth cutting a tile row into. Narrow DRAM fragments read inefficiently, so a few-row
// tensor gets wide blocks on few cores. A cut that would use fewer than kMinSplitCores keeps the full grid.
uint32_t max_column_blocks(const TilizeGeometry& g, uint32_t elem_size, uint32_t grid_cores) {
    const uint32_t grid_blocks = grid_cores / g.tile_rows;
    const uint32_t fragment_bytes = tt::constants::TILE_WIDTH * elem_size;
    const uint32_t align_tiles = std::max(1u, tt::tt_metal::hal::get_dram_alignment() / fragment_bytes);
    const uint32_t col_units = tt::div_up(g.wt, align_tiles);
    const uint32_t min_units = std::max(1u, tt::div_up(kMinCoreColBytes / fragment_bytes, align_tiles));
    const uint32_t wide_blocks = std::max(1u, col_units / min_units);
    const uint32_t row_bytes = g.wt * fragment_bytes;
    const uint32_t fragment_blocks =
        std::min(col_units, std::max(1u, row_bytes / std::max(kMinFragmentBytes, align_tiles * fragment_bytes)));
    const uint32_t few_row_blocks = std::min(fragment_blocks, tt::div_up(kTargetCores, g.tile_rows));
    const uint32_t capped = std::min(grid_blocks, std::max(wide_blocks, few_row_blocks));
    return g.tile_rows * capped < kMinSplitCores ? grid_blocks : capped;
}

// Everything about a case that does not depend on how much L1 is free.
struct Geometry : TilizeGeometry {
    uint32_t row_cores = 0;
    uint32_t in_tile_bytes = 0;
    uint32_t out_tile_bytes = 0;
    uint32_t pending_l1_output_bytes = 0;
};

std::optional<Geometry> analyze(
    const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config, DataType output_dtype) {
    using tt::tt_metal::Layout;
    using namespace tt::constants;

    if (input.storage_type() != ttnn::StorageType::DEVICE || !input.is_allocated() || input.device() == nullptr) {
        return std::nullopt;
    }
    if (input.layout() != Layout::ROW_MAJOR || input.is_sharded() || output_mem_config.is_sharded()) {
        return std::nullopt;
    }
    if (!supported_dtype_pair(input.dtype(), output_dtype)) {
        return std::nullopt;
    }

    // The tile size, page count and row pitch below are all 32x32 quantities; a transposed tile would
    // additionally permute the datums, which nothing here configures.
    if (!is_default_tile(input.tensor_spec().tile())) {
        return std::nullopt;
    }

    const auto& shape = input.padded_shape();
    if (shape.rank() < kMinRank || input.logical_shape() != shape || input.logical_shape().volume() == 0) {
        return std::nullopt;
    }
    const uint32_t width = shape[kWidthAxis];
    const uint32_t height = shape[kHeightAxis];
    if (width % TILE_WIDTH != 0 || height % TILE_HEIGHT != 0) {
        return std::nullopt;
    }

    const auto grid = input.device()->compute_with_storage_grid_size();

    Geometry g;
    static_cast<TilizeGeometry&>(g) = tilize_geometry(input, output_dtype);
    g.in_tile_bytes = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(input.dtype()));
    g.out_tile_bytes = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(output_dtype));
    g.pending_l1_output_bytes = get_pending_l1_output_reservation(
        input, shape, output_mem_config, output_dtype, Layout::TILE, /*require_constructible=*/true);

    g.row_cores = std::get<0>(tt::tt_metal::split_work_to_cores(grid, g.tile_rows, /*row_wise=*/true));
    return g;
}

// Fits the column chunk, buffering depth and write batch to `usable_l1` bytes per core, less the output's
// pending reservation. The route does not depend on the budget; nullopt means not even a one-tile chunk fits.
std::optional<TilizeCodegenPlan> plan_for_budget(const Geometry& g, uint64_t usable_l1) {
    const uint64_t budget = usable_l1 > g.pending_l1_output_bytes ? usable_l1 - g.pending_l1_output_bytes : 0;

    // Bytes of every CB for a chunk of `chunk` tiles held `depth` chunks deep; the conversion CB holds one chunk.
    auto cb_bytes = [&](uint64_t chunk, uint64_t depth, uint64_t out_depth_tiles) {
        return depth * chunk * g.in_tile_bytes + out_depth_tiles * g.out_tile_bytes +
               (g.convert ? chunk * g.in_tile_bytes : 0);
    };
    auto fits = [&](uint64_t chunk, uint64_t depth) { return cb_bytes(chunk, depth, depth * chunk) <= budget; };

    uint32_t chunk_wt = g.block_wt;
    uint32_t depth = kDoubleBuffered;
    if (!fits(g.block_wt, kDoubleBuffered)) {
        if (fits(g.block_wt, kSingleBuffered)) {
            depth = kSingleBuffered;
        } else {
            const uint64_t per_tile =
                static_cast<uint64_t>(g.in_tile_bytes) + g.out_tile_bytes + (g.convert ? g.in_tile_bytes : 0);
            const uint64_t max_chunk = budget / per_tile;
            if (max_chunk == 0) {
                return std::nullopt;
            }
            chunk_wt =
                largest_divisor_at_most(g.block_wt, static_cast<uint32_t>(std::min<uint64_t>(max_chunk, g.block_wt)));
            depth = fits(chunk_wt, kDoubleBuffered) ? kDoubleBuffered : kSingleBuffered;
        }
    }
    const uint32_t num_col_chunks = g.block_wt / chunk_wt;

    const bool multi_row_per_core = g.route == TilizeRoute::Row && g.tile_rows > g.row_cores;
    const bool minimal_work = num_col_chunks == 1 && chunk_wt == 1 && !multi_row_per_core;
    if (minimal_work) {
        depth = kSingleBuffered;
    }

    // A core with at most one tile row writes at most block_wt tiles. Batching waits on two batches at once, so
    // it is used only when that whole block fits the output CB without wrapping the FIFO.
    uint32_t write_batch = 1;
    uint32_t cb_out_depth = depth * chunk_wt;
    if (!multi_row_per_core && !minimal_work) {
        const uint32_t batched_out_depth =
            std::max(cb_out_depth, tt::round_up(kDoubleBuffered * kBatchedWriteTiles, chunk_wt));
        if (g.block_wt <= batched_out_depth && cb_bytes(chunk_wt, depth, batched_out_depth) <= budget) {
            write_batch = kBatchedWriteTiles;
            cb_out_depth = batched_out_depth;
        }
    }

    return TilizeCodegenPlan{
        .chunk_wt = chunk_wt,
        .num_col_chunks = num_col_chunks,
        .cb_in_depth = depth * chunk_wt,
        .cb_out_depth = cb_out_depth,
        .write_batch = write_batch};
}

}  // namespace

TilizeGeometry tilize_geometry(const Tensor& input, DataType output_dtype) {
    using namespace tt::constants;

    const uint32_t width = input.padded_shape()[kWidthAxis];
    const auto grid = input.device()->compute_with_storage_grid_size();
    const uint32_t grid_cores = static_cast<uint32_t>(grid.x * grid.y);

    TilizeGeometry g;
    g.wt = width / TILE_WIDTH;
    g.tile_rows = input.logical_shape().volume() / width / TILE_HEIGHT;
    g.convert = output_dtype != input.dtype();
    g.bfloat16_input = input.dtype() == DataType::BFLOAT16;
    if (g.tile_rows < grid_cores && g.wt >= kColumn2DMinWt) {
        const uint32_t blocks = column_blocks(g.wt, max_column_blocks(g, input.element_size(), grid_cores));
        if (blocks >= kMinColumnBlocks) {
            g.route = TilizeRoute::Column2D;
            g.num_col_blocks = blocks;
        }
    }
    g.block_wt = g.wt / g.num_col_blocks;
    return g;
}

bool uses_compact_reader(const TilizeGeometry& geometry, const TilizeCodegenPlan& plan) {
    return geometry.route == TilizeRoute::Row && !geometry.convert && geometry.bfloat16_input &&
           plan.num_col_chunks == 1 && geometry.wt <= kCompactReaderMaxWt;
}

// The chunk, depth and batch fields are frontier-derived, so the plan is part of the program cache key.
std::optional<TilizeCodegenPlan> plan_codegen(
    const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config, DataType output_dtype) {
    const auto geometry = analyze(input, output_mem_config, output_dtype);
    if (!geometry.has_value()) {
        return std::nullopt;
    }
    return plan_for_budget(*geometry, get_max_l1_space(input));
}

bool supported_by_codegen(
    const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config, DataType output_dtype) {
    const auto geometry = analyze(input, output_mem_config, output_dtype);
    return geometry.has_value() && plan_for_budget(*geometry, get_static_l1_space(input)).has_value();
}

std::optional<std::string_view> unsupported_execution_control(
    bool use_multicore,
    bool use_low_perf,
    const tt::tt_metal::Tile& tile,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (!use_multicore) {
        return "use_multicore=False";
    }
    if (use_low_perf) {
        return "use_low_perf=True";
    }
    if (sub_core_grids.has_value()) {
        return "sub_core_grids";
    }
    if (!is_default_tile(tile)) {
        return "tile";
    }
    return std::nullopt;
}

bool supported_execution_controls(
    bool use_multicore,
    bool use_low_perf,
    const tt::tt_metal::Tile& tile,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    return !unsupported_execution_control(use_multicore, use_low_perf, tile, sub_core_grids).has_value();
}

namespace {

using tt::tt_metal::BufferType;

constexpr std::array kBf16 = {DataType::BFLOAT16};
constexpr std::array kF32 = {DataType::FLOAT32};
constexpr std::array kInts = {DataType::INT32, DataType::UINT16, DataType::UINT32};
constexpr std::array kBf16Ints = {DataType::BFLOAT16, DataType::INT32, DataType::UINT16, DataType::UINT32};

constexpr std::optional<BufferType> kAny = std::nullopt;
constexpr std::optional<BufferType> kDram = BufferType::DRAM;
constexpr std::optional<BufferType> kL1 = BufferType::L1;

struct DemotedCase {
    // Zero-padded past the rank; a real dim is never zero because empty tensors are not supported.
    std::array<uint32_t, 4> dims;
    std::span<const DataType> dtypes;
    std::optional<BufferType> input;
    std::optional<BufferType> output;
};

// Unsettled: native was faster or the margin was not established, so these route to native for now.
// Same-dtype cases only; nullopt matches any buffer type.
const auto kUnsettled = std::to_array<DemotedCase>({
    {{1, 1, 64, 64}, kInts, kAny, kAny},   {{1, 10, 64, 96}, kBf16, kDram, kAny}, {{1, 12, 32, 96}, kBf16, kDram, kAny},
    {{1, 160, 192}, kBf16, kDram, kAny},   {{1, 3, 64, 64}, kF32, kAny, kAny},    {{1, 32, 64}, kInts, kAny, kAny},
    {{1, 4, 96, 32}, kBf16, kAny, kDram},  {{1, 96, 64}, kF32, kAny, kAny},       {{12, 32, 160}, kBf16, kDram, kAny},
    {{160, 192}, kBf16, kDram, kAny},      {{192, 160}, kBf16, kDram, kAny},      {{192, 192}, kBf16, kDram, kAny},
    {{192, 224}, kBf16, kDram, kAny},      {{2, 1, 96, 32}, kInts, kAny, kAny},   {{2, 192, 224}, kBf16, kDram, kAny},
    {{2, 32, 32}, kInts, kAny, kAny},      {{2, 4, 64, 96}, kF32, kDram, kAny},   {{2, 96, 32}, kF32, kAny, kAny},
    {{224, 128}, kBf16, kDram, kAny},      {{224, 32}, kBf16, kAny, kAny},        {{3, 2, 128, 32}, kF32, kAny, kAny},
    {{3, 2, 64, 32}, kF32, kAny, kAny},    {{3, 2, 96, 32}, kF32, kAny, kAny},    {{3, 8, 96, 32}, kBf16, kL1, kAny},
    {{3, 96, 32}, kF32, kAny, kAny},       {{32, 32}, kInts, kAny, kAny},         {{32, 64}, kInts, kAny, kAny},
    {{4, 12, 96, 96}, kBf16, kDram, kAny}, {{4, 4, 32, 64}, kF32, kL1, kDram},    {{4, 4, 64, 64}, kF32, kAny, kAny},
    {{4, 6, 96, 32}, kBf16, kL1, kAny},    {{4, 9, 64, 32}, kBf16, kL1, kAny},    {{5, 128, 32}, kBf16, kL1, kAny},
    {{5, 3, 64, 32}, kBf16, kL1, kAny},    {{5, 3, 64, 32}, kBf16, kDram, kL1},   {{5, 8, 64, 64}, kBf16, kL1, kAny},
    {{6, 10, 32, 64}, kBf16, kL1, kDram},  {{6, 4, 96, 64}, kBf16, kL1, kAny},    {{64, 32}, kBf16Ints, kAny, kAny},
    {{64, 64}, kInts, kAny, kAny},         {{7, 96, 160}, kBf16, kDram, kAny},    {{8, 32, 160}, kBf16, kDram, kAny},
    {{9, 128, 32}, kBf16, kL1, kL1},
});

bool shape_matches(const auto& shape, const std::array<uint32_t, 4>& dims) {
    for (size_t i = 0; i < dims.size(); ++i) {
        const uint32_t dim = i < shape.rank() ? shape[i] : 0;
        if (dim != dims[i]) {
            return false;
        }
    }
    return true;
}

bool buffer_matches(const std::optional<BufferType>& wanted, BufferType actual) {
    return !wanted.has_value() || *wanted == actual;
}

}  // namespace

// Perf-only and routing-only. Entries must come from device kernel durations, never end-to-end timing.
bool is_demoted(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config, DataType output_dtype) {
    if (output_dtype != input.dtype()) {
        return false;
    }
    return std::any_of(kUnsettled.begin(), kUnsettled.end(), [&](const DemotedCase& c) {
        return std::find(c.dtypes.begin(), c.dtypes.end(), output_dtype) != c.dtypes.end() &&
               buffer_matches(c.input, input.memory_config().buffer_type()) &&
               buffer_matches(c.output, output_mem_config.buffer_type()) &&
               shape_matches(input.logical_shape(), c.dims);
    });
}

}  // namespace ttnn::operations::data_movement::tilize_codegen
