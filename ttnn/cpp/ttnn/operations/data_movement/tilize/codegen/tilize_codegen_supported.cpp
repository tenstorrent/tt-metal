// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_supported.hpp"

#include <algorithm>
#include <optional>
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

constexpr uint32_t kDoubleBuffered = 2;
constexpr uint32_t kSingleBuffered = 1;
constexpr uint32_t kMinColumnBlocks = 2;

// Route and split tunings below are empirical, measured on Blackhole (P150); performance choices, not hardware or
// correctness limits.
// Widest tile row, in tiles, that the compact reader takes; wider rows use the chunked reader.
constexpr uint32_t kCompactReaderMaxWt = 8;
// Tiles per writer batch on a core that holds at most one tile row.
constexpr uint32_t kBatchedWriteTiles = 4;
// A column split needs at least this many tiles per row to amortize its extra reader geometry.
constexpr uint32_t kColumn2DMinWt = 3;
// Few-row DRAM split: narrowest column block, in bytes per stick, one core reads.
constexpr uint32_t kMinCoreColBytes = 1024;
// Few-row DRAM split: smallest per-stick read fragment a tile row is cut into.
constexpr uint32_t kMinFragmentBytes = 256;
// Few-row DRAM split: cores the cut aims to occupy; more would only read narrower fragments.
constexpr uint32_t kTargetCores = 24;
// Few-row DRAM split: a cut onto fewer cores than this keeps the full grid split instead.
constexpr uint32_t kMinSplitCores = 5;
// Most tile rows of a one-tile-wide, non-converting input that still routes to native; see is_demoted().
constexpr uint32_t kDemoteMaxTileRows = 20;

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

// Narrow DRAM fragments read inefficiently, so a few-row DRAM tensor gets wide blocks on few cores. A cut onto
// fewer than kMinSplitCores cores keeps the full grid; L1 input or a conversion is bounded only by the grid.
uint32_t max_column_blocks(const TilizeGeometry& g, uint32_t elem_size, uint32_t grid_cores, bool fragment_limited) {
    const uint32_t grid_blocks = grid_cores / g.tile_rows;
    if (!fragment_limited) {
        return grid_blocks;
    }
    const uint32_t fragment_bytes = tt::constants::TILE_WIDTH * elem_size;
    const uint32_t align_tiles = std::max(1u, tt::tt_metal::hal::get_dram_alignment() / fragment_bytes);
    const uint32_t col_units = tt::div_up(g.wt, align_tiles);
    const uint32_t min_units = tt::div_up(kMinCoreColBytes / fragment_bytes, align_tiles);
    const uint32_t wide_blocks = std::max(1u, col_units / min_units);
    const uint32_t row_bytes = g.wt * fragment_bytes;
    const uint32_t fragment_blocks =
        std::min(col_units, std::max(1u, row_bytes / std::max(kMinFragmentBytes, align_tiles * fragment_bytes)));
    const uint32_t few_row_blocks = std::min(fragment_blocks, tt::div_up(kTargetCores, g.tile_rows));
    const uint32_t capped = std::min(grid_blocks, std::max(wide_blocks, few_row_blocks));
    return g.tile_rows * capped < kMinSplitCores ? grid_blocks : capped;
}

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

    // Tile size, page count and row pitch below are 32x32 quantities; a transposed tile would also permute datums.
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

std::optional<TilizeCodegenPlan> plan_for_budget(const Geometry& g, uint64_t usable_l1) {
    const uint64_t budget = usable_l1 > g.pending_l1_output_bytes ? usable_l1 - g.pending_l1_output_bytes : 0;

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

    // Batching waits on two batches at once, so use it only when the whole block fits without wrapping the FIFO.
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
    if (g.tile_rows < grid_cores && g.wt >= kColumn2DMinWt) {
        const bool fragment_limited =
            !g.convert && input.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM;
        const uint32_t blocks =
            column_blocks(g.wt, max_column_blocks(g, input.element_size(), grid_cores, fragment_limited));
        if (blocks >= kMinColumnBlocks) {
            g.route = TilizeRoute::Column2D;
            g.num_col_blocks = blocks;
        }
    }
    g.block_wt = g.wt / g.num_col_blocks;
    return g;
}

bool uses_compact_reader(const TilizeGeometry& geometry, const TilizeCodegenPlan& plan) {
    return geometry.route == TilizeRoute::Row && plan.num_col_chunks == 1 && geometry.wt <= kCompactReaderMaxWt;
}

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

// One tile per row has no width parallelism, so kernels this short don't amortize codegen's fixed per-launch cost.
bool is_demoted(const Tensor& input, DataType output_dtype) {
    using namespace tt::constants;

    if (output_dtype != input.dtype()) {
        return false;
    }
    const auto& shape = input.padded_shape();
    if (shape.rank() < kMinRank) {
        return false;
    }
    const uint32_t width = shape[kWidthAxis];
    if (width == 0) {
        return false;
    }
    const uint32_t wt = width / TILE_WIDTH;
    const uint32_t tile_rows = input.logical_shape().volume() / width / TILE_HEIGHT;
    return wt == 1 && tile_rows <= kDemoteMaxTileRows;
}

}  // namespace ttnn::operations::data_movement::tilize_codegen
