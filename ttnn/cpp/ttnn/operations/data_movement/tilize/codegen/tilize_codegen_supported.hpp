// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tile.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"

namespace ttnn::operations::data_movement::tilize_codegen {

enum class TilizeRoute : uint8_t {
    Row,       // each core owns whole tile rows
    Column2D,  // each core owns one tile row and one contiguous column block of it
};

// Work decomposition of one in-scope input, a pure function of the tensor, the output dtype and the
// compute grid. The program factory recomputes it rather than receiving it in the operation attributes.
struct TilizeGeometry {
    TilizeRoute route = TilizeRoute::Row;
    uint32_t tile_rows = 0;       // batch * (H / tile height)
    uint32_t wt = 0;              // tiles per tile row
    uint32_t num_col_blocks = 1;  // cores that split one tile row (1 for the row route)
    uint32_t block_wt = 0;        // tiles of a tile row one core covers (wt / num_col_blocks)
    bool convert = false;         // output dtype differs from the input dtype
    bool bfloat16_input = false;
};

// The part of the plan that depends on how much L1 is free. It is carried in the operation attributes, so
// it is part of the program cache key; everything derivable from the tensor lives in TilizeGeometry.
struct TilizeCodegenPlan {
    uint32_t chunk_wt = 0;        // tiles per column chunk (a divisor of block_wt)
    uint32_t num_col_chunks = 0;  // block_wt / chunk_wt
    uint32_t cb_in_depth = 0;     // input CB depth, in tiles
    uint32_t cb_out_depth = 0;    // output CB depth, in tiles
    uint32_t write_batch = 0;     // tiles per writer batch

    bool operator==(const TilizeCodegenPlan&) const = default;
};

// Geometry of an input that already passed the scope checks of plan_codegen.
TilizeGeometry tilize_geometry(const Tensor& input, tt::tt_metal::DataType output_dtype);

// True when the single-chunk bfloat16 row reader with the default read ABI applies.
bool uses_compact_reader(const TilizeGeometry& geometry, const TilizeCodegenPlan& plan);

// The plan for this case sized against the L1 free right now (below the lowest live buffer, less the
// output's pending reservation), or nullopt when out of scope or when not even a one-tile chunk fits.
// The output dtype is the input dtype, or INT32 for a bfloat16 input (converted on device).
std::optional<TilizeCodegenPlan> plan_codegen(
    const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config, tt::tt_metal::DataType output_dtype);

// Static scope and feasibility against the allocator's full L1 window, independent of live occupancy.
// Never gated on performance.
bool supported_by_codegen(
    const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config, tt::tt_metal::DataType output_dtype);

// The codegen program always places work over the full compute-with-storage grid and emits the default
// tile. The first control it cannot honour, or nullopt when it honours all of them.
std::optional<std::string_view> unsupported_execution_control(
    bool use_multicore,
    bool use_low_perf,
    const tt::tt_metal::Tile& tile,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

// True iff no execution control is unsupported. False means the call must go to native.
bool supported_execution_controls(
    bool use_multicore,
    bool use_low_perf,
    const tt::tt_metal::Tile& tile,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

// Perf-only, routing-only: true for in-scope cases where codegen is correct but does not beat native.
bool is_demoted(
    const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config, tt::tt_metal::DataType output_dtype);

}  // namespace ttnn::operations::data_movement::tilize_codegen
