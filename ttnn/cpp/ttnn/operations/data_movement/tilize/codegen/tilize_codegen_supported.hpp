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
    Row,
    Column2D,
};

struct TilizeGeometry {
    TilizeRoute route = TilizeRoute::Row;
    uint32_t tile_rows = 0;
    uint32_t wt = 0;
    uint32_t num_col_blocks = 1;
    uint32_t block_wt = 0;
    bool convert = false;
};

// Depends on free L1, so it is part of the program cache key.
struct TilizeCodegenPlan {
    uint32_t chunk_wt = 0;
    uint32_t num_col_chunks = 0;
    uint32_t cb_in_depth = 0;
    uint32_t cb_out_depth = 0;
    uint32_t write_batch = 0;

    bool operator==(const TilizeCodegenPlan&) const = default;
};

TilizeGeometry tilize_geometry(const Tensor& input, tt::tt_metal::DataType output_dtype);

bool uses_compact_reader(const TilizeGeometry& geometry, const TilizeCodegenPlan& plan);

// Sized against the L1 free right now.
std::optional<TilizeCodegenPlan> plan_codegen(
    const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config, tt::tt_metal::DataType output_dtype);

// Sized against the allocator's full L1 window, independent of live occupancy.
bool supported_by_codegen(
    const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config, tt::tt_metal::DataType output_dtype);

std::optional<std::string_view> unsupported_execution_control(
    bool use_multicore,
    bool use_low_perf,
    const tt::tt_metal::Tile& tile,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

bool supported_execution_controls(
    bool use_multicore,
    bool use_low_perf,
    const tt::tt_metal::Tile& tile,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

// Routing-only: codegen is correct here but measured slower than native.
bool is_demoted(const Tensor& input, tt::tt_metal::DataType output_dtype);

}  // namespace ttnn::operations::data_movement::tilize_codegen
