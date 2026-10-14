// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_plan.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

// Why the codegen program cannot reproduce native's output for the caller's request (explicit padded shape, volume
// change, explicit pad_value on a padded tile output), or nullptr. Shared by routing and the forced entry.
const char* call_contract_violation(
    const Tensor& input,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const std::optional<ttnn::PadValue>& pad_value);

// Why the codegen program cannot run this call, or nullptr. Budgets the smallest plan against static_l1_budget only,
// never live free L1: the op allocates its output between routing and validation, so a live read could disagree with
// itself. Live L1 picks the plan at program creation (live_tile_rows_choice).
const char* unsupported_reason(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt);

// unsupported_reason(...) == nullptr. Output keeps the input's memory config when none is given.
bool supported_by_codegen(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt);
bool supported_by_codegen(const Tensor& input, const ttnn::Shape& logical_output_shape);

// Perf routing only, TILE: native's tiled reshape measures faster than the tile-rows program, so routing sends the
// call to native. True unless the full plan cuts reads 8x, saves 16+ reads per output tile, or copies whole face
// halves over more than one native round; thresholds fitted on Blackhole. Never true for uint16, which native rejects,
// nor for ROW_MAJOR, whose demotion routed_launch_plan decides from stick sizes and the static fit.
bool is_demoted(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt);

// The plan when routing sends this call to codegen (supported_by_codegen, not demoted, and some plan fits the L1 free
// once the output is allocated), else nullopt.
std::optional<LaunchPlan> routed_launch_plan(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

// The plan for the forced entry, which ignores demotion; raises with unsupported_reason's text otherwise, or when no
// row-major program fits the L1 free at launch.
LaunchPlan forced_launch_plan(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

// `sub_core_grids`, when given, must be a non-empty subset of the device's compute grid.
bool supported_execution_controls(const Tensor& input, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::data_movement::reshape_codegen
