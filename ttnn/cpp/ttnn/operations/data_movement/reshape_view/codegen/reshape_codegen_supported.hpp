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
// itself. Live L1 picks the plan at program creation (live_tile_choice, live_rm_choice).
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

// Perf routing only, for a call supported_by_codegen: native measures faster than the program the gate admits.
// TILE: the program taken against the static window does not beat native's tiled reshape. Tile rows wins when the
// full plan cuts reads 8x, saves 16+ reads per output tile, or copies whole face halves over more than one native
// round; LCM when its tiles cost under 6x native's per round; the segment map on widths that are not multiples of 32.
// Tile rows and LCM win only while the segment map they fall back to is within its size caps.
// ROW_MAJOR: the smallest stick or slab plan needs more L1 than native's smallest footprint, or rm_native_faster's
// stick-size rule, taken only while the stick program and native's scratch both fit the static L1 window. Depends on
// shapes and static device geometry only, never live L1. Never true for uint16: native rejects it, so demoting would
// fail the call.
bool is_demoted(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt);

// The static plan for a supported call; the program factory shrinks it to live L1. Raises with unsupported_reason's
// text otherwise. Built by prim::reshape_codegen.
LaunchPlan launch_plan(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

// `sub_core_grids`, when given, must be a non-empty subset of the device's compute grid.
bool supported_execution_controls(const Tensor& input, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::data_movement::reshape_codegen
