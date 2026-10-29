// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
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

// Why the codegen program cannot run this call, or nullptr. With `plan`, also why that plan's smallest size does not
// fit static_l1_budget; without one, why plan_codegen found none. The budget is static, never live free L1: the op
// allocates its output between routing and program creation, so a live read could disagree with itself. Live L1 sizes
// the plan at program creation (live_tile_choice).
const char* unsupported_reason(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    const LaunchPlan* plan);

// The gate for an already-built plan: unsupported_reason(...) == nullptr. plan_codegen delegates to it and the codegen
// prim's validation calls it.
bool supported_by_codegen(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    const LaunchPlan& plan);

// Routing and forced entry: tile rows when the gate accepts it, else the segment map, else nullopt.
std::optional<LaunchPlan> plan_codegen(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt);

// Whether the segment map takes these shapes on `num_cores` cores within its size caps, at any budget.
bool segment_map_within_caps(const Tensor& input, const ttnn::Shape& logical_output_shape, uint32_t num_cores);

// Perf-demotion gate, routing-only: codegen is correct but native's tiled reshape measures faster. `plan` is
// plan_codegen's and `num_cores` the count of reshape_core_grid it planned on.
bool is_demoted(
    const Tensor& input, const ttnn::Shape& logical_output_shape, uint32_t num_cores, const LaunchPlan& plan);

// `sub_core_grids`, when given, must be a non-empty subset of the device's compute grid.
bool supported_execution_controls(const Tensor& input, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::data_movement::reshape_codegen
