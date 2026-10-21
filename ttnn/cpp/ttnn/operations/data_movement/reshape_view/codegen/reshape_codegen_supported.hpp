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

// Why the codegen program cannot run this call before any plan is built (placement, layout, dtype, shapes, grid),
// or nullptr. The program cache revalidates with it on a miss.
const char* scope_violation(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

// The TILE program one call takes, planned against static_l1_budget only, never live free L1: the op allocates its
// output between routing and program creation, so a live read could disagree with itself. Live L1 sizes the plan at
// program creation (live_tile_choice).
struct RouteDecision {
    // Set when the call is in scope and the smallest plan of some program fits the static budget.
    std::optional<LaunchPlan> plan;
    // Why `plan` is unset.
    const char* violation = nullptr;
    // Perf routing only: native's tiled reshape measures faster than the planned program. Never true for uint16,
    // which native rejects.
    bool demoted = false;
};

// Plans the call once for both routing and the forced entry.
RouteDecision plan_codegen(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt);

// `sub_core_grids`, when given, must be a non-empty subset of the device's compute grid.
bool supported_execution_controls(const Tensor& input, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::data_movement::reshape_codegen
