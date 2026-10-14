// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_plan.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

// The program one call runs, planned once against the L1 free at launch: the live free L1 less the output the launch
// allocates. The device operation hashes, selects and builds from it and reads no allocator state itself.
struct LaunchPlan {
    uint64_t l1_budget = 0;
    std::optional<tt::tt_metal::TensorSpec> output_spec;
};

struct CodegenDecision {
    // supported_by_codegen's verdict. Not evaluated, so false, for a routing call native_faster settles.
    bool supported = false;
    // Routing only: native's kernel measures faster than the program the gate would admit. Forced codegen still runs
    // these.
    bool native_faster = false;
    // The live plan, or nullopt when no program fits the L1 free now. Not planned for unsupported or native-faster
    // routing calls.
    std::optional<LaunchPlan> plan;
};

// Correctness and device-resource gate shared by ttnn::reshape routing, the forced entry and the codegen prim's
// validation. Plans against static_l1_budget only: the gate runs more than once per call with the op's own output
// allocated in between, so a live read could admit a call in routing and refuse it in validation.
bool supported_by_codegen(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt);

// Output keeps the input's memory config and the whole compute grid is used.
bool supported_by_codegen(const Tensor& input, const ttnn::Shape& logical_output_shape);

// For `routing` whether native is faster, then the gate, then the live plan, sharing one geometry per program and one
// live L1 read. Routing takes codegen when supported && !native_faster && plan; the forced entry needs supported and
// fails without a plan. Without `plan_launch` it stops after the gate and reads no live L1.
CodegenDecision decide_codegen(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    bool routing,
    bool plan_launch = true);

// Execution controls the codegen path can honour: `sub_core_grids`, when given, must be a non-empty subset of the
// device's compute grid.
bool supported_execution_controls(const Tensor& input, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::data_movement::reshape_codegen
