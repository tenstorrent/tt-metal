// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::concat_codegen {

// Correctness/feasibility gate for the four row-major builders. Consulted by the free
// function's routing, by concat_force_codegen, and by prim::concat_codegen's validate, so
// all three agree on the supported scope. `dim` is already normalized (0..rank-1).
bool supported_by_codegen(
    const std::vector<Tensor>& input_tensors, uint32_t dim, const tt::tt_metal::MemoryConfig& output_mem_config);

// Perf-demotion gate: correct but not worth the codegen path. Routing-only --
// consulted by the auto branch only, never by validate.
bool is_demoted(const std::vector<Tensor>& input_tensors, uint32_t dim);

// Native execution controls (`groups`, `sub_core_grids`) that no ConcatCodegen
// builder honours. Kept out of supported_by_codegen()/is_demoted(): the codegen
// prim carries neither attribute, so this is a free-function-only routing
// concern, shared between the auto branch and the forced-codegen TT_FATAL.
bool supported_execution_controls(unsigned int groups, const std::optional<ttnn::CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::data_movement::concat_codegen
