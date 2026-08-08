// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::gather {

// Selects which prim ttnn::gather() dispatches to; shared by the free function's routing
// and the codegen prim's validation step (see call_parity.routing in the gather manifest).
enum class ImplementationSelector { kAuto, kNative, kCodegen };

ImplementationSelector parse_implementation(std::string_view implementation);

// Correctness gate: true iff the codegen prim can produce a result matching native's contract
// for this input/dim/index combination. Evaluated on the ORIGINAL (pre pre_gather_transform_tensor)
// tensors and the caller's raw dim, before any transpose/4D-fold/tilize — the same point the
// gather manifest's cases and sweep vectors describe (dim, layout, dtype as the caller supplied
// them). The codegen kernels themselves are dim-agnostic (they only ever see a post-transform,
// already-last-dim tensor), so `dim` is not consulted for correctness here.
bool supported_by_codegen(const Tensor& input_tensor, int8_t dim, const Tensor& input_index_tensor);

// The half of the call contract supported_by_codegen() cannot see: the caller-controlled output
// placement, which is not an attribute of the codegen prim and so has no place in its validation
// step. False means the codegen factories cannot honour what the caller asked for — `auto` must
// route to native and forced `codegen` must fail rather than silently override the placement.
// Not a perf question, so it is kept out of is_demoted() too.
bool supported_execution_controls(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor);

// Perf gate consulted only when ImplementationSelector::kAuto and supported_by_codegen() is true:
// true means fall back to native despite codegen support. Evaluated at the same pre-transform
// point as supported_by_codegen() (same tensors/dim), matching the case_id vectors any demotion
// would be measured against. Currently demotes nothing -- no measured in-scope configuration loses
// to the native prim -- but it stays in the routing expression so a future demotion has one place
// to live and forced implementation="codegen" is never affected by it.
bool is_demoted(const Tensor& input_tensor, int8_t dim, const Tensor& input_index_tensor);

}  // namespace ttnn::operations::data_movement::gather
