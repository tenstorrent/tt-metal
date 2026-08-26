// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "pad_codegen_device_operation_types.hpp"

namespace ttnn::operations::data_movement::pad_codegen {

// Correctness gate: can the codegen path produce a bit-exact result for these inputs?
// Transcribed from codegen_pad.py's invalidate_vector + ops/pad/pad.py's guards.
bool supported_by_codegen(
    const ttnn::prim::PadCodegenParams& operation_attributes, const ttnn::prim::PadCodegenInputs& tensor_args);

// Every codegen builder places work over the full compute-with-storage grid and has no single-core
// variant, so it can honour neither of the native op's core-placement controls. False means the
// case must go to native, or be rejected outright by pad_force_codegen. Separate from
// supported_by_codegen() because these are free-function attributes: the codegen prim carries no
// such fields, so its validate has nothing to check.
bool supported_execution_controls(bool use_multicore, const std::optional<CoreRangeSet>& sub_core_grids);

// Perf gate, auto-only: an in-scope case that codegen can do but shouldn't be routed to.
bool is_demoted(
    const ttnn::prim::PadCodegenParams& operation_attributes, const ttnn::prim::PadCodegenInputs& tensor_args);

}  // namespace ttnn::operations::data_movement::pad_codegen
