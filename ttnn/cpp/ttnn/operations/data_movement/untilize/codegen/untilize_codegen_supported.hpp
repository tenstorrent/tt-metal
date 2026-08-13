// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::untilize_codegen {

enum class ImplementationSelector { Auto, Native, Codegen };

// Parses the `implementation` kwarg. TT_FATALs on an unrecognized value.
ImplementationSelector parse_implementation(const std::string& implementation);

// Correctness / device-resource-feasibility gate. `input` is the tensor as
// passed to ttnn::untilize, before the DRAM+padding redirect. Transcribed from
// ops/untilize/spec.py's build_untilize_tile / build_untilize_with_unpadding
// scope (layout/dtype/alignment) plus this port's own CB-fit bound at every
// dispatch-tree leaf (single, cliff, column-parallel, 2D-column, with-unpadding).
bool supported_by_codegen(const Tensor& input);

// Perf-demotion gate: correct but not worth the codegen path. Routing-only —
// consulted by the auto branch only, never by validate.
bool is_demoted(const Tensor& input);

// Execution-control guard: true when a caller-specified placement/execution
// override (use_multicore=false, sub_core_grids) is set. None of the codegen
// builders honour these -- they always dispatch over the full
// compute_with_storage_grid_size() -- so both "auto" and "codegen" must not
// silently ignore them (see porting guide's use_multicore/sub_core_grids trap).
bool has_execution_control_override(bool use_multicore, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::data_movement::untilize_codegen
