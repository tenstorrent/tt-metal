// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::gather {

// Correctness gate: true iff the codegen prim can produce a result matching native's contract
// for this input/dim/index combination. Consulted by ttnn::gather()'s routing, by
// gather_force_codegen(), and by the codegen prim's own validation step. Evaluated on the ORIGINAL
// (pre pre_gather_transform_tensor) tensors and the caller's raw dim, before any
// transpose/4D-fold/tilize — the attributes the supported scope is expressed in (dim, layout, dtype
// as the caller supplied them). The codegen kernels themselves are dim-agnostic (they only ever see
// a post-transform, already-last-dim tensor), so `dim` is not consulted for correctness here.
bool supported_by_codegen(const Tensor& input_tensor, int8_t dim, const Tensor& input_index_tensor);

// The half of the call contract supported_by_codegen() cannot see: the caller-controlled output
// placement and, when the caller preallocates a destination, the spec that destination carries.
// False means the codegen factories cannot honour what the caller asked for — `auto` must route to
// native and forced `codegen` must fail rather than silently write through the mismatch. Not a perf
// question, so it is kept out of is_demoted() too.
bool supported_execution_controls(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor);

// Perf gate consulted only by ttnn::gather()'s routing, and only when supported_by_codegen() is
// true: true means fall back to native despite codegen support. Evaluated at the same pre-transform
// point as supported_by_codegen() (same tensors/dim), matching the configurations any demotion
// would be measured over. Currently demotes nothing; the known ~2% single-logical-row regression is
// deliberately accepted (see is_demoted()), but this hook remains the single place for future demotions.
// gather_force_codegen() is never affected by it.
bool is_demoted(const Tensor& input_tensor, int8_t dim, const Tensor& input_index_tensor);

}  // namespace ttnn::operations::data_movement::gather
