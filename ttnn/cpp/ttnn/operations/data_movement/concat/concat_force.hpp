// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

// Verification-only entry points that pin which concat implementation runs.
//
// `ttnn.concat` chooses on its own and takes no argument to override that: which implementation
// serves a call is an internal decision. These exist because the choice still has to be
// *checkable* -- the guarantees are bit-exactness against the existing implementation over the
// supported scope and a device-time win, and neither is observable through an entry point that
// silently picks one. They are bound only under `ttnn._ttnn.operations.data_movement` and are
// deliberately not registered into the `ttnn.*` namespace, so they are reachable from tests and
// benchmarking without being part of the public API. This header is likewise kept out of the
// installed `api` file set, and they sit in `detail` rather than alongside the real op entries so
// that a caller reaching for one has to say so.
//
// Prefer `ttnn::concat` everywhere else, including for native-only cases: it already declines to
// route sharded, non-row-major, out-of-dtype and execution-control-constrained cases to codegen.

// The existing implementation, unconditionally.
ttnn::Tensor concat_force_native(
    const std::vector<ttnn::Tensor>& input_tensors,
    int dim,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    unsigned int groups = 1,
    const std::optional<ttnn::CoreRangeSet>& sub_core_grids = std::nullopt);

// The generated implementation, unconditionally. Throws for a case outside its support scope
// instead of falling back, so that a comparison against the native path cannot silently end up
// measuring native twice and reporting it as agreement. Takes neither `groups` nor
// `sub_core_grids`: no generated builder honours either.
ttnn::Tensor concat_force_codegen(
    const std::vector<ttnn::Tensor>& input_tensors,
    int dim,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::operations::data_movement::detail
