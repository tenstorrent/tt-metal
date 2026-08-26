// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

struct PadSpecDim;

}  // namespace ttnn::operations::data_movement

namespace ttnn::operations::data_movement::detail {

// Verification-only entry points that pin which pad prim runs.
//
// `ttnn.pad` chooses between the native and codegen prims on its own and takes no argument to
// override that: which prim serves a call is an internal decision. These exist because the choice
// still has to be *checkable* -- the codegen port's guarantees are that it is bit-exact against
// native over its supported scope and faster on device, and neither is observable through an entry
// point that silently picks one. They are bound only under `ttnn._ttnn.operations.data_movement`
// and are deliberately not registered into the `ttnn.*` namespace, so they are reachable from
// tests and benchmarking without being part of the public API. This header is likewise kept out
// of the installed `api` file set, and they sit in `detail` rather than alongside the real op
// entries so that a caller reaching for one has to say so.
//
// Prefer `ttnn::pad` everywhere else, including for native-only cases: it already declines to
// route sharded, unsupported, execution-control-constrained, or perf-demoted cases to codegen.

// The native prim, unconditionally. Carries the two core-placement controls because a routing test
// that compares against this leg has to be able to ask for the same placement `ttnn::pad` was given:
// native builds a different program per placement, so a golden taken at the default placement would
// make any placement-driven fallback look like a mis-route.
ttnn::Tensor pad_force_native(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<PadSpecDim>& padding,
    float value,
    bool use_multicore = true,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

// The codegen prim, unconditionally. Throws for a case outside the codegen support scope instead of
// falling back to native, so that a comparison against the native prim cannot silently end up
// measuring native twice and reporting it as agreement.
ttnn::Tensor pad_force_codegen(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<PadSpecDim>& padding,
    float value,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::operations::data_movement::detail
