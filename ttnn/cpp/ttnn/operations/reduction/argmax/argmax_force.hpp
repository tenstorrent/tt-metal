// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/types.hpp"

namespace ttnn::operations::reduction::detail {

// Verification-only entry points that pin which argmax path runs.
//
// `ttnn.argmax` chooses on its own and takes no argument to override that:
// which path serves a call is an internal decision (see select_argmax_path in
// argmax.cpp). These exist because the choice still has to be *checkable*.
// Each path carries its own claim -- the RVV scan is bit-identical to the
// scalar readers, the SFPU reduction diverges from them in a specific,
// measured way on special values, and both are supposed to beat them by an
// order of magnitude -- and none of that is observable through an entry point
// that silently picks one. They also give the suites a stable golden leg: on
// Blackhole a plain `ttnn.argmax` over a TILE bfloat16 last dim does not run
// the scalar readers, so "compare against the scalar readers" needs a way to
// say so.
//
// They are bound only under `ttnn._ttnn.operations.reduction` and are
// deliberately not registered into the `ttnn.*` namespace, so they are
// reachable from tests and benchmarking without being part of the public API.
// This header is likewise kept out of the installed `api` file set, and they
// sit in `detail` rather than alongside the real op entry so that a caller
// reaching for one has to say so.
//
// None of the three falls back: a forced leg that quietly served a different
// path would make any comparison against it vacuous. Asking for a path that
// cannot serve the case raises out of the device op's validation.
//
// Prefer `ttnn::argmax` everywhere else.

// The scalar reader kernels (single- or multi-core), unconditionally, on any
// architecture. This is the golden leg the other two are compared against.
ttnn::Tensor argmax_force_scalar_reader(
    const ttnn::Tensor& input_tensor,
    const std::optional<int>& dim = std::nullopt,
    bool keepdim = false,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt,
    std::optional<ttnn::Tensor> optional_maxval_tensor = std::nullopt);

// The Blackhole RVV (Zve32f pack-RISC) TILE last-dim scan, unconditionally.
ttnn::Tensor argmax_force_rvv(
    const ttnn::Tensor& input_tensor,
    const std::optional<int>& dim = std::nullopt,
    bool keepdim = false,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt,
    std::optional<ttnn::Tensor> optional_maxval_tensor = std::nullopt);

// The Blackhole SFPU TILE last-dim reduction, unconditionally.
ttnn::Tensor argmax_force_sfpu(
    const ttnn::Tensor& input_tensor,
    const std::optional<int>& dim = std::nullopt,
    bool keepdim = false,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt,
    std::optional<ttnn::Tensor> optional_maxval_tensor = std::nullopt);

}  // namespace ttnn::operations::reduction::detail
