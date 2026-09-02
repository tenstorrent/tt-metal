// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/types.hpp"

namespace ttnn::operations::reduction::detail {

// Verification-only entry points that pin which argmax path runs.
//
// `ttnn.argmax` chooses internally and takes no override, so per-path claims --
// the RVV scan's bit-identity with the scalar readers, the SFPU reduction's
// measured special-value divergence, either path's speedup over them -- are not
// checkable through it, and the suites have no scalar-reader golden on
// Blackhole, where a plain TILE bfloat16 last-dim call never runs them.
//
// None of the three falls back: a forced leg that quietly served a different
// path would make any comparison against it vacuous. Asking for a path that
// cannot serve the case raises out of the device op's validation.
//
// Bound only under `ttnn._ttnn.operations.reduction`, deliberately not
// registered into `ttnn.*`, kept out of the installed `api` file set, and in
// `detail`, so reaching for one is explicit. Prefer `ttnn::argmax` elsewhere.

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
