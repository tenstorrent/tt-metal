// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/types.hpp"

namespace ttnn {

// Indices of the maximum values. Which engine serves the call -- the scalar
// reader kernels, the Blackhole RVV scan, or the Blackhole SFPU reduction --
// is decided internally from the input spec; see select_argmax_engine in
// argmax.cpp for the heuristic and the measurements behind it.
//
// exact_special_values constrains that choice to the engines that are
// bit-identical to the scalar readers on every input, special values included.
// Leave it false (the default) unless the caller actually depends on the
// scalar readers' NaN / denormal / signed-zero behaviour; setting it can only
// cost throughput, never correctness.
//
// optional_maxval_tensor (a preallocated BFLOAT16 ROW_MAJOR tensor shaped like
// the index output) receives the winning max VALUES. Only the accelerated
// engines can produce it, so a call that supplies it and does not qualify for
// one raises rather than returning a stale buffer.
Tensor argmax(
    const Tensor& input_tensor,
    const std::optional<int>& dim = std::nullopt,
    bool keepdim = false,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<Tensor> optional_output_tensor = std::nullopt,
    bool exact_special_values = false,
    std::optional<Tensor> optional_maxval_tensor = std::nullopt);

}  // namespace ttnn
