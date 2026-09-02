// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/types.hpp"

namespace ttnn {

// Indices of the maximum values along `dim`.
//
// exact_special_values restricts the internally chosen kernel path to the ones
// bit-identical to the scalar reader kernels on every input, NaN / denormal /
// signed zero included; it can only cost throughput, never correctness.
// optional_maxval_tensor -- a preallocated BFLOAT16 ROW_MAJOR tensor shaped
// like the index output -- receives the winning max VALUES; not every path can
// produce it, so a call that supplies it and does not qualify raises rather
// than returning a stale buffer.
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
