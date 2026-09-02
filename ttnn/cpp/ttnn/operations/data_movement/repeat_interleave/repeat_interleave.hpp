// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"

namespace ttnn {

// # This operation does not support the following cases:
// #   - `repeats` as a per-element Tensor (e.g. Shape([2, 2]) -> repeats = Tensor[1,2], dim = 1).
// #     Only a single scalar `repeats` applied uniformly across the dim is supported.
// # (Small-shape cases such as Shape([2, 2]) -> repeats = 2, dim = 0 are supported.)

ttnn::Tensor repeat_interleave(
    const ttnn::Tensor& input_a,
    uint32_t repeats,
    int32_t dim,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

}  // namespace ttnn
