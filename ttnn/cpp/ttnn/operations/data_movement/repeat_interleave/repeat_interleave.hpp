// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

// # This operation does not support the following cases:
// #   - Shape([2[32], 2[32]]) -> repeats = 2, dim = 0
// #   - Shape([2[32], 2[32]]) -> repeats = Tensor[1,2], dim = 1

ttnn::Tensor repeat_interleave(
    const ttnn::Tensor& input_a,
    uint32_t repeats,
    int32_t dim,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

}  // namespace ttnn
