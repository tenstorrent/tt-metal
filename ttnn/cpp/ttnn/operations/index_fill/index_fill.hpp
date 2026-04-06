// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"

namespace ttnn {

Tensor index_fill(
    const Tensor& input,
    uint32_t dim,
    const Tensor& index,
    std::variant<float, int> value,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
