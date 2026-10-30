// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/types.hpp"

namespace ttnn {

// Descriptor-based port of ttnn::index_fill (#42392); same contract, removed once it replaces the legacy factory.
Tensor index_fill_new(
    const Tensor& input,
    uint32_t dim,
    const Tensor& index,
    std::variant<float, int> value,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
