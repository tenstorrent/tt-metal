// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/types.hpp"

namespace ttnn {

// Descriptor-based port of ttnn::index_fill (Phase 1 of the descriptor-migration recipe, #42392).
// Same contract as ttnn::index_fill; exists side by side with it until Phase 3 replaces the old
// program factory and deletes this operation.
Tensor index_fill_new(
    const Tensor& input,
    uint32_t dim,
    const Tensor& index,
    std::variant<float, int> value,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
