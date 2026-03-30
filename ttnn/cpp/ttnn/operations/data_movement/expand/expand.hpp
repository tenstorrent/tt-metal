// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "ttnn/types.hpp"

namespace ttnn {
Tensor expand(
    const ttnn::Tensor& tensor,
    ttsl::Span<const int32_t> shape_vector,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);
}  // namespace ttnn
