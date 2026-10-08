// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar {

// C = A + B for bfloat16, TILE layout, DRAM-interleaved tensors of the same shape, on a single node.
Tensor simple_add(
    const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::operations::experimental::quasar
