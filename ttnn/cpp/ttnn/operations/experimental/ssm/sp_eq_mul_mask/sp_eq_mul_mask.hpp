// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/base_types.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>

namespace ttnn::experimental {

// Fused eq + mask: output = A if A == B else 0.
// Tile-elementwise; expects A and B to have identical shape, tile layout,
// and bfloat16 dtype (interleaved memory).
ttnn::Tensor sp_eq_mul_mask(
    const Tensor& a,
    const Tensor& b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<DataType> dtype = std::nullopt);

}  // namespace ttnn::experimental
