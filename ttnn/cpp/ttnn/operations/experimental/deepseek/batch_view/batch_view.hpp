// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek {

// Creates a view of a single batch from a 3D tensor [b, M, N] -> [M, N]
// This is a zero-copy operation that creates metadata pointing to the same underlying buffer.
//
// Requirements:
// - Input must be a device tensor (DRAM interleaved, not sharded)
// - Input must be 3D with shape [b, M, N]
// - batch_index must be in range [0, b)
// - For TILE layout: M * N must be divisible by tiles_per_page (1024 for bfloat16, 512 for float32)
// - For ROW_MAJOR layout: always valid (page size = N * element_size)
ttnn::Tensor batch_view(const ttnn::Tensor& input_tensor, uint32_t batch_index);

}  // namespace ttnn::operations::experimental::deepseek
