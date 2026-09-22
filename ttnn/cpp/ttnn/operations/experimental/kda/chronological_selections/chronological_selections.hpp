// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/tensor/tensor.hpp"
namespace ttnn::experimental::kda {
Tensor chronological_selections(
    const Tensor& actual_start,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    uint32_t batch_heads,
    uint32_t key_dim,
    uint32_t value_dim);
}
