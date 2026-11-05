// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

// Builds this chip's plan table for sparse_kv_push from its top-k indices. Returns {plan table, indices rewritten to
// compact-buffer slots}. With the chunk start below dense_below (0: off), the indices address a dense gather into the
// compact buffer instead and the table lists no rows to push.
std::vector<Tensor> sparse_push_plan(
    const Tensor& indices,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    uint32_t page_bytes,
    std::optional<uint32_t> chunk_start = std::nullopt,
    const std::optional<Tensor>& chunk_start_tensor = std::nullopt,
    uint32_t dense_below = 0);

}  // namespace ttnn::operations::experimental::sparse_kv_push

namespace ttnn::experimental {
using operations::experimental::sparse_kv_push::sparse_push_plan;
}  // namespace ttnn::experimental
