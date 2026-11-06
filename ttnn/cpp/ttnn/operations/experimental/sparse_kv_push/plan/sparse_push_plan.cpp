// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_push_plan.hpp"

#include "device/sparse_push_plan_device_operation.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

std::vector<Tensor> sparse_push_plan(
    const Tensor& indices,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    uint32_t page_bytes,
    std::optional<uint32_t> chunk_start,
    const std::optional<Tensor>& chunk_start_tensor,
    uint32_t dense_below) {
    return ttnn::prim::sparse_push_plan(
        indices, stripe_chunk, max_rows, page_bytes, chunk_start, chunk_start_tensor, dense_below);
}

}  // namespace ttnn::operations::experimental::sparse_kv_push
