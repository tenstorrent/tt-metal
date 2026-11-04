// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "device/kernels/sparse_push_constants.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

// Builds this chip's plan table for sparse_kv_push from its top-k indices. Returns {plan table, indices rewritten to
// compact-buffer slots}.
std::vector<Tensor> sparse_push_plan(
    const Tensor& indices,
    uint32_t num_devices,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    uint32_t page_bytes,
    const std::optional<Tensor>& chunk_start = std::nullopt);

// Owner chips push the cache rows each requester's top-k needs into that requester's compact output_tensor, as laid
// out by the all-gathered sparse_push_plan tables; `part` picks all of them, the prefix or the rest (see PushPart).
Tensor sparse_kv_push(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const Tensor& table,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    const GlobalSemaphore& ready_semaphore,
    const GlobalSemaphore& data_valid_semaphore,
    ::sparse_push::PushPart part = ::sparse_push::PushPart::kAll,
    uint32_t num_links = 2,
    uint32_t batch_index = 0,
    // Trace-safe: the slot is read on device as batch_index_tensor[0] * batch_slot_num_layers + batch_slot_layer_idx.
    const std::optional<Tensor>& batch_index_tensor = std::nullopt,
    uint32_t batch_slot_num_layers = 1,
    uint32_t batch_slot_layer_idx = 0,
    const std::optional<CoreRangeSet>& core_grid = std::nullopt);

}  // namespace ttnn::operations::experimental::sparse_kv_push

namespace ttnn::experimental {
using operations::experimental::sparse_kv_push::sparse_kv_push;
using operations::experimental::sparse_kv_push::sparse_push_plan;
}  // namespace ttnn::experimental
