// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

// Owner chips push the cache rows each requester's top-k needs into its compact output_tensor, as laid out by the
// all-gathered sparse_push_plan tables. part 1 pushes only the prefix
// (rows written by earlier chunks), part 2 the rest.
Tensor sparse_kv_push(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const Tensor& table,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    const GlobalSemaphore& ready_semaphore,
    const GlobalSemaphore& data_valid_semaphore,
    uint32_t part = 0,
    uint32_t num_links = 2,
    uint32_t batch_index = 0,
    // Trace-safe: the slot is read on device as batch_index_tensor[0] * batch_slot_num_layers + batch_slot_layer_idx.
    const std::optional<Tensor>& batch_index_tensor = std::nullopt,
    uint32_t batch_slot_num_layers = 1,
    uint32_t batch_slot_layer_idx = 0,
    const std::optional<CoreRangeSet>& core_grid = std::nullopt);

// Returns {plan table, indices rewritten to compact-buffer slots}.
std::vector<Tensor> sparse_push_plan(
    const Tensor& indices,
    uint32_t num_devices,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    uint32_t page_bytes,
    const std::optional<Tensor>& chunk_start = std::nullopt);

// Rows per packet and DRAM banks of the push's compact layout for row pages of page_bytes.
std::tuple<uint32_t, uint32_t> sparse_kv_push_packet_shape(MeshDevice* mesh_device, uint32_t page_bytes);

}  // namespace ttnn::operations::experimental::sparse_kv_push

namespace ttnn::experimental {
using operations::experimental::sparse_kv_push::sparse_kv_push;
using operations::experimental::sparse_kv_push::sparse_kv_push_packet_shape;
using operations::experimental::sparse_kv_push::sparse_push_plan;
}  // namespace ttnn::experimental
