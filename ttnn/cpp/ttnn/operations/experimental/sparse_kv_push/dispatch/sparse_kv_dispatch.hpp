// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

// The KV rows every chip's top-k needs, sent over sparse_dispatch_2d-style trees into `output_tensor` (the full-size
// gather buffer) at the rows' owner-major pages. `table` is this chip's owner-major plan table (sparse_push_plan with
// owner_major); `inbox` holds one slot per chip; `scratch` is sparse_kv_dispatch_scratch_spec's DRAM tensor. The four
// semaphores are persistent words on the core grid, zero between calls.
Tensor sparse_kv_dispatch(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const Tensor& table,
    const Tensor& inbox,
    const Tensor& scratch,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    const GlobalSemaphore& fence_semaphore,
    const GlobalSemaphore& arrived_semaphore,
    const GlobalSemaphore& credit_semaphore,
    const GlobalSemaphore& inbox_semaphore,
    uint32_t num_links = 2,
    uint32_t rows_per_packet = 0,
    uint32_t landing_slots = 16,
    uint32_t planners = 48,
    uint32_t batch_index = 0,
    const std::optional<Tensor>& batch_index_tensor = std::nullopt,
    uint32_t batch_slot_num_layers = 1,
    uint32_t batch_slot_layer_idx = 0,
    const std::optional<CoreRangeSet>& core_grid = std::nullopt,
    uint32_t lanes_per_link = 2,
    const std::optional<Tensor>& plan_indices = std::nullopt,
    std::optional<uint32_t> plan_chunk_start = std::nullopt,
    const std::optional<Tensor>& plan_chunk_start_tensor = std::nullopt,
    uint32_t part = 0,
    bool skip_exchange = false,
    const std::optional<Tensor>& rank_table = std::nullopt,
    uint32_t sp_axis = 0,
    uint32_t replay = 0,
    const std::optional<Tensor>& replay_buffer = std::nullopt,
    uint32_t replay_generation = 0,
    bool single_program = false);

// The scratch tensor sparse_kv_dispatch needs: [1, 1, pages, 1024] UINT32, interleaved DRAM, per chip.
tt::tt_metal::TensorSpec sparse_kv_dispatch_scratch_spec(
    MeshDevice* mesh_device,
    uint32_t max_rows,
    uint32_t stripe_chunk,
    uint32_t planners = 48,
    uint32_t num_links = 2,
    uint32_t lanes_per_link = 2);

// The routing record sparse_kv_dispatch's replay needs: [1, 1, pages, 1024] UINT32 (4 KB pages), interleaved DRAM,
// per chip, one per plan build that replays.
tt::tt_metal::TensorSpec sparse_kv_dispatch_replay_spec(
    MeshDevice* mesh_device, uint32_t max_rows, uint32_t stripe_chunk);

}  // namespace ttnn::operations::experimental::sparse_kv_push

namespace ttnn::experimental {
using operations::experimental::sparse_kv_push::sparse_kv_dispatch;
using operations::experimental::sparse_kv_push::sparse_kv_dispatch_replay_spec;
using operations::experimental::sparse_kv_push::sparse_kv_dispatch_scratch_spec;
}  // namespace ttnn::experimental
