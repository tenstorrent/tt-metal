// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "sparse_kv_push_types.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

// Owner chips push the cache rows each requester's top-k needs into that requester's compact output_tensor, as laid
// out by the all-gathered sparse_push_plan tables; `part` picks all of them, the prefix or the rest (see
// SparseKvPushPart).
Tensor sparse_kv_push(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const Tensor& table,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    const GlobalSemaphore& ready_semaphore,
    const GlobalSemaphore& done_semaphore,
    SparseKvPushPart part = SparseKvPushPart::kAll,
    uint32_t num_links = 2,
    uint32_t batch_index = 0,
    // Trace-safe: the slot is read on device as batch_index_tensor[0] * batch_slot_num_layers + batch_slot_layer_idx.
    const std::optional<Tensor>& batch_index_tensor = std::nullopt,
    uint32_t batch_slot_num_layers = 1,
    uint32_t batch_slot_layer_idx = 0,
    const std::optional<CoreRangeSet>& core_grid = std::nullopt);

// Whether the push and its plan can serve a cache of max_rows rows in blocks of stripe_chunk rows on this mesh: a 2D
// fabric (FABRIC_2D or a torus on either or both axes) on a 2D mesh of 2 to 32 chips, and a cache geometry the plan
// table and the plan's reduce fit. When false the ops would raise, so callers gather the whole prefix instead.
bool is_sparse_kv_push_supported(MeshDevice* mesh_device, uint32_t stripe_chunk, uint32_t max_rows);

// The compact buffer sparse_kv_push writes for up to distinct_rows needed rows per chip (a requester's query rows x
// top-k bounds them), for a cache of max_rows rows in blocks of stripe_chunk with rows of row_width dtype elements:
// [1, 1, rows, row_width] ROW_MAJOR interleaved DRAM.
tt::tt_metal::TensorSpec sparse_kv_push_output_spec(
    MeshDevice* mesh_device,
    uint32_t distinct_rows,
    uint32_t max_rows,
    uint32_t stripe_chunk,
    uint32_t row_width,
    DataType dtype);

// Allocates that compact buffer, labelled sharded over its rows across every chip (each chip's holds the rows its own
// top-k needs). Contents are undefined until a push writes them; only pushed rows are read.
Tensor sparse_kv_push_create_output_buffer(
    MeshDevice* mesh_device,
    uint32_t distinct_rows,
    uint32_t max_rows,
    uint32_t stripe_chunk,
    uint32_t row_width,
    DataType dtype);

}  // namespace ttnn::operations::experimental::sparse_kv_push

namespace ttnn::experimental {
using operations::experimental::sparse_kv_push::is_sparse_kv_push_supported;
using operations::experimental::sparse_kv_push::sparse_kv_push;
using operations::experimental::sparse_kv_push::sparse_kv_push_create_output_buffer;
using operations::experimental::sparse_kv_push::sparse_kv_push_output_spec;
}  // namespace ttnn::experimental
