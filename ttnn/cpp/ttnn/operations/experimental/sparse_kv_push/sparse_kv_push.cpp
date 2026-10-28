// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_kv_push.hpp"

#include "device/sparse_kv_push_device_operation.hpp"
#include "device/sparse_push_plan_device_operation.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

Tensor sparse_kv_push(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const Tensor& table,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    const GlobalSemaphore& ready_semaphore,
    const GlobalSemaphore& data_valid_semaphore,
    ::sparse_push::PushPart part,
    uint32_t num_links,
    uint32_t batch_index,
    const std::optional<Tensor>& batch_index_tensor,
    uint32_t batch_slot_num_layers,
    uint32_t batch_slot_layer_idx,
    const std::optional<CoreRangeSet>& core_grid) {
    return ttnn::prim::sparse_kv_push(
        input_tensor,
        output_tensor,
        table,
        stripe_chunk,
        max_rows,
        ready_semaphore,
        data_valid_semaphore,
        part,
        num_links,
        batch_index,
        batch_index_tensor,
        batch_slot_num_layers,
        batch_slot_layer_idx,
        core_grid);
}

std::vector<Tensor> sparse_push_plan(
    const Tensor& indices,
    uint32_t num_devices,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    uint32_t page_bytes,
    const std::optional<Tensor>& chunk_start) {
    return ttnn::prim::sparse_push_plan(indices, num_devices, stripe_chunk, max_rows, page_bytes, chunk_start);
}

}  // namespace ttnn::operations::experimental::sparse_kv_push
