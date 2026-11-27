// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_kv_push.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/operations/experimental/sparse_kv_push/common/sparse_kv_push_common.hpp"
#include "device/sparse_kv_push_device_operation.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

Tensor sparse_kv_push(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const Tensor& table,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    const GlobalSemaphore& ready_semaphore,
    const GlobalSemaphore& done_semaphore,
    SparseKvPushPart part,
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
        done_semaphore,
        part,
        num_links,
        batch_index,
        batch_index_tensor,
        batch_slot_num_layers,
        batch_slot_layer_idx,
        core_grid);
}

bool is_sparse_kv_push_supported(MeshDevice* mesh_device, uint32_t stripe_chunk, uint32_t max_rows) {
    const auto& mesh_shape = mesh_device->shape();
    // The push sends over all four torus links of every chip: each mesh axis wraps, with distinct neighbours.
    const bool torus = tt::tt_fabric::GetFabricConfig() == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY &&
                       mesh_shape.dims() == 2 && mesh_shape[0] >= 3 && mesh_shape[1] >= 3;
    const bool stripe = stripe_chunk > 0 && stripe_chunk % 32 == 0 && stripe_chunk <= ::sparse_push::kMaxStripeChunk &&
                        max_rows > 0 && max_rows % stripe_chunk == 0;
    return torus && mesh_shape.mesh_size() <= ::sparse_push::plan_table::kMaxDevices && stripe &&
           plan_fits(mesh_device, max_rows);
}

tt::tt_metal::TensorSpec sparse_kv_push_output_spec(
    MeshDevice* mesh_device,
    uint32_t distinct_rows,
    uint32_t max_rows,
    uint32_t stripe_chunk,
    uint32_t row_width,
    DataType dtype) {
    const tt::tt_metal::TensorLayout layout(
        dtype,
        tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM});
    // The push reads and writes whole DRAM-aligned row pages.
    const uint32_t page_bytes = tt::round_up(
        tt::tt_metal::TensorSpec(Shape({1, 1, 1, row_width}), layout).compute_page_size_bytes(),
        mesh_device->allocator()->get_alignment(BufferType::DRAM));
    const uint32_t rows = output_rows(mesh_device, distinct_rows, max_rows, stripe_chunk, page_bytes);
    return tt::tt_metal::TensorSpec(Shape({1, 1, rows, row_width}), layout);
}

Tensor sparse_kv_push_create_output_buffer(
    MeshDevice* mesh_device,
    uint32_t distinct_rows,
    uint32_t max_rows,
    uint32_t stripe_chunk,
    uint32_t row_width,
    DataType dtype) {
    auto buffer = create_device_tensor(
        sparse_kv_push_output_spec(mesh_device, distinct_rows, max_rows, stripe_chunk, row_width, dtype), mesh_device);
    buffer.update_tensor_topology(rows_over_every_chip(mesh_device->shape()));
    return buffer;
}

}  // namespace ttnn::operations::experimental::sparse_kv_push
