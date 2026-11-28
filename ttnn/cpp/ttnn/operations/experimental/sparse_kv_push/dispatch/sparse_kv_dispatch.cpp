// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_kv_dispatch.hpp"

#include "device/sparse_kv_dispatch_device_operation.hpp"
#include "device/sparse_kv_dispatch_layout.hpp"
#include "ttnn/operations/experimental/sparse_kv_push/common/sparse_kv_push_common.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

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
    uint32_t num_links,
    uint32_t rows_per_packet,
    uint32_t landing_slots,
    uint32_t planners,
    uint32_t batch_index,
    const std::optional<Tensor>& batch_index_tensor,
    uint32_t batch_slot_num_layers,
    uint32_t batch_slot_layer_idx,
    const std::optional<CoreRangeSet>& core_grid,
    uint32_t lanes_per_link,
    const std::optional<Tensor>& plan_indices,
    std::optional<uint32_t> plan_chunk_start,
    const std::optional<Tensor>& plan_chunk_start_tensor,
    uint32_t part,
    bool skip_exchange,
    const std::optional<Tensor>& rank_table,
    uint32_t sp_axis,
    uint32_t replay,
    const std::optional<Tensor>& replay_buffer,
    uint32_t replay_generation,
    bool single_program) {
    return ttnn::prim::sparse_kv_dispatch(
        input_tensor,
        output_tensor,
        table,
        inbox,
        scratch,
        stripe_chunk,
        max_rows,
        fence_semaphore,
        arrived_semaphore,
        credit_semaphore,
        inbox_semaphore,
        num_links,
        rows_per_packet,
        landing_slots,
        planners,
        batch_index,
        batch_index_tensor,
        batch_slot_num_layers,
        batch_slot_layer_idx,
        core_grid,
        lanes_per_link,
        plan_indices,
        plan_chunk_start,
        plan_chunk_start_tensor,
        part,
        skip_exchange,
        rank_table,
        sp_axis,
        replay,
        replay_buffer,
        replay_generation,
        single_program);
}

tt::tt_metal::TensorSpec sparse_kv_dispatch_scratch_spec(
    MeshDevice* mesh_device,
    uint32_t max_rows,
    uint32_t stripe_chunk,
    uint32_t planners,
    uint32_t num_links,
    uint32_t lanes_per_link) {
    const uint32_t n = mesh_device->shape().mesh_size();
    validate_stripe_chunk(stripe_chunk, max_rows);
    TT_FATAL(planners >= 1 && planners <= 64, "sparse_kv_dispatch planners must be 1..64");
    TT_FATAL(num_links >= 1 && num_links <= 2, "sparse_kv_dispatch supports 1 or 2 links");
    TT_FATAL(
        lanes_per_link >= 1 && lanes_per_link <= kvd::kMaxLanes / num_links,
        "sparse_kv_dispatch supports at most {} lanes",
        kvd::kMaxLanes);
    const auto mesh = kv_dispatch_mesh(mesh_device, num_links);
    const auto s =
        KvDispatchScratch::make(max_rows, stripe_chunk, n, planners, mesh.gmax(), num_links * lanes_per_link);
    const tt::tt_metal::TensorLayout layout(
        DataType::UINT32,
        tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM});
    return tt::tt_metal::TensorSpec(Shape({1, 1, s.pages, kvd::kScratchPageBytes / 4}), layout);
}

tt::tt_metal::TensorSpec sparse_kv_dispatch_replay_spec(
    MeshDevice* mesh_device, uint32_t max_rows, uint32_t stripe_chunk) {
    validate_stripe_chunk(stripe_chunk, max_rows);
    const uint64_t entries =
        sparse_kv_dispatch_replay_entries(max_rows, stripe_chunk, mesh_device->shape().mesh_size());
    const uint32_t pages = static_cast<uint32_t>((entries * 16 + 4095) / 4096);
    const tt::tt_metal::TensorLayout layout(
        DataType::UINT32,
        tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM});
    return tt::tt_metal::TensorSpec(Shape({1, 1, pages, 1024}), layout);
}

}  // namespace ttnn::operations::experimental::sparse_kv_push
