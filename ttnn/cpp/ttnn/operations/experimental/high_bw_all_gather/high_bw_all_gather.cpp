// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "high_bw_all_gather.hpp"

#include "device/high_bw_all_gather_device_operation.hpp"
#include "device/sparse_push_plan_device_operation.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather {

Tensor high_bw_all_gather(
    const Tensor& input_tensor,
    int32_t dim,
    const Tensor& output_tensor,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid,
    std::optional<uint32_t> num_links,
    std::optional<uint32_t> input_batch_index,
    std::optional<uint32_t> gathered_dim_size,
    const std::optional<Tensor>& input_batch_index_tensor,
    uint32_t batch_slot_num_layers,
    uint32_t batch_slot_layer_idx,
    const std::optional<Tensor>& gathered_prefix_tensor,
    uint32_t gathered_slab_global,
    const std::optional<GlobalSemaphore>& ready_semaphore,
    const std::optional<GlobalSemaphore>& data_valid_semaphore,
    const std::optional<Tensor>& push_table,
    uint32_t push_stripe_chunk,
    uint32_t push_max_rows,
    uint32_t push_part) {
    return ttnn::prim::high_bw_all_gather(
        input_tensor,
        output_tensor,
        dim,
        cluster_axis,
        subdevice_id,
        sub_core_grid,
        num_links,
        input_batch_index,
        gathered_dim_size,
        input_batch_index_tensor,
        batch_slot_num_layers,
        batch_slot_layer_idx,
        gathered_prefix_tensor,
        gathered_slab_global,
        ready_semaphore,
        data_valid_semaphore,
        push_table,
        push_stripe_chunk,
        push_max_rows,
        push_part);
}

std::vector<Tensor> sparse_push_plan(
    const Tensor& indices,
    uint32_t num_devices,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    uint32_t pages_per_packet,
    uint32_t banks,
    const std::optional<Tensor>& chunk_start) {
    return ttnn::prim::sparse_push_plan(
        indices, num_devices, stripe_chunk, max_rows, pages_per_packet, banks, chunk_start);
}

}  // namespace ttnn::operations::experimental::high_bw_all_gather
