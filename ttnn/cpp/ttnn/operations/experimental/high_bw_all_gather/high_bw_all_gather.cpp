// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "high_bw_all_gather.hpp"

#include "device/high_bw_all_gather_device_operation.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather {

Tensor high_bw_all_gather(
    const Tensor& input_tensor,
    int32_t dim,
    const Tensor& output_tensor,
    uint32_t cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid,
    std::optional<uint32_t> num_links,
    std::optional<uint32_t> input_batch_index,
    std::optional<uint32_t> gathered_dim_size,
    const std::optional<Tensor>& input_batch_index_tensor,
    uint32_t batch_slot_num_layers,
    uint32_t batch_slot_layer_idx,
    const std::optional<Tensor>& gathered_prefix_tensor,
    uint32_t gathered_slab_global) {
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
        gathered_slab_global);
}

}  // namespace ttnn::operations::experimental::high_bw_all_gather
