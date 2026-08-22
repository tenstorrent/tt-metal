// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "offset_cumsum.hpp"
#include "device/offset_cumsum_device_operation.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum {

std::array<ttnn::Tensor, 3> offset_cumsum(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    uint32_t num_links,
    uint32_t experts_per_chip,
    const ttnn::MemoryConfig& memory_config,
    bool use_l1_small_for_semaphores) {
    const auto& shape = input_tensor.logical_shape();
    uint32_t n_routed_experts = shape[-1];

    auto reshaped = ttnn::reshape(input_tensor, ttnn::Shape({1, n_routed_experts}));

    // A one-device cluster axis is a useful configuration for independent
    // expert-parallel groups (for example, one local-only group per mesh
    // column). all_gather deliberately rejects a ring of size one, and it is
    // also unnecessary here: the reshaped histogram is already the complete
    // [1, num_experts] input expected by the primitive.
    auto gathered = reshaped;
    if (::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis) > 1) {
        gathered = ttnn::all_gather(
            reshaped,
            /*dim=*/0,
            /*cluster_axis=*/cluster_axis,
            /*subdevice_id=*/std::nullopt,
            /*memory_config=*/memory_config,
            /*optional_output_tensor=*/std::nullopt,
            /*num_links=*/num_links,
            /*topology=*/std::nullopt,
            /*chunks_per_sync=*/std::nullopt,
            /*num_workers_per_link=*/std::nullopt,
            /*num_buffers_per_channel=*/std::nullopt,
            /*sub_core_grid=*/std::nullopt,
            /*use_l1_small_for_semaphores=*/use_l1_small_for_semaphores);
    }

    auto row_major = ttnn::to_layout(gathered, tt::tt_metal::Layout::ROW_MAJOR, std::nullopt, std::nullopt);

    return ttnn::prim::offset_cumsum(row_major, cluster_axis, experts_per_chip);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum
