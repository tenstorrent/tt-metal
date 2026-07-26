// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "high_bw_all_gather.hpp"

#include "device/high_bw_all_gather_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::high_bw_all_gather {

Tensor high_bw_all_gather(
    const Tensor& input_tensor,
    int32_t dim,
    const Tensor& output_tensor,
    uint32_t cluster_axis,
    const GlobalSemaphore& data_valid_semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid,
    std::optional<uint32_t> num_links) {
    return ttnn::prim::high_bw_all_gather(
        input_tensor, output_tensor, dim, cluster_axis, data_valid_semaphore, subdevice_id, sub_core_grid, num_links);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::high_bw_all_gather
