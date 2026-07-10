// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine {

ttnn::Tensor combine(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& dispatched_metadata,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& expert_region_offsets,
    uint32_t dispatch_group_size,
    uint32_t experts_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
    std::optional<uint32_t> cluster_axis = 0,
    std::optional<uint32_t> num_links = 1,
    std::optional<tt::tt_fabric::Topology> topology = tt::tt_fabric::Topology::Linear,
    bool init_zeros = true,
    bool use_l1_small_for_semaphores = false,
    bool use_fp8_combine = false,
    // Debug/experiment: place the combine cores on exactly this CoreRangeSet instead of
    // deriving them from `subdevice_id` via worker_cores(). Lets us run combine on an edge
    // (first/last row/column) core layout WITHOUT a sub-device.
    const std::optional<CoreRangeSet>& core_grid_override = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine

namespace ttnn {
using operations::experimental::deepseek_prefill::combine::combine;
}  // namespace ttnn
