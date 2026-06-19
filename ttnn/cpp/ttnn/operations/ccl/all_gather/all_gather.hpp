// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>  // for tt::tt_fabric::Topology

namespace ttnn {

ttnn::Tensor all_gather(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::Tensor>& persistent_output_tensor = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt,
    // The following args are deprecated and will be removed in a future update
    std::optional<uint32_t> num_links = std::nullopt,
    std::optional<tt::tt_fabric::Topology> topology = std::nullopt,
    std::optional<uint32_t> chunks_per_sync = std::nullopt,
    std::optional<uint32_t> num_workers_per_link = std::nullopt,
    std::optional<uint32_t> num_buffers_per_channel = std::nullopt,
    bool use_l1_small_for_semaphores = false);

}  // namespace ttnn
