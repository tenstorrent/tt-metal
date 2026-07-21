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
    std::optional<uint32_t> cluster_axis = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::Tensor>& persistent_output_tensor = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt,
    // The following args are deprecated and will be removed in a future update
    std::optional<uint32_t> num_links = std::nullopt,
    std::optional<tt::tt_fabric::Topology> topology = std::nullopt,
    std::optional<uint32_t> chunks_per_sync = std::nullopt,
    std::optional<uint32_t> num_workers_per_link = std::nullopt,
    std::optional<uint32_t> num_buffers_per_channel = std::nullopt,
    bool use_l1_small_for_semaphores = false);

// Deprecated C++ entry point (legacy argument order).
// This is intentionally a function template to resolve overload ambiguity with the above.
template <typename = void>
[[deprecated(
    "ttnn::all_gather arg order has changed. This legacy overload will be removed after 01-08-2026; "
    "migrate to the new arg order: input_tensor, dim, cluster_axis, memory_config, "
    "persistent_output_tensor, subdevice_id, sub_core_grid, num_links, topology, chunks_per_sync, "
    "num_workers_per_link, num_buffers_per_channel, use_l1_small_for_semaphores.")]]
ttnn::Tensor all_gather(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt,
    std::optional<uint32_t> num_links = std::nullopt,
    std::optional<tt::tt_fabric::Topology> topology = std::nullopt,
    std::optional<uint32_t> chunks_per_sync = std::nullopt,
    std::optional<uint32_t> num_workers_per_link = std::nullopt,
    std::optional<uint32_t> num_buffers_per_channel = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt,
    bool use_l1_small_for_semaphores = false) {
    // Forward to the correct overload
    return all_gather(
        input_tensor,
        dim,
        cluster_axis,
        memory_config,
        optional_output_tensor,
        subdevice_id,
        sub_core_grid,
        num_links,
        topology,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel,
        use_l1_small_for_semaphores);
}

}  // namespace ttnn
