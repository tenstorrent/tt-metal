// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Experimental fork of all_gather_async with a separate device op and kernel paths under
// all_gather_ce/device/kernels/ for cross-entropy / fused work. Host routing matches all_gather_async
// (including composite_all_gather when applicable); the minimal path uses all_gather_ce kernels.

ttnn::Tensor all_gather_ce(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    std::optional<uint32_t> num_links = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
    bool use_optimal_ccl_for_llama = false,
    const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
    bool reverse_order = false,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt,
    std::optional<uint32_t> num_workers_per_link = std::nullopt,
    std::optional<uint32_t> num_buffers_per_channel = std::nullopt);

ttnn::Tensor all_gather_ce(
    const ttnn::Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    std::optional<uint32_t> num_links = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    bool use_optimal_ccl_for_llama = false,
    const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
    bool use_all_gather_async_via_broadcast = false,
    std::optional<uint32_t> chunks_per_sync = std::nullopt,
    std::optional<uint32_t> num_workers_per_link = std::nullopt,
    std::optional<uint32_t> num_buffers_per_channel = std::nullopt,
    bool reverse_order = false,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

std::vector<ttnn::Tensor> all_gather_ce(
    const std::vector<ttnn::Tensor>& input_tensors,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    int32_t dim,
    const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
    std::optional<uint32_t> num_links = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    bool use_optimal_ccl_for_llama = false,
    const std::optional<std::vector<GlobalSemaphore>>& barrier_semaphore = std::nullopt,
    std::optional<uint32_t> chunks_per_sync = std::nullopt,
    std::optional<uint32_t> num_workers_per_link = std::nullopt,
    std::optional<uint32_t> num_buffers_per_channel = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

ttnn::Tensor all_gather_ce(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_tensor = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<size_t> num_preferred_links = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
    bool use_optimal_ccl_for_llama = false,
    bool use_all_gather_async_via_broadcast = false,
    const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
    bool reverse_order = false,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt,
    std::optional<uint32_t> num_workers_per_link = std::nullopt,
    std::optional<uint32_t> num_buffers_per_channel = std::nullopt);

}  // namespace ttnn::experimental
