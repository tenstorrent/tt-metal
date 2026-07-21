// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::experimental {

ttnn::Tensor reduce_scatter_minimal_async(
    const ttnn::Tensor& input_tensor,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_output_buffers,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
    std::optional<uint32_t> num_links = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& intermediate_memory_config = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    std::optional<uint32_t> chunks_per_sync = std::nullopt,
    std::optional<uint32_t> num_workers_per_link = std::nullopt,
    std::optional<uint32_t> num_buffers_per_channel = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

// Allocates the intermediate persistent buffer for the contiguous ring reduce-scatter fast path.
//
// When reduce_scatter_minimal_async runs on the Ring topology with scatter dim != 0, the intermediate
// is a chunk-paged, row-major, interleaved-DRAM staging tensor (not the input-shaped tensor). Callers
// that want to reuse a persistent intermediate must allocate it with the exact layout the op expects;
// this helper does that by reusing the op's own sizing helper, so the returned tensor is guaranteed to
// match. Pass it as persistent_output_buffers[0]. `dim`, `topology`, `cluster_axis`, and
// `compute_kernel_config` must match the values passed to reduce_scatter_minimal_async.
//
// TT_FATALs if the configuration does not use the contiguous path (Ring + dim != 0); for the legacy
// path the intermediate has the input tensor's shape and can be allocated directly.
ttnn::Tensor reduce_scatter_minimal_async_create_intermediate_buffer(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental
