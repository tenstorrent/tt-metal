// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn::experimental {

ttnn::Tensor neighbor_pad_async(
    const ttnn::Tensor& input_tensor,
    std::vector<int32_t> dim,
    std::vector<uint32_t> padding_left,
    std::vector<uint32_t> padding_right,
    const std::string& padding_mode,
    std::vector<uint32_t> cluster_axis,
    std::vector<GlobalSemaphore> neighbor_semaphore,
    std::vector<GlobalSemaphore> barrier_semaphore,
    const std::optional<std::vector<size_t>>& num_preferred_links = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<ttnn::ccl::Topology> topology = std::nullopt,
    const std::optional<ttnn::Tensor>& persistent_output_buffer = std::nullopt,
    const std::optional<GlobalSemaphore>& progress_semaphore = std::nullopt,
    uint32_t progress_t_batch_size = 0,
    bool fabric_only = false,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);

}  // namespace ttnn::experimental
