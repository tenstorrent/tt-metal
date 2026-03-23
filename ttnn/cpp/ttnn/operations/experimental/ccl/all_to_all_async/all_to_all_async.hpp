// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::experimental {

ttnn::Tensor all_to_all_async(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& persistent_intermediate_buffer,
    ttnn::Tensor& persistent_output_buffer,
    int32_t in_dim,
    int32_t out_dim,
    const GlobalSemaphore& multi_device_global_semaphore,
    uint32_t num_links = 1,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt);

}  // namespace ttnn::experimental
