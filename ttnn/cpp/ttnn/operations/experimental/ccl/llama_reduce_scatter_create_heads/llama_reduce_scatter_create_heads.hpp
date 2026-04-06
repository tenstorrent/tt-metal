// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> llama_rs_create_heads(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& intermediate_packet_buffer,
    int32_t dim,
    const GlobalSemaphore& cross_device_semaphore,
    const tt::tt_metal::SubDeviceId& subdevice_id,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    std::optional<uint32_t> num_links,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& qkv_memory_config = std::nullopt,
    bool use_noc1_only = false,
    bool use_optimal_ccl_for_llama = false);

}  // namespace ttnn::experimental
