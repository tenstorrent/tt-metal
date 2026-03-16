// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::expermental {

ttnn::Tensor llama_reduce_scatter(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& intermediate_packet_buffer,
    int32_t dim,
    const GlobalSemaphore& cross_device_semaphore,
    const tt::tt_metal::SubDeviceId& subdevice_id,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear,
    bool use_noc1_only = false);

}  // namespace ttnn::expermental
