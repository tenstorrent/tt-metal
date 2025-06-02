// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/fabric_edm_types.hpp>

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteLlamaReduceScatter {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        ttnn::Tensor& intermediate_packet_buffer,
        const int32_t dim,
        const GlobalSemaphore& cross_device_semaphore,
        const tt::tt_metal::SubDeviceId& subdevice_id,
        const uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const uint32_t num_links,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear);
};

}  // namespace operations::experimental::ccl

namespace experimental {
constexpr auto llama_reduce_scatter = ttnn::register_operation<
    "ttnn::experimental::llama_reduce_scatter",
    ttnn::operations::experimental::ccl::ExecuteLlamaReduceScatter>();
}  // namespace experimental

}  // namespace ttnn
