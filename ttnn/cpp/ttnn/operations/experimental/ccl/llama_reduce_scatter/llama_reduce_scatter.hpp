// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "cpp/ttnn/global_semaphore.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteLlamaReduceScatter {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        ttnn::Tensor& intermediate_packet_buffer,
        const int32_t dim,
        const global_semaphore::MultiDeviceGlobalSemaphore& cross_device_semaphore,
        const tt::tt_metal::SubDeviceId& subdevice_id,
        const uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const uint32_t num_links,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {
constexpr auto llama_reduce_scatter = ttnn::register_operation<
    "ttnn::experimental::llama_reduce_scatter",
    ttnn::operations::experimental::ccl::ExecuteLlamaReduceScatter>();
}  // namespace experimental

}  // namespace ttnn
