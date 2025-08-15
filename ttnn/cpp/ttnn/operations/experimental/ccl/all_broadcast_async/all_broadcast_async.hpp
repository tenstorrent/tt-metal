// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteAllBroadcastAsync {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,
        const GlobalSemaphore& multi_device_global_semaphore,
        const GlobalSemaphore& barrier_semaphore,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto all_broadcast_async = ttnn::register_operation<
    "ttnn::experimental::all_broadcast_async",
    ttnn::operations::experimental::ccl::ExecuteAllBroadcastAsync>();

}  // namespace experimental
}  // namespace ttnn
