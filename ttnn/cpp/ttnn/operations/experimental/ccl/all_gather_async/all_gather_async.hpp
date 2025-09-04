// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteAllGatherAsync {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const int32_t dim,
        const std::optional<ttnn::Tensor>& persistent_output_buffer = std::nullopt,
        const std::optional<std::vector<GlobalSemaphore>>& multi_device_global_semaphore = std::nullopt,
        bool do_sync = false,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        bool use_optimal_ccl_for_llama = false,
        std::optional<uint32_t> chunks_per_sync = std::nullopt,
        std::optional<uint32_t> num_workers_per_link = std::nullopt,
        std::optional<uint32_t> num_buffers_per_channel = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto all_gather_async = ttnn::register_operation<
    "ttnn::experimental::all_gather_async",
    ttnn::operations::experimental::ccl::ExecuteAllGatherAsync>();

}  // namespace experimental
}  // namespace ttnn
