// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/common.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteStridedAllGatherAsync {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<ttnn::Tensor>& persistent_output_buffer,
        int32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
        std::optional<uint32_t> tiles_per_chunk = std::nullopt,
        std::optional<uint32_t> num_workers_per_link = std::nullopt,
        std::optional<uint32_t> num_buffers_per_channel = std::nullopt);
};
}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto strided_all_gather_async = ttnn::register_operation<
    "ttnn::experimental::strided_all_gather_async",
    ttnn::operations::experimental::ccl::ExecuteStridedAllGatherAsync>();

}  // namespace experimental
}  // namespace ttnn
