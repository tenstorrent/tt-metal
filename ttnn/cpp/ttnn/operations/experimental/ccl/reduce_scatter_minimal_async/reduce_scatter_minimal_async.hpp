// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteReduceScatterMinimalAsync {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<std::vector<ttnn::Tensor>>& persistent_output_buffers,
        int32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
        std::optional<uint32_t> cluster_axis = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto reduce_scatter_minimal_async = ttnn::register_operation<
    "ttnn::experimental::reduce_scatter_minimal_async",
    ttnn::operations::experimental::ccl::ExecuteReduceScatterMinimalAsync>();

}  // namespace experimental
}  // namespace ttnn
