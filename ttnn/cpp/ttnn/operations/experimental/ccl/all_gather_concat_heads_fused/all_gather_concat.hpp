// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteAllGatherConcat {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        ttnn::Tensor& buffer_tensor,
        const int32_t dim,
        const uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const GlobalSemaphore& global_semaphore,
        const uint32_t num_heads,
        const ttnn::MemoryConfig& memory_config,
        const std::optional<uint32_t> num_links = std::nullopt,
        const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto all_gather_concat = ttnn::register_operation<
    "ttnn::experimental::all_gather_concat",
    ttnn::operations::experimental::ccl::ExecuteAllGatherConcat>();

}  // namespace experimental
}  // namespace ttnn
