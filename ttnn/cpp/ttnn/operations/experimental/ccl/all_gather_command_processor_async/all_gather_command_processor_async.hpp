// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteAllGatherCommandProcessorAsync {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int32_t dim,
        const GlobalSemaphore& multi_device_global_semaphore,
        const std::optional<ttnn::Tensor>& persistent_output_buffer = std::nullopt,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt);

    static std::vector<ttnn::Tensor> invoke(
        const std::vector<Tensor>& input_tensors,
        int32_t dim,
        const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
        const std::optional<ttnn::Tensor>& persistent_output_buffer = std::nullopt,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto all_gather_command_processor_async = ttnn::register_operation<
    "ttnn::experimental::all_gather_command_processor_async",
    ttnn::operations::experimental::ccl::ExecuteAllGatherCommandProcessorAsync>();

}  // namespace experimental
}  // namespace ttnn
