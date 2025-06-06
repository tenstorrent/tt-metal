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
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        ttnn::Tensor& persistent_intermediate_buffer,
        ttnn::Tensor& persistent_output_buffer,
        const int32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt);

    static std::vector<ttnn::Tensor> invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        const int32_t dim,
        const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const int32_t dim,
        const uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const ttnn::ccl::Topology topology,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        const std::optional<ttnn::Tensor>& persistent_output_tensor = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<size_t> num_preferred_links = std::nullopt,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt);

    static std::vector<ttnn::Tensor> invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        const int32_t dim,
        const uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const ttnn::ccl::Topology topology,
        const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
        const std::optional<ttnn::Tensor>& persistent_output_tensor = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<size_t> num_preferred_links = std::nullopt,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto all_gather_async = ttnn::register_operation<
    "ttnn::experimental::all_gather_async",
    ttnn::operations::experimental::ccl::ExecuteAllGatherAsync>();

}  // namespace experimental
}  // namespace ttnn
